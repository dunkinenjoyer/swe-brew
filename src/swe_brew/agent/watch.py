"""Live fleet progress + health monitor.

Reads the JSONL telemetry files that :mod:`enhance` writes and re-renders
a health dashboard on your terminal without touching the running process.

Data model
----------
Each ``telemetry.jsonl`` file corresponds to one **attempt** of a case.
Two levels:

  * :class:`AttemptState` replays a single ``telemetry.jsonl`` file.
    Carries an ``attempt_label`` parsed from the path (``attempt-NNN``
    or ``single``) so panel titles distinguish attempts of the same case.
  * :class:`CaseGroup` groups the attempts sharing a ``case_id``.
    Fleet-level aggregates (done / active / queued) count CASES, not
    attempts. When any attempt is running, :meth:`_active_attempt`
    picks the freshest one so freshness + LLM stats reflect the
    attempt currently emitting events.

What it shows
-------------
* Fleet header: wall clock, aggregate tokens + cost, 5-min burn rate,
  case-level status counts.
* Per-attempt row: 6-stage completion + current activity annotation.
* LLM / tool / recent-turns panels: scoped to the freshest running
  attempt so a completed sibling doesn't pollute the panel.
* Verify matrix: 6-way ``{base|gold|pred} × {pre|post}`` progress for
  the attempt most recently in verify.
* Anomaly panel: stalls, error flapping, LLM latency spikes.
* Trial history + submitted diffs.

Why a separate viewer. When the fleet is fired in the background,
Rich detects the non-TTY stdout and skips its live animation. Rather
than touch the running process, this tool owns its own TTY — the same
pattern as ``tail -F``-style monitors — and multiple viewers can attach
concurrently.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
)
from rich.table import Table
from rich.text import Text

__all__ = ["watch", "CaseState"]


_STAGE_ORDER = ("snapshot", "sandbox", "setup", "organize", "generate", "verify")
_VERIFY_CONFIGS = ("base_pre", "gold_pre", "pred_pre", "base_post", "gold_post", "pred_post")
_MAX_RECENT_TURNS = 10
_LATENCY_WINDOW = 20                # rolling window for LLM latency stats

# Context-aware stall thresholds. Maven legitimately spends 5-15 min on
# a test-compile on a big project, and a slow LLM reasoning turn can
# silently think for minutes before emitting; we bias toward false
# negatives over false positives.
_STALL_SECONDS_GENERIC = 600
_STALL_SECONDS_MVN     = 900        # mvn_test / mvn_test_compile / mvn_install
_STALL_SECONDS_VERIFY  = 1800       # single verify config-run
_BURN_WINDOW_SEC = 300              # token burn-rate rolling window

# DeepSeek V4 Pro high-effort public pricing: $0.43/M input, $0.87/M output.
# Override with SWE_BREW_COST_PER_1K_INPUT / SWE_BREW_COST_PER_1K_OUTPUT if
# your model gateway route has different billing.
_COST_PER_1K_INPUT = float(os.environ.get("SWE_BREW_COST_PER_1K_INPUT", "0.00043"))
_COST_PER_1K_OUTPUT = float(os.environ.get("SWE_BREW_COST_PER_1K_OUTPUT", "0.00087"))


def _parse_ts(ts: str) -> float:
    """Parse ISO-8601 → epoch seconds. Tolerant of trailing 'Z' + microseconds."""
    if not ts:
        return 0.0
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0


_ATTEMPT_DIR_RE = re.compile(r"^enhanced-attempt-(\d+)$")


def parse_case_and_attempt(telemetry_path: Path) -> tuple[str, str]:
    """Derive ``(case_id, attempt_label)`` from a telemetry.jsonl path.

    The path's ``.parent`` (``enhanced`` or ``enhanced-attempt-NNN``)
    carries the attempt identity; ``.parent.parent`` is the case root.

    Returns
    -------
    case_id : str
        The case root directory name (e.g. ``case_XXX_repo_YYY``).
    attempt_label : str
        ``"attempt-NNN"`` for pass@k paths, ``"single"`` for direct
        single-attempt paths under ``.../enhanced/``.
    """
    parent = telemetry_path.parent
    case_id = parent.parent.name
    m = _ATTEMPT_DIR_RE.match(parent.name)
    if m:
        attempt_label = f"attempt-{int(m.group(1)):03d}"
    else:
        attempt_label = "single"
    return case_id, attempt_label


@dataclass
class CaseState:
    """Reconstructed progress + health state for ONE ATTEMPT of a case.

    An instance models a single attempt (one ``telemetry.jsonl``);
    fleet-level aggregation across attempts of the same case lives in
    :class:`CaseGroup` below. The name is kept for continuity with
    downstream callers even though semantically each instance is an
    attempt, not a case.

    Everything is state-derived — replay the event stream and update
    counters. No side effects, no state stored between replays.
    """

    case_id: str
    telemetry_path: Path
    # ``attempt-NNN`` for pass@k, ``single`` for a direct-``enhanced``
    # single-attempt run. Rendered into panel titles + progress rows.
    attempt_label: str = "single"
    # ─── stage tracking ─────────────────────────────────────────────────
    stages_done: set[str] = field(default_factory=set)
    current_stage: str = ""
    stage_started_at_epoch: float = 0.0
    # ─── agent-loop position ────────────────────────────────────────────
    trial: int = 0
    max_trials: int = 0
    step: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    # ─── tool health ────────────────────────────────────────────────────
    tool_counter: Counter = field(default_factory=Counter)                        # {kind: n}
    tool_exit_ok: Counter = field(default_factory=Counter)                        # {kind: n_ok}
    tool_exit_err: Counter = field(default_factory=Counter)                       # {kind: n_err}
    parser_drift: int = 0                                                          # __none__ turns
    # ─── LLM health ─────────────────────────────────────────────────────
    llm_model: str = ""
    llm_calls_ok: int = 0
    llm_calls_err: int = 0
    llm_last_error: str = ""
    llm_latency_window: deque = field(default_factory=lambda: deque(maxlen=_LATENCY_WINDOW))
    llm_latency_derived: bool = False    # True if latency is derived from agent.turn deltas
    # Latches True the first time an authoritative ``llm.call.end``
    # event lands. Once set, derived samples stop being appended so
    # authoritative and derived measurements never mix in the window.
    _llm_authoritative_seen: bool = False
    # ─── recent activity ────────────────────────────────────────────────
    recent_turns: deque = field(default_factory=lambda: deque(maxlen=_MAX_RECENT_TURNS))
    last_event_epoch: float = 0.0
    _prev_turn_epoch: float = 0.0
    _prev_turn_tool_elapsed: float = 0.0
    # ─── submit + verify ────────────────────────────────────────────────
    submitted: bool = False
    submitted_diff_chars: int = 0
    submitted_at_epoch: float = 0.0
    verify_runs: dict[str, dict] = field(default_factory=dict)                    # cfg → {tests, exit, elapsed_s, running}
    verify_started: bool = False
    gate: dict | None = None
    # E4 (docs/DESIGN §7): per-stage per-trial history. Populated from
    # ``agent.trial.begin`` + ``agent.submit`` events; each entry has
    # {stage, trial, submitted, steps_used, ended_at_epoch}.
    trial_history: list[dict] = field(default_factory=list)
    _current_stage_role: str = ""       # tracks role for the currently-active generate stage
    _in_flight_tool: str = ""           # action_kind of tool currently in execute_action; "" when idle
    # ─── bookkeeping ────────────────────────────────────────────────────
    _fh_offset: int = 0
    done: bool = False
    started_at_epoch: float = 0.0        # first event ts

    # ─── replay ─────────────────────────────────────────────────────────

    def refresh_from_file(self) -> None:
        try:
            size = self.telemetry_path.stat().st_size
        except FileNotFoundError:
            return
        if size <= self._fh_offset:
            return
        with self.telemetry_path.open("r", encoding="utf-8") as f:
            f.seek(self._fh_offset)
            chunk = f.read()
            self._fh_offset = f.tell()
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._apply(rec)

    def _apply(self, rec: dict) -> None:
        ev = rec.get("event", "")
        ts_epoch = _parse_ts(rec.get("ts", ""))
        if ts_epoch > 0:
            self.last_event_epoch = ts_epoch
            if self.started_at_epoch == 0.0:
                self.started_at_epoch = ts_epoch

        if ev == "stage.begin":
            self.current_stage = rec.get("stage_name", "")
            self.stage_started_at_epoch = ts_epoch
        elif ev == "stage.end":
            name = rec.get("stage_name", "")
            if rec.get("ok"):
                self.stages_done.add(name)
            # Record a "trial" summary for this stage if it produced turns.
            # For non-agent stages (snapshot, sandbox) this is a no-op.
            self.current_stage = ""
        elif ev == "agent.trial.begin":
            # Close out the previous trial if it's still open (didn't submit).
            # BUT: don't double-log if the previous trial already ended via
            # a submit (the agent.submit branch below already added its
            # entry to trial_history and reset self.trial=0). This was the
            # "phantom setup 1 exhausted" bug — organize's trial.begin was
            # incorrectly closing an already-closed setup trial.
            if self.trial > 0 and self._current_stage_role:
                self.trial_history.append({
                    "stage":       self._current_stage_role,
                    "trial":       self.trial,
                    "submitted":   False,
                    "steps_used":  self.step,
                    "ended_at":    ts_epoch,
                    "note":        "trial exhausted or superseded",
                })
            self.trial = rec.get("trial", 0)
            self.max_trials = rec.get("max_trials", 0)
            self.step = 0
            self._current_stage_role = rec.get("role", self.current_stage) or self.current_stage
        elif ev == "agent.turn":
            self._apply_turn(rec, ts_epoch)
            role_hint = rec.get("role")
            if role_hint:
                self._current_stage_role = role_hint
            # Tool has returned — clear the in-flight marker so the
            # next stall check falls back to the generic threshold.
            self._in_flight_tool = ""
        elif ev == "agent.submit":
            # Only mark case-level "submitted" if this was the test_gen
            # agent — setup and organize agents also emit agent.submit
            # for their own terminal signals, and we don't want the
            # dashboard's "verify next" status to fire when setup submits.
            role = rec.get("role") or self._current_stage_role
            if role == "test_gen":
                self.submitted = True
                self.submitted_diff_chars = rec.get("payload_chars", rec.get("diff_chars", 0))
                self.submitted_at_epoch = ts_epoch
            # Close the trial in history — this trial succeeded via submit.
            self.trial_history.append({
                "stage":       role,
                "trial":       rec.get("trial", self.trial),
                "submitted":   True,
                "steps_used":  rec.get("step", self.step),
                "ended_at":    ts_epoch,
                "note":        f"submitted {rec.get('payload_chars', rec.get('diff_chars', 0)):,} chars",
            })
            # Reset the trial-in-progress marker so the NEXT
            # ``agent.trial.begin`` (typically fired by the next
            # stage's agent) doesn't spuriously log an "exhausted or
            # superseded" entry for the just-completed trial.
            self.trial = 0
            self._current_stage_role = ""
        elif ev == "llm.call.begin":
            self.llm_model = rec.get("model", self.llm_model)
        elif ev == "llm.call.end":
            self._apply_llm_call_end(rec)
        elif ev in ("agent.tool.begin", "agent.tool.heartbeat"):
            # Track in-flight tool for context-aware stall selection.
            # ``agent.tool.begin`` sets the tool; heartbeats refresh
            # ``last_event_epoch`` so a long legit tool doesn't trip
            # the stall alert.
            kind = rec.get("action_kind", "")
            if kind:
                self._in_flight_tool = kind
        elif ev == "agent.tool.error":
            # ``execute_action`` raised. Clear the in-flight marker so
            # subsequent silence falls back to the generic stall
            # threshold instead of the long-tool exemption.
            self._in_flight_tool = ""
        elif ev == "agent.trial.timeout":
            # Wall-clock watchdog fired. Terminal for this trial but not
            # necessarily for the case (higher trials may follow).
            self.trial_history.append({
                "stage":       self._current_stage_role,
                "trial":       rec.get("trial", self.trial),
                "submitted":   False,
                "steps_used":  rec.get("step", self.step),
                "ended_at":    ts_epoch,
                "note":        f"trial.timeout: {rec.get('reason', 'wall-clock')}",
            })
            self.trial = 0
            self._in_flight_tool = ""
        elif ev == "verify.run.begin":
            cfg = rec.get("configuration", "")
            if cfg:
                self.verify_runs.setdefault(cfg, {})["running"] = True
                self.verify_started = True
        elif ev == "verify.run.end":
            cfg = rec.get("configuration", "")
            if cfg:
                self.verify_runs[cfg] = {
                    "tests": rec.get("tests_count", 0),
                    "exit": rec.get("exit_code"),
                    "elapsed_s": rec.get("elapsed_s", 0.0),
                    "running": False,
                }
                self.verify_started = True
        elif ev == "verify.gate":
            self.gate = rec
        elif ev == "telemetry.close":
            self.done = True
            # Reset ``verify_started`` so a finished attempt doesn't keep
            # qualifying for the "running attempts" panel forever.
            self.verify_started = False
            self._in_flight_tool = ""

    def _apply_turn(self, rec: dict, ts_epoch: float) -> None:
        self.trial = rec.get("trial", self.trial)
        self.step = rec.get("step", self.step)
        self.tokens_in = rec.get("cost_input_tokens_cumulative", self.tokens_in)
        self.tokens_out = rec.get("cost_output_tokens_cumulative", self.tokens_out)
        kind = rec.get("action_kind", "?")
        exit_code = rec.get("exit_code", 1)
        tool_elapsed = rec.get("elapsed_s", 0.0) or 0.0

        self.tool_counter[kind] += 1
        if kind == "__none__":
            self.parser_drift += 1
        elif exit_code == 0:
            self.tool_exit_ok[kind] += 1
        else:
            self.tool_exit_err[kind] += 1
        self.recent_turns.append(rec)

        # Derive an LLM latency sample only when no authoritative
        # ``llm.call.end`` event has landed yet. Once the first
        # authoritative sample arrives, ``_llm_authoritative_seen``
        # latches True and derived samples stop accumulating.
        if not self._llm_authoritative_seen and ts_epoch > 0:
            if self._prev_turn_epoch > 0:
                lat = max(0.0, ts_epoch - self._prev_turn_epoch - self._prev_turn_tool_elapsed)
                if 0.1 < lat < 600:
                    self.llm_latency_window.append(lat)
                    self.llm_latency_derived = True
            elif self.stage_started_at_epoch > 0:
                lat = max(0.0, ts_epoch - self.stage_started_at_epoch)
                if 0.1 < lat < 600:
                    self.llm_latency_window.append(lat)
                    self.llm_latency_derived = True
        self._prev_turn_epoch = ts_epoch
        self._prev_turn_tool_elapsed = tool_elapsed

    def _apply_llm_call_end(self, rec: dict) -> None:
        # Authoritative latency source. The first authoritative sample
        # with a valid ``elapsed_s`` retires the derived samples we'd
        # been accumulating; subsequent authoritative samples extend
        # the window.
        elapsed = rec.get("elapsed_s", 0.0) or 0.0
        if elapsed > 0:
            if not self._llm_authoritative_seen:
                self.llm_latency_window.clear()
                self._llm_authoritative_seen = True
                self.llm_latency_derived = False
            self.llm_latency_window.append(elapsed)
        # Model can arrive on either llm.call.begin or llm.call.end.
        self.llm_model = rec.get("model", self.llm_model) or self.llm_model
        if rec.get("ok"):
            self.llm_calls_ok += 1
        else:
            self.llm_calls_err += 1
            self.llm_last_error = rec.get("error", "unknown")

    # ─── derived helpers ─────────────────────────────────────────────────

    def stage_progress_pct(self) -> float:
        """Return percent complete, including partial progress in the
        current stage.

        Stage completion alone is too coarse for long setup/organize/
        generate/verify phases. This estimate interpolates from trial/step
        or verify-run progress so the bar moves between stage boundaries.
        """
        if self.done:
            return 100.0
        total_stages = len(_STAGE_ORDER)
        stage_weight = 100.0 / total_stages

        base = 0.0
        if not self.current_stage:
            return stage_weight * len(self.stages_done)

        current_idx = _STAGE_ORDER.index(self.current_stage) if self.current_stage in _STAGE_ORDER else 0
        base = stage_weight * sum(
            1 for stage in _STAGE_ORDER[:current_idx] if stage in self.stages_done
        )

        if self.current_stage == "verify":
            done = sum(
                1 for run in self.verify_runs.values()
                if not run.get("running") and "exit" in run
            )
            fraction = done / max(len(_VERIFY_CONFIGS), 1)
        elif self.max_trials and self.max_trials > 0:
            trial = max(0, self.trial - 1)
            step_fraction = self.step / max(self.step + 1, 1)
            fraction = min(
                0.95,
                (trial + step_fraction) / self.max_trials,
            )
        else:
            fraction = 0.5

        return min(100.0, base + stage_weight * fraction)

    def elapsed_seconds(self) -> float:
        """Wall-clock elapsed since the attempt's first event."""
        if self.started_at_epoch == 0.0:
            return 0.0
        end_epoch = self.last_event_epoch if self.done else time.time()
        return max(0.0, end_epoch - self.started_at_epoch)

    def freshness_seconds(self) -> float:
        if self.last_event_epoch == 0.0:
            return 0.0
        return max(0.0, time.time() - self.last_event_epoch)

    def _stall_threshold(self) -> int:
        """Context-aware stall threshold.

        During mvn / verify legs, minutes of silence is normal (test-
        compile + test-run per configuration). Elsewhere, a much
        shorter timeout applies. In-flight tool state (set by
        ``agent.tool.begin``) wins over the last-completed turn's tool
        so a fresh mvn call gets its extended threshold immediately.
        """
        if self.current_stage == "verify":
            return _STALL_SECONDS_VERIFY
        if self._in_flight_tool in ("mvn_test", "mvn_test_compile", "mvn_install"):
            return _STALL_SECONDS_MVN
        # Fallback: peek at the most-recent turn's tool_kind for
        # telemetry that predates agent.tool.begin.
        if self.recent_turns:
            last_kind = self.recent_turns[-1].get("action_kind", "")
            if last_kind in ("mvn_test", "mvn_test_compile", "mvn_install"):
                return _STALL_SECONDS_MVN
        return _STALL_SECONDS_GENERIC

    def is_stalled(self) -> bool:
        return (
            not self.done
            and self.last_event_epoch > 0
            and self.freshness_seconds() > self._stall_threshold()
        )

    def total_cost_usd(self) -> float:
        return (
            (self.tokens_in / 1000.0) * _COST_PER_1K_INPUT
            + (self.tokens_out / 1000.0) * _COST_PER_1K_OUTPUT
        )

    def llm_latency_stats(self) -> dict[str, float]:
        w = list(self.llm_latency_window)
        if not w:
            return {}
        return {
            "n":      float(len(w)),
            "median": statistics.median(w),
            "p95":    sorted(w)[max(0, int(len(w) * 0.95) - 1)],
            "max":    max(w),
        }

    def anomalies(self) -> list[str]:
        out: list[str] = []
        if self.is_stalled():
            out.append(
                f"[red]stall[/red] · no new event in {self.freshness_seconds():.0f}s"
                f" (last stage: {self.current_stage or 'unknown'})"
            )
        # 3+ consecutive tool errors on the same tool = flapping
        last_five = list(self.recent_turns)[-5:]
        if len(last_five) >= 3:
            tail = last_five[-3:]
            if all(t.get("exit_code", 0) != 0 and t.get("action_kind") != "__none__" for t in tail):
                kinds = {t.get("action_kind") for t in tail}
                out.append(
                    f"[yellow]flapping[/yellow] · 3+ consecutive tool errors ({', '.join(kinds)})"
                )
        # LLM latency spike
        stats = self.llm_latency_stats()
        if stats and stats["n"] >= 5:
            latest = self.llm_latency_window[-1]
            if latest > stats["median"] * 3 and latest > 30:
                out.append(
                    f"[yellow]LLM slow[/yellow] · last call {latest:.1f}s vs median {stats['median']:.1f}s"
                )
        if self.parser_drift >= 3:
            out.append(f"[yellow]parser drift[/yellow] · {self.parser_drift} turns without a valid action tag")
        if self.llm_calls_err > 0:
            out.append(f"[red]LLM errors[/red] · {self.llm_calls_err} failures · latest: {self.llm_last_error[:80]}")
        return out

    def status_line(self) -> str:
        if self.done:
            if self.gate and self.gate.get("valid"):
                return f"[green]done · gate=✅ · pred_f2p={self.gate.get('pred_f2p_n',0)}[/green]"
            if self.gate:
                return f"[red]done · gate=❌ · pred_f2p={self.gate.get('pred_f2p_n',0)}[/red]"
            return "[dim]done (no gate)[/dim]"
        stale = self.freshness_seconds()
        stale_tag = f" · [red]{stale:.0f}s stale[/red]" if stale > 60 else ""
        if self.verify_started:
            n_done = sum(1 for c in self.verify_runs.values() if not c.get("running") and "exit" in c)
            return f"[cyan]verify · {n_done}/6 configs[/cyan]{stale_tag}"
        if self.submitted:
            return f"[cyan]submitted · {self.submitted_diff_chars:,} chars · verify next[/cyan]{stale_tag}"
        if self.current_stage == "generate":
            return f"[yellow]generate · trial {self.trial}/{self.max_trials} · step {self.step}[/yellow]{stale_tag}"
        if self.current_stage:
            return f"[yellow]stage: {self.current_stage}[/yellow]{stale_tag}"
        if self.stages_done:
            return f"[dim]idle · done: {sorted(self.stages_done)}[/dim]"
        return "[dim]queued[/dim]"


# ─── CaseGroup: attempts sharing a case_id ─────────────────────────────────


@dataclass
class CaseGroup:
    """A logical case with its N pass@k attempts.

    Fleet-level counters (done / active / queued) count CASES, not
    attempts — so three attempts of one case never inflate the header.
    """

    case_id: str
    attempts: list[CaseState] = field(default_factory=list)

    def is_done(self) -> bool:
        """Case is done iff EVERY attempt has emitted ``telemetry.close``.

        Under pass@k, a case is not "done" until all N attempts finish
        and the merger writes the union. We approximate that here by
        requiring every attempt's own ``done`` flag — the merger runs
        after and doesn't touch the telemetry stream.
        """
        return bool(self.attempts) and all(a.done for a in self.attempts)

    def is_active(self) -> bool:
        """At least one attempt has begun a stage but not all are done."""
        if self.is_done():
            return False
        return any(a.current_stage or a.stages_done for a in self.attempts)

    def is_queued(self) -> bool:
        """No attempt has emitted a single stage event yet."""
        return all(
            not a.done and not a.current_stage and not a.stages_done
            for a in self.attempts
        )

    def is_stale(self) -> bool:
        """Every attempt has emitted events, but none has finished and
        every attempt is stalled.

        This is the live-viewer analogue of a timed-out container whose
        telemetry file never receives ``telemetry.close``. Without this
        category those cases are otherwise miscounted as active forever.
        """
        if not self.attempts or self.is_done() or self.is_queued():
            return False
        return all(a.is_stalled() for a in self.attempts)

    def freshest_attempt(self) -> CaseState | None:
        """Attempt with the newest ``last_event_epoch`` — used to pick
        which attempt drives freshness/LLM/tool panels for this case."""
        if not self.attempts:
            return None
        return max(self.attempts, key=lambda a: a.last_event_epoch)


def _group_by_case(attempts: list[CaseState]) -> list[CaseGroup]:
    """Group ``CaseState`` instances by ``case_id``. Preserves order:
    a case appears in the returned list at the position of its first
    attempt in the input; attempts within each group preserve their
    input order (which the discovery layer sorts by attempt index)."""
    groups: dict[str, CaseGroup] = {}
    order: list[str] = []
    for a in attempts:
        if a.case_id not in groups:
            groups[a.case_id] = CaseGroup(case_id=a.case_id, attempts=[])
            order.append(a.case_id)
        groups[a.case_id].attempts.append(a)
    return [groups[cid] for cid in order]


_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def _spinner_char() -> str:
    return _SPINNER_FRAMES[int(time.time() * 8) % len(_SPINNER_FRAMES)]


def _compact_bar(pct: float, width: int = 32) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100.0))
    if filled == width:
        return "━" * width
    if filled == 0:
        return "╺" + " " * (width - 1)
    return "━" * filled + "╸" + " " * (width - filled - 1)


# ─── panel renderers ─────────────────────────────────────────────────────────


def _header_panel(cases: list[CaseState], wall_started_at: float) -> Panel:
    """Fleet-level totals, burn rate, indicative cost.

    Cases are counted at the group level via :class:`CaseGroup` (three
    attempts of one case never inflate the counters). Tokens + cost
    sum across every attempt because each attempt is a real independent
    LLM run whose spend accumulates linearly.
    """
    tot_in = sum(c.tokens_in for c in cases)
    tot_out = sum(c.tokens_out for c in cases)
    tot_cost = sum(c.total_cost_usd() for c in cases)

    # Burn rate — tokens accumulated over min(elapsed, window).
    fleet_elapsed = time.time() - wall_started_at if wall_started_at > 0 else 0.0
    window = min(fleet_elapsed, _BURN_WINDOW_SEC) if fleet_elapsed > 0 else 1.0
    burn_in = (tot_in / window) * 60 if window else 0
    burn_out = (tot_out / window) * 60 if window else 0

    groups = _group_by_case(cases)
    n_cases = len(groups)
    n_done = sum(1 for g in groups if g.is_done())
    n_active = sum(1 for g in groups if g.is_active() and not g.is_stale())
    n_queued = sum(1 for g in groups if g.is_queued())
    n_stale = sum(1 for g in groups if g.is_stale())

    n_attempts_total = len(cases)
    n_attempts_done = sum(1 for c in cases if c.done)

    failed = sum(
        1 for g in groups
        if g.is_done()
        and not any(a.gate and a.gate.get("valid") for a in g.attempts)
    )
    pct = 100.0 * n_done / n_cases if n_cases else 0.0
    lines: list[str] = [
        f"{_spinner_char()} {_compact_bar(pct)} [bold]{pct:>3.0f}%[/bold] "
        f"fleet [cyan]{_fmt_dur(fleet_elapsed)}[/cyan]   "
        f"[green]{n_done}[/green] done · "
        f"[red]{failed}[/red] failed · "
        f"[yellow]{n_active}[/yellow] active · "
        f"[magenta]{n_stale}[/magenta] stale · "
        f"[dim]{n_queued}[/dim] queued "
        f"([dim]{n_cases} total[/dim])",
    ]
    if n_attempts_total != n_cases:
        per_case = n_attempts_total // max(n_cases, 1)
        lines.append(
            f"attempts [green]{n_attempts_done}[/green]/{n_attempts_total} finished  "
            f"([dim]pass@{per_case}[/dim])",
        )
    lines.append(
        f"tokens [green]{tot_in:>10,}[/green] in · [red]{tot_out:>7,}[/red] out   "
        f"cost ≈ [magenta]${tot_cost:.3f}[/magenta]   "
        f"burn {burn_in:>6,.0f} in/min · {burn_out:>5,.0f} out/min",
    )
    return Panel(
        "\n".join(lines),
        border_style="cyan",
        title=f"[bold]swe-brew[/bold] · fleet monitor  [dim]{dt.datetime.now():%H:%M:%S}[/dim]",
        title_align="left",
    )


def _row_label(c: CaseState) -> str:
    """Row label as ``<case> · attempt-NNN`` (or just ``<case>`` for
    single-attempt runs) so attempts of the same case never render as
    visually identical rows."""
    if c.attempt_label == "single":
        return c.case_id[:52]
    return f"{c.case_id[:38]} · {c.attempt_label}"


def _progress_bars(cases: list[CaseState]) -> Progress:
    """RepoLaunch-style progress bar. Elapsed time comes from a
    :class:`TextColumn` wired to the attempt's own ``started_at_epoch``
    rather than :class:`TimeElapsedColumn` (which would reset every
    render because we recreate the Progress each frame)."""
    prog = Progress(
        TextColumn(" ", table_column=None),
        TextColumn("[bold]{task.description}", table_column=None),
        BarColumn(bar_width=25),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.fields[status]}"),
        TextColumn("{task.fields[elapsed]}"),
        expand=True,
    )
    for c in cases:
        prog.add_task(
            _row_label(c),
            total=100,
            completed=c.stage_progress_pct(),
            status=c.status_line(),
            elapsed=_fmt_dur(c.elapsed_seconds()),
        )
    return prog


def _llm_health_panel(cases: list[CaseState]) -> Panel:
    """LLM call latencies + error rate + freshness for the freshest attempt."""
    active = _active_case(cases)
    if active is None:
        body: Any = "[dim]no active case[/dim]"
    else:
        stats = active.llm_latency_stats()
        model = active.llm_model or "[dim]-[/dim]"
        n_ok = active.llm_calls_ok
        n_err = active.llm_calls_err
        # If we haven't seen explicit llm.call.end events, fall back to
        # derived count from agent.turn events.
        if n_ok == 0 and n_err == 0:
            n_ok = sum(1 for _ in active.recent_turns)
        source = "derived" if active.llm_latency_derived else "auth"
        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim")
        t.add_column()
        t.add_row("model", model if model.startswith("[") else f"[cyan]{model}[/cyan]")
        t.add_row("calls", f"[green]{n_ok}[/green] ok · [red]{n_err}[/red] err")
        if stats:
            t.add_row(
                "latency",
                f"n={int(stats['n'])} ({source})   "
                f"p50 {stats['median']:.1f}s · p95 {stats['p95']:.1f}s · max {stats['max']:.1f}s",
            )
        else:
            t.add_row("latency", "[dim]no data yet[/dim]")
        if active.llm_calls_err > 0:
            t.add_row("last err", f"[red]{active.llm_last_error[:60]}[/red]")
        t.add_row("freshness", f"{active.freshness_seconds():.0f}s since last event")
        body = t
    return Panel(body, title="[bold]LLM health[/bold]", border_style="magenta", title_align="left")


def _tool_health_panel(cases: list[CaseState]) -> Panel:
    """Per-tool histogram with exit-code breakdown for the freshest attempt."""
    active = _active_case(cases)
    if active is None or not active.tool_counter:
        return Panel(
            "[dim]no tool calls yet[/dim]",
            title="[bold]tool health[/bold]",
            border_style="blue",
            title_align="left",
        )
    t = Table.grid(padding=(0, 2))
    t.add_column("tool", style="cyan", min_width=18)
    t.add_column("n", justify="right", min_width=4)
    t.add_column("ok", justify="right", style="green", min_width=4)
    t.add_column("err", justify="right", style="red", min_width=4)
    for kind, n in active.tool_counter.most_common():
        ok = active.tool_exit_ok.get(kind, 0)
        err = active.tool_exit_err.get(kind, 0)
        t.add_row(kind, str(n), str(ok), str(err))
    if active.parser_drift:
        t.add_row("[dim]parser-drift[/dim]", str(active.parser_drift), "-", "-")
    return Panel(t, title="[bold]tool health[/bold]", border_style="blue", title_align="left")


def _verify_matrix_panel(cases: list[CaseState]) -> Panel | None:
    """6-way matrix status for whichever case is in verify."""
    active = next((c for c in cases if c.verify_started), None)
    if active is None:
        return None
    t = Table.grid(padding=(0, 2))
    t.add_column("config", style="cyan")
    t.add_column("state")
    t.add_column("tests", justify="right")
    t.add_column("exit", justify="right")
    t.add_column("elapsed", justify="right")
    for cfg in _VERIFY_CONFIGS:
        entry = active.verify_runs.get(cfg)
        if entry is None:
            t.add_row(cfg, "[dim]pending[/dim]", "-", "-", "-")
        elif entry.get("running"):
            t.add_row(cfg, "[yellow]running[/yellow]", "-", "-", "-")
        else:
            exit_code = entry.get("exit")
            style = "green" if exit_code == 0 else "yellow"
            t.add_row(
                cfg,
                f"[{style}]done[/{style}]",
                str(entry.get("tests", 0)),
                str(exit_code),
                f"{entry.get('elapsed_s', 0.0):.1f}s",
            )
    title = f"[bold]verify matrix[/bold] · {_attempt_title(active, 40)}"
    if active.gate:
        gate = active.gate
        recall = gate.get("recall_pct")
        recall_txt = f"{recall:.1f}%" if recall is not None else "-"
        summary = (
            f"gate={'[green]pass[/green]' if gate.get('valid') else '[red]fail[/red]'}  "
            f"pred_f2p={gate.get('pred_f2p_n', 0)}  "
            f"gold_f2p={gate.get('gold_f2p_n', 0)}  "
            f"pred∩gold={gate.get('intersection_n', 0)}  "
            f"recall={recall_txt}"
        )
        return Panel(
            Group(t, Text.from_markup(f"\n[bold]{summary}[/bold]")),
            title=title, border_style="green", title_align="left",
        )
    return Panel(t, title=title, border_style="green", title_align="left")


def _anomaly_panel(cases: list[CaseState]) -> Panel | None:
    """Render only when at least one anomaly is present. Rows are
    labeled ``<case> · attempt-NNN`` so a stalled attempt is
    distinguishable from healthy siblings."""
    rows: list[str] = []
    for c in cases:
        for a in c.anomalies():
            rows.append(f"[dim]{_attempt_title(c, 40)}[/dim]  {a}")
    if not rows:
        return None
    return Panel(
        "\n".join(rows),
        title="[bold]anomalies[/bold]",
        border_style="red",
        title_align="left",
    )


def _trial_history_panel(cases: list[CaseState]) -> Panel | None:
    """Per-stage per-trial outcomes across setup / organize / test-gen.
    Rendered only when there's at least one trial to show."""
    active = _active_case(cases)
    if active is None or not active.trial_history:
        return None
    t = Table.grid(padding=(0, 2))
    t.add_column("stage", style="cyan", min_width=10)
    t.add_column("trial", justify="right", min_width=6)
    t.add_column("outcome", min_width=12)
    t.add_column("steps", justify="right", min_width=6)
    t.add_column("note")
    for entry in active.trial_history[-15:]:
        outcome = "[green]submit[/green]" if entry.get("submitted") else "[yellow]exhausted[/yellow]"
        t.add_row(
            entry.get("stage", "?"),
            str(entry.get("trial", "?")),
            outcome,
            str(entry.get("steps_used", "?")),
            str(entry.get("note", ""))[:60],
        )
    return Panel(
        t,
        title=f"[bold]trial history[/bold] · {_attempt_title(active, 40)}",
        border_style="cyan",
        title_align="left",
    )


def _submit_panel(cases: list[CaseState]) -> Panel | None:
    """Submitted diff path + size for every case that has submitted.
    Points the operator at where to inspect. ``candidate_path`` is
    derived from each attempt's own ``telemetry_path.parent`` so
    diffs at ``enhanced-attempt-NNN/`` resolve correctly."""
    submits = [c for c in cases if c.submitted]
    if not submits:
        return None
    t = Table.grid(padding=(0, 2))
    t.add_column(style="cyan", min_width=52)
    t.add_column("chars", justify="right")
    t.add_column("path")
    for c in submits:
        # ``telemetry_path`` is <case_dir>/<enhanced_dir>/telemetry.jsonl.
        # Stages live alongside telemetry.jsonl in that <enhanced_dir>.
        attempt_dir = c.telemetry_path.parent
        candidate_path = str(attempt_dir / "stages" / "generate" / "trial-*" / "candidate.diff")
        t.add_row(_attempt_title(c, 52), f"{c.submitted_diff_chars:,}", candidate_path)
    return Panel(
        t,
        title="[bold]submitted diffs[/bold]",
        border_style="green",
        title_align="left",
    )


def _recent_turns_panel(cases: list[CaseState]) -> Panel:
    active = _active_case(cases)
    if active is None or not active.recent_turns:
        return Panel(
            "[dim]no agent turns yet[/dim]",
            title="[bold]recent turns[/bold]",
            border_style="dim",
            title_align="left",
        )
    t = Table(show_header=True, header_style="bold", expand=True)
    t.add_column("trial", justify="right", width=5)
    t.add_column("step", justify="right", width=4)
    t.add_column("action", style="cyan", min_width=18)
    t.add_column("exit", justify="right", width=4)
    t.add_column("tool_s", justify="right", width=7)
    t.add_column("Δtok_in", justify="right", width=9)
    t.add_column("Δtok_out", justify="right", width=9)
    prev_in = 0
    prev_out = 0
    for r in list(active.recent_turns)[-_MAX_RECENT_TURNS:]:
        tok_in = r.get("tokens_in", 0)
        tok_out = r.get("tokens_out", 0)
        din = max(0, tok_in - prev_in)
        dout = max(0, tok_out - prev_out)
        exit_code = r.get("exit_code", 0)
        exit_style = "green" if exit_code == 0 else "red"
        t.add_row(
            str(r.get("trial", "?")),
            str(r.get("step", "?")),
            str(r.get("action_kind", "?")),
            f"[{exit_style}]{exit_code}[/{exit_style}]",
            f"{r.get('elapsed_s', 0.0):.1f}",
            f"{din:,}",
            f"{dout:,}",
        )
        prev_in, prev_out = tok_in, tok_out
    return Panel(
        t,
        title=f"[bold]recent turns[/bold] · [cyan]{_attempt_title(active, 60)}[/cyan]",
        border_style="cyan",
        title_align="left",
    )


def _active_case(cases: list[CaseState]) -> CaseState | None:
    """The freshest attempt currently emitting events.

    Among attempts still active (has a ``current_stage`` or
    ``verify_started`` and hasn't emitted ``telemetry.close``), pick
    the one with the newest ``last_event_epoch``. Fall back to the
    most-recently-touched attempt overall when nothing is active.
    """
    if not cases:
        return None
    running = [
        c for c in cases
        if not c.done and (c.current_stage or c.verify_started)
    ]
    if running:
        return max(running, key=lambda c: c.last_event_epoch)
    return max(cases, key=lambda c: c.last_event_epoch)


def _attempt_title(c: CaseState, max_len: int = 60) -> str:
    """Panel-title formatter: ``<case> · attempt-NNN`` (or just
    ``<case>`` for single-attempt runs)."""
    if c.attempt_label == "single":
        return c.case_id[:max_len]
    return f"{c.case_id[:max_len - len(c.attempt_label) - 3]} · {c.attempt_label}"


def _render(
    cases: list[CaseState],
    wall_started_at: float,
) -> Group:
    parts: list[Any] = [
        _header_panel(cases, wall_started_at),
    ]
    parts.extend([
        _progress_bars(cases),
        _llm_health_panel(cases),
        _tool_health_panel(cases),
    ])
    # Conditional panels — only render when there's content.
    for panel_fn in (_verify_matrix_panel, _anomaly_panel, _trial_history_panel, _submit_panel):
        p = panel_fn(cases)
        if p is not None:
            parts.append(p)
    parts.append(_recent_turns_panel(cases))
    return Group(*parts)


def _fmt_dur(sec: float) -> str:
    if sec < 60:
        return f"{sec:.0f}s"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


# ─── discovery ───────────────────────────────────────────────────────────────


def _discover_examples() -> list[Path]:
    """Discover telemetry files across single-attempt and pass@k layouts.

    Under pass@k each attempt writes to
    ``<case>/enhanced-attempt-NNN/telemetry.jsonl`` and the merger
    writes only ``<case>/enhanced/result.json`` afterwards. Single-
    attempt runs write directly to ``<case>/enhanced/telemetry.jsonl``.

    Dedup rule: when a case has both layouts (a stale ``enhanced/``
    file left behind alongside fresh attempt files), skip the
    ``enhanced/`` file — per-attempt files are the ground truth.
    """
    root = Path("data/examples/playground")
    if not root.exists():
        return []
    single = list(root.glob("*/enhanced/telemetry.jsonl"))
    attempts = list(root.glob("*/enhanced-attempt-*/telemetry.jsonl"))
    cases_with_attempts = {p.parent.parent for p in attempts}
    single_deduped = [p for p in single if p.parent.parent not in cases_with_attempts]
    return sorted(single_deduped + attempts)


def _resolve_paths(case_dirs: list[str] | None, file_paths: list[str] | None) -> list[Path]:
    """Resolve user-provided case dirs / explicit files into a list of
    telemetry.jsonl paths.

    For a case in pass@k mode, fan out to every
    ``enhanced-attempt-*/telemetry.jsonl`` under it; otherwise fall
    back to the single ``enhanced/telemetry.jsonl`` path.
    """
    paths: list[Path] = []
    if file_paths:
        paths.extend(Path(p) for p in file_paths)
    if case_dirs:
        for cd in case_dirs:
            p = Path(cd)
            if p.name == "telemetry.jsonl":
                paths.append(p)
                continue
            attempt_files = sorted(p.glob("enhanced-attempt-*/telemetry.jsonl"))
            if attempt_files:
                paths.extend(attempt_files)
            else:
                paths.append(p / "enhanced" / "telemetry.jsonl")
    if not paths:
        paths = _discover_examples()
    return paths


# ─── entry point ─────────────────────────────────────────────────────────────


app = typer.Typer(add_completion=False)


@app.command()
def watch(
    case_dirs: list[str] = typer.Argument(
        None,
        help="Case dirs (each containing enhanced/ or enhanced-attempt-*/ "
             "telemetry.jsonl). Omit to auto-discover under "
             "data/examples/playground/.",
    ),
    file: list[str] = typer.Option(
        None, "--file", "-f",
        help="Explicit telemetry.jsonl path(s). Repeatable.",
    ),
    interval: float = typer.Option(
        1.0, "--interval", "-i",
        help="Refresh interval in seconds.",
    ),
    once: bool = typer.Option(
        False, "--once",
        help="Render one frame and exit (for scripting / CI).",
    ),
    exit_on_done: bool = typer.Option(
        False, "--exit-on-done",
        help="Exit automatically when every case emits telemetry.close. "
             "Off by default so you can inspect final state; use this in "
             "unattended contexts.",
    ),
    no_alt_screen: bool = typer.Option(
        False, "--no-alt-screen",
        help="Disable Rich alt-screen mode. Useful when a terminal "
             "multiplexer fails to redraw the live dashboard.",
    ),
    force_live: bool = typer.Option(
        False, "--force-live",
        help="Ignore TTY detection and redraw with ANSI clear/home. Use this "
             "when the SSH command otherwise renders one static frame.",
    ),
    stream: bool = typer.Option(
        False, "--stream",
        help="Append a full frame on every interval instead of using Rich "
             "Live. This is a fallback for broken terminals and intentionally "
             "accumulates scrollback.",
    ),
) -> None:
    """Live fleet monitor.

    Displays fleet totals, per-case progress bars, LLM call health,
    tool exit-code histogram, six-way verify matrix, anomaly warnings,
    and a recent-turns trace. Reads only telemetry files; never signals
    the running process, so multiple viewers can attach concurrently.
    """
    console = Console()
    paths = _resolve_paths(case_dirs, file)

    if not paths:
        console.print(
            "[yellow]No telemetry.jsonl files found.[/yellow] Either the fleet "
            "hasn't started yet, or the case dirs you passed don't have any.\n"
            "Auto-discovery scans [bold]data/examples/playground/[/bold] relative "
            "to your current working directory.",
        )
        raise typer.Exit(code=1)

    # Parse both case_id and attempt_label from each path so attempts
    # of the same case are distinguishable in every panel + row.
    cases = []
    for p in paths:
        case_id, attempt_label = parse_case_and_attempt(p)
        cases.append(
            CaseState(
                case_id=case_id,
                telemetry_path=p,
                attempt_label=attempt_label,
            ),
        )
    wall_started_at = time.time()      # will be overwritten below with earliest case start

    # Rich's Live screen mode only clears cleanly when stdout is a real
    # terminal. If this process is piped or captured, print one complete
    # frame instead of appending a new frame on every refresh.
    if not console.is_terminal and not force_live:
        for c in cases:
            c.refresh_from_file()
        earliest = min(
            (c.started_at_epoch for c in cases if c.started_at_epoch > 0),
            default=time.time(),
        )
        console.print(_render(cases, earliest))
        console.print(
            "[dim]non-TTY watcher: rendered one frame; live mode requires a "
            "real terminal. Use --once for scripts or attach to a TTY for "
            "the alt-screen dashboard.[/dim]",
        )
        return

    if force_live and not console.is_terminal:
        while True:
            for c in cases:
                c.refresh_from_file()
            earliest = min(
                (c.started_at_epoch for c in cases if c.started_at_epoch > 0),
                default=time.time(),
            )
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            console.print(_render(cases, earliest))
            sys.stdout.flush()
            time.sleep(interval)

    if stream:
        done_since: float | None = None
        while True:
            for c in cases:
                c.refresh_from_file()
            earliest = min(
                (c.started_at_epoch for c in cases if c.started_at_epoch > 0),
                default=time.time(),
            )
            console.rule(f"[bold]swe-brew · fleet monitor  {dt.datetime.now():%H:%M:%S}[/bold]")
            console.print(_render(cases, earliest))
            groups = _group_by_case(cases)
            if exit_on_done and groups and all(g.is_done() for g in groups):
                if done_since is None:
                    done_since = time.time()
                elif time.time() - done_since > 5.0:
                    break
            else:
                done_since = None
            time.sleep(interval)
        return

    if once:
        for c in cases:
            c.refresh_from_file()
        earliest = min((c.started_at_epoch for c in cases if c.started_at_epoch > 0), default=time.time())
        console.print(_render(cases, earliest))
        return

    # Live loop.
    for c in cases:
        c.refresh_from_file()
    earliest = min((c.started_at_epoch for c in cases if c.started_at_epoch > 0), default=time.time())
    wall_started_at = earliest

    # Post-completion behaviour: keep rendering so post-mortem inspection
    # is possible without re-launching; ``--exit-on-done`` opts into
    # auto-exit for CI. Ctrl+C exits cleanly at any time — the underlying
    # run is unaffected.
    #
    # Rediscovery: re-scan for new telemetry files every
    # ``rediscover_every_s`` seconds so pass@k attempts started after
    # the watcher was launched appear on the dashboard live. Existing
    # attempts keep their replay cursor via ``_fh_offset``.
    known_paths: set[Path] = {c.telemetry_path for c in cases}
    rediscover_every_s = 10.0
    last_rediscover_at = time.monotonic()

    def _rediscover_and_append() -> int:
        """Pick up any telemetry files that didn't exist at startup.
        Returns the number of new files added."""
        nonlocal wall_started_at
        current = _resolve_paths(case_dirs, file)
        new_paths = [p for p in current if p not in known_paths]
        if not new_paths:
            return 0
        for p in new_paths:
            case_id, attempt_label = parse_case_and_attempt(p)
            fresh = CaseState(
                case_id=case_id,
                telemetry_path=p,
                attempt_label=attempt_label,
            )
            fresh.refresh_from_file()
            cases.append(fresh)
            known_paths.add(p)
            # If this new attempt's first event is older than the
            # current fleet start, extend the fleet clock back so
            # elapsed time stays honest.
            if 0 < fresh.started_at_epoch < wall_started_at:
                wall_started_at = fresh.started_at_epoch
        return len(new_paths)

    with Live(
        _render(cases, wall_started_at),
        console=console,
        refresh_per_second=1,
        auto_refresh=False,
        screen=not no_alt_screen,
        vertical_overflow="crop",
    ) as live:
        try:
            done_since: float | None = None
            while True:
                # Periodic rediscovery so new attempts + new cases
                # spun up mid-fleet appear live.
                now_mono = time.monotonic()
                if now_mono - last_rediscover_at >= rediscover_every_s:
                    _rediscover_and_append()
                    last_rediscover_at = now_mono
                for c in cases:
                    c.refresh_from_file()
                live.update(_render(cases, wall_started_at))

                # Completeness check at the CASE level: under pass@k a
                # case is done only when every attempt has emitted
                # ``telemetry.close``.
                groups = _group_by_case(cases)
                if groups and all(g.is_done() for g in groups):
                    # Every case has emitted telemetry.close. In auto-exit mode
                    # (CI / unattended), leave after a short grace period so any
                    # final events land in the JSONL. Otherwise, keep rendering
                    # for post-mortem inspection — the user Ctrl+Cs when done.
                    if exit_on_done:
                        if done_since is None:
                            done_since = time.time()
                        elif time.time() - done_since > 5.0:
                            break
                    # else: don't exit; keep the final frame visible.
                else:
                    done_since = None

                time.sleep(interval)
        except KeyboardInterrupt:
            # Live's __exit__ restores the terminal; print AFTER so the message
            # lands cleanly instead of being overwritten by the final Live frame.
            pass
    console.print("[dim]watcher exiting; underlying run (if any) is unaffected.[/dim]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
