"""Rich progress + JSONL telemetry for the enhance pipeline.

# SOURCE ATTRIBUTION
# ==================
#
# The ``Progress`` column layout (SpinnerColumn / BarColumn /
# TextColumn / TimeElapsedColumn) is **verbatim** from
# ``launch/run.py:168-176`` in SWE-bench-Live/launch (RepoLaunch).
# Only two things change:
#   - Our task is single-instance right now (POC), so we render
#     stage-level progress within one case instead of case-level
#     progress across N cases. When we scale to fleet, the same
#     factory produces the fleet-level bar too.
#   - We add a JSONL telemetry sink (``TelemetryLog``) that RepoLaunch
#     doesn't have — one line per event, machine-readable, so
#     downstream analysis (agent-loop convergence curves, tokens-per-
#     F2P plots, etc.) can consume it directly.
#
# The class-level factory ``make_progress()`` also drives the fleet-
# level bar used in ``run_setup()``/``run_organize()`` in RepoLaunch,
# which we'll wire up when we scale beyond the POC.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

__all__ = [
    "TelemetryLog",
    "make_progress",
    "stage",
]


# ─── Progress factory — verbatim RepoLaunch column layout ─────────────────────


def make_progress(console: Console) -> Progress:
    """Return a Progress with RepoLaunch's exact column configuration.

    Verbatim from ``launch/run.py:168-177``:

        with Progress(
            SpinnerColumn(),
            TextColumn(
                "[green]Success: {task.fields[success]}[/green] | [red]Fail: {task.fields[fail]}[/red]"
            ),
            BarColumn(),
            TextColumn(f"Total: {len(dataset)}"),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=console,
        ) as progress: ...

    We drop the ``Total: {len(dataset)}`` column (POC = single case) and
    keep everything else — the Success/Fail counter is useful for the
    per-turn agent-loop progression, the spinner keeps the terminal
    alive during long Maven runs.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        TextColumn("[green]ok: {task.fields[success]}[/green] | [red]err: {task.fields[fail]}[/red]"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
    )


# ─── Telemetry sink — JSONL, one line per event ──────────────────────────────


@dataclass
class TelemetryLog:
    """One JSONL file per ATTEMPT.

    Single-attempt runs write to ``<case_dir>/enhanced/telemetry.jsonl``;
    pass@k runs write to ``<case_dir>/enhanced-attempt-NNN/telemetry.jsonl``.
    The ``--out-dir`` CLI flag selects which. The orchestrator writes
    exactly one instance per ``enhance()`` invocation.

    Each ``emit()`` writes a line. The schema is intentionally flat so
    downstream tooling (pandas, jq, …) can consume it without joins.

    Every event carries ``ts``, ``event``, ``case_id``, and any additional
    payload keys. Emitted events (grep-able for downstream analysis):

        stage.begin        stage_name
        stage.end          stage_name, ok, elapsed_s
        resolve.done       base_commit, head_commit, method
        snapshot.done      base_commit, work_dir
        sandbox.done       image_tag, container_short_id
        agent.trial.begin  trial, max_trials
        agent.turn         trial, step, action_kind, exit_code, elapsed_s,
                           tokens_in, tokens_out, cost_input_tokens_cumulative,
                           cost_output_tokens_cumulative
        agent.submit       trial, step, diff_chars
        verify.run.begin   configuration
        verify.run.end     configuration, tests_count, exit_code, elapsed_s
        verify.gate        valid, reasons, pred_f2p_n, gold_f2p_n,
                           intersection_n, recall_pct
    """

    path: Path
    case_id: str
    _fh: Any = field(default=None, init=False, repr=False)

    def __enter__(self) -> TelemetryLog:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 'a' mode so retries append rather than clobber (RepoLaunch's
        # ``safe_read_result`` uses the same idempotency semantics).
        self._fh = self.path.open("a", encoding="utf-8")
        self.emit("telemetry.open", version=1)
        return self

    def __exit__(self, *_exc: Any) -> None:
        try:
            self.emit("telemetry.close")
        finally:
            if self._fh is not None:
                self._fh.close()

    def emit(self, event: str, **payload: Any) -> None:
        rec = {
            "ts": dt.datetime.now(tz=dt.UTC).isoformat(),
            "event": event,
            "case_id": self.case_id,
            **payload,
        }
        if self._fh is not None:
            self._fh.write(json.dumps(rec, default=str) + "\n")
            self._fh.flush()


# ─── Stage context manager: emits begin/end + updates a progress task ────────


@contextmanager
def stage(
    progress: Progress,
    task_id: int,
    telemetry: TelemetryLog,
    name: str,
) -> Iterator[None]:
    """Wrap one pipeline stage. Bumps the progress bar, times the stage,
    routes exceptions through telemetry so they appear in the JSONL log.
    """
    t0 = time.perf_counter()
    telemetry.emit("stage.begin", stage_name=name)
    progress.update(task_id, description=f"[cyan]{name}")
    ok = True
    try:
        yield
    except Exception:
        ok = False
        raise
    finally:
        elapsed = time.perf_counter() - t0
        telemetry.emit("stage.end", stage_name=name, ok=ok, elapsed_s=round(elapsed, 3))
        progress.update(
            task_id,
            advance=1,
            success=progress.tasks[0].fields["success"] + (1 if ok else 0),
            fail=progress.tasks[0].fields["fail"] + (0 if ok else 1),
        )
