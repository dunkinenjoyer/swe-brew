"""Per-trial per-stage archival — RepoLaunch-parallel case management.

# PROVENANCE
# ==========
# Grounded on ``SWE-bench-Live/launch/launch/agent/{setup,organize}/save.py``.
# Their pattern: at the END of each workflow, capture the whole state
# (commands, docs, exception, duration, cost) into a JSON blob and
# write to ``playground/<instance_id>/result.json``. Even on exception,
# the save still runs — that's the "never lose the trajectory" invariant.
#
# We extend to per-STAGE per-TRIAL archives so a multi-trial agent's
# work isn't overwritten between trials. Layout is documented in
# ``docs/DESIGN-ENHANCED-2026-09-16.md`` §5:
#
#     enhanced/
#     ├── result.json               # RepoLaunch-shape aggregate
#     ├── setup.json                # SetupOutput
#     ├── organize.json             # OrganizeOutput
#     ├── verify_report.json        # full 6-way matrix
#     ├── test_results.json         # lean scorecard
#     ├── telemetry.jsonl           # per-event stream
#     ├── stages/
#     │   ├── setup/trial-001/     # {turns.jsonl, modifications.diff, outcome.json}
#     │   ├── organize/trial-001/
#     │   └── generate/trial-001/  # {candidate.diff, turns.jsonl, outcome.json}
#     └── logs/
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swe_brew.agent.framework import LoopResult

__all__ = [
    "archive_trial",
    "archive_final",
    "capture_sandbox_diff",
    "safe_write_json",
    "TrialArchive",
]


@dataclass
class TrialArchive:
    """Per-trial per-stage archive path anchor."""

    enhanced_dir:   Path
    stage:          str                    # "setup" | "organize" | "generate"
    trial:          int                    # 1-indexed
    diff_path:      Path = field(init=False)
    turns_path:     Path = field(init=False)
    outcome_path:   Path = field(init=False)

    def __post_init__(self) -> None:
        trial_dir = self.enhanced_dir / "stages" / self.stage / f"trial-{self.trial:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        # generate → candidate.diff (test-gen's submitted diff)
        # setup/organize → modifications.diff (git diff of sandbox before/after)
        name = "candidate.diff" if self.stage == "generate" else "modifications.diff"
        self.diff_path = trial_dir / name
        self.turns_path = trial_dir / "turns.jsonl"
        self.outcome_path = trial_dir / "outcome.json"


# ─── low-level safe I/O ──────────────────────────────────────────────────────


def safe_write_json(path: Path, payload: Any, indent: int = 2) -> None:
    """Write JSON atomically; never let a serialization error crash the caller.

    RepoLaunch's save.py wraps its dump in try/except and logs on
    failure; we do the same, but write to a temp file first and rename
    for atomicity. If serialization fails at all, we write a minimal
    ``{"_serialization_error": "..."}`` marker so downstream inspectors
    at least see something. Even if that also fails, no exception
    escapes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        content = json.dumps(payload, indent=indent, default=str, ensure_ascii=False)
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
        return
    except Exception as e:                     # noqa: BLE001
        # Use type(payload) not repr(payload) — the payload's own __repr__
        # may be what's raising in the first place.
        marker_content = json.dumps(
            {
                "_serialization_error": f"{type(e).__name__}: {e!s}",
                "_payload_type":        str(type(payload)),
            },
            indent=indent,
        )
        try:
            path.write_text(marker_content, encoding="utf-8")
        except Exception:                       # noqa: BLE001
            # Absolutely last resort — write a static error string.
            try:
                path.write_text('{"_serialization_error": "unknown"}', encoding="utf-8")
            except Exception:                   # noqa: BLE001
                pass


# ─── sandbox diff capture ────────────────────────────────────────────────────


def capture_sandbox_diff(work_dir: Path, base_commit: str) -> str:
    """Capture a diff of the sandbox worktree against base_commit.

    Used by setup + organize archival to record what the write-allowed
    agent actually modified (settings.xml overrides, pom edits, etc.).
    """
    try:
        r = subprocess.run(
            ["git", "-C", str(work_dir), "diff", base_commit, "--"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0:
            return r.stdout
        # Fall back to workdir-only diff if HEAD moved
        r2 = subprocess.run(
            ["git", "-C", str(work_dir), "diff", "HEAD", "--"],
            capture_output=True, text=True, timeout=60,
        )
        return r2.stdout
    except Exception as e:                     # noqa: BLE001
        return f"# diff capture failed: {type(e).__name__}: {e!s}\n"


# ─── trial-level archival ────────────────────────────────────────────────────


def archive_trial(
    enhanced_dir: Path,
    stage: str,
    trial: int,
    loop_result: LoopResult,
    *,
    diff_content: str = "",
    extra_outcome: dict[str, Any] | None = None,
) -> TrialArchive:
    """Write per-trial artefacts under ``enhanced/stages/<stage>/trial-N/``.

    Fault-tolerant: any I/O failure is captured in the outcome payload
    but never raised. This mirrors RepoLaunch's save.py invariant that
    "the save always runs, even on exception."
    """
    arc = TrialArchive(enhanced_dir=enhanced_dir, stage=stage, trial=trial)

    # 1. diff (candidate for generate; modifications for setup/organize)
    try:
        arc.diff_path.write_text(diff_content or "", encoding="utf-8")
    except Exception as e:                     # noqa: BLE001
        arc.diff_path.write_text(f"# diff write failed: {e!s}\n", encoding="utf-8")

    # 2. turns.jsonl — one line per message in the state
    try:
        with arc.turns_path.open("w", encoding="utf-8") as f:
            for m in loop_result.state.messages:
                f.write(json.dumps({"role": m.role, "content": m.content}, default=str) + "\n")
    except Exception as e:                     # noqa: BLE001
        arc.turns_path.write_text(f"# turns write failed: {e!s}\n", encoding="utf-8")

    # 3. outcome.json — trial-level metadata
    outcome: dict[str, Any] = {
        "stage":         stage,
        "trial":         trial,
        "submitted":     loop_result.submitted,
        "trials_used":   loop_result.trials_used,
        "steps_used":    loop_result.steps_used,
        "tool_calls":    [{"kind": k, "exit_code": e, "elapsed_s": s}
                          for (k, e, s) in loop_result.tool_calls],
        "cost":          loop_result.state.cost.get(stage, {}),
        "submit_payload_chars": len(loop_result.submit_payload),
    }
    if extra_outcome:
        outcome.update(extra_outcome)
    safe_write_json(arc.outcome_path, outcome)

    return arc


# ─── stage-level final artefact ──────────────────────────────────────────────


def archive_stage_final(
    enhanced_dir: Path,
    filename: str,
    payload: dict[str, Any] | Any,
) -> Path:
    """Write a stage's canonical output file (e.g. setup.json, organize.json).

    Uses ``safe_write_json`` to never crash on serialization issues.
    """
    path = enhanced_dir / filename
    # Pydantic model? Try to model_dump.
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()          # type: ignore[union-attr]
    elif hasattr(payload, "to_json"):
        payload = payload.to_json()             # type: ignore[union-attr]
    safe_write_json(path, payload)
    return path


# ─── final aggregate manifest (RepoLaunch-parity result.json) ────────────────


def archive_final(
    enhanced_dir: Path,
    manifest: dict[str, Any],
) -> Path:
    """Write ``enhanced/result.json`` — the RepoLaunch-shape aggregate.

    Fault-tolerant: even if serialization fails on some nested value,
    the file still gets written (see ``safe_write_json``).
    """
    path = enhanced_dir / "result.json"
    safe_write_json(path, manifest)
    return path
