"""Pass@k union merger — aggregate multiple independent attempts of a case.

The pass@k protocol runs the whole ``swe-brew-enhance`` pipeline N
independent times per case, then merges the resulting
``verify_report.json`` / ``result.json`` files into a single case-level
aggregate that unions the F2Ps.

Semantic
========
For each attempt i in 1..N:
  - The attempt produces its own ``verify_report.json`` with
    ``fail_to_pass``, ``pass_to_pass``, ``validity_gate``.
- An attempt's F2P count ONLY if its own validity_gate passed.
  A failed gate means the attempt did not execute cleanly or introduced
  regressions. A gated attempt may still contribute zero F2P; that is a
  valid completed observation, not a benchmark-style rejection.

Merged output:
  - ``fail_to_pass``       = union of ``a.fail_to_pass`` for every
                             gated attempt ``a``.
  - ``pass_to_pass``       = union of ``a.pass_to_pass`` for every
                             gated attempt ``a``.
  - ``validity_gate``      = OR of all attempts' gates.
  - ``passk_provenance``   = per test_id → list of attempt indices that
                             produced it (audit trail, tells us which
                             attempts contributed which tests).
  - ``per_attempt_summary`` = one row per attempt with its F2P/P2P
                             counts + gate outcome.

Why per-attempt gating?
=======================
An attempt whose gate=False may still emit tests that "look" like F2P by
SWT-Bench semantics but failed a load-bearing sanity check (for example,
no executed candidate outcomes or candidate-induced regressions). Only
the gated attempts have earned F2P credit. Completed P2P evidence is
retained even from ungated attempts because it measures the regression
basis.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

__all__ = [
    "AttemptSummary",
    "PassKAggregate",
    "merge_attempts",
    "discover_attempt_dirs",
]

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)


# Match subdirs named ``enhanced-attempt-{001,002,...}`` (zero-padded).
# Regular ``enhanced/`` (no suffix) is NOT included — that's where the
# merger writes its output.
_ATTEMPT_DIR_RE = re.compile(r"^enhanced-attempt-(\d+)$")


@dataclass
class AttemptSummary:
    """One attempt's contribution to the pass@k union."""

    attempt_index: int
    path: Path
    validity_gate: bool
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    assertion_f2p: list[str] = field(default_factory=list)
    compile_f2p: list[str] = field(default_factory=list)
    gold_fail_to_pass: list[str] = field(default_factory=list)
    pred_gold_developer_intersection: list[str] = field(default_factory=list)
    gold_recall_measurable: bool = False
    gold_recall_denominator: int = 0
    recall_vs_gold_pct: float | None = None
    exception: str | None = None
    completed: bool = False
    gate_reasons: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "attempt_index":  self.attempt_index,
            "path":           str(self.path),
            "validity_gate":  self.validity_gate,
            "completed":      self.completed,
            "exception":      self.exception,
            "gate_reasons":   self.gate_reasons,
            "duration_s":     self.duration_s,
            "fail_to_pass_n": len(self.fail_to_pass),
            "pass_to_pass_n": len(self.pass_to_pass),
            "assertion_f2p_n": len(self.assertion_f2p),
            "compile_f2p_n": len(self.compile_f2p),
            "gold_fail_to_pass_n": len(self.gold_fail_to_pass),
            "pred_gold_developer_intersection_n": len(self.pred_gold_developer_intersection),
            "gold_recall_measurable": self.gold_recall_measurable,
            "gold_recall_denominator": self.gold_recall_denominator,
            "recall_vs_gold_pct": self.recall_vs_gold_pct,
        }


@dataclass
class PassKAggregate:
    """Merged pass@k output — what gets written to <case_dir>/enhanced/."""

    case_id: str
    attempts_total: int
    attempts_gated: int
    validity_gate: bool
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    assertion_f2p: list[str] = field(default_factory=list)
    compile_f2p: list[str] = field(default_factory=list)
    gold_fail_to_pass: list[str] = field(default_factory=list)
    pred_gold_developer_intersection: list[str] = field(default_factory=list)
    gold_recall_measurable: bool = False
    gold_recall_denominator: int = 0
    recall_vs_gold_pct: float | None = None
    passk_provenance: dict[str, list[int]] = field(default_factory=dict)
    per_attempt_summary: list[AttemptSummary] = field(default_factory=list)

    def to_result_json(self, started_at: str = "") -> dict[str, Any]:
        """Render in the RepoLaunch-shape used by result.json.

        Mirrors ``result_json.build_result_json`` so downstream tooling
        that consumes result.json (fleet scoreboards, benchmark
        dashboards) reads the merged output identically to a single
        attempt — with a ``swe_brew.passk`` sub-block for the audit
        trail. Fields with no per-attempt analog (docker_image,
        test_commands) are taken from the highest-gated attempt if any,
        else empty; downstream consumers key on ``swe_brew.gate``.
        """
        best = _best_attempt_result_json(self.per_attempt_summary)
        result: dict[str, Any] = {
            "instance_id":         self.case_id,
            "case_id":             self.case_id,
            "completed":           any(a.completed for a in self.per_attempt_summary),
            "organize_completed":  bool(best.get("organize_completed")),
            "exception":           _first_exception(self.per_attempt_summary),
            "duration":            best.get("duration", 0),
            "duration_seconds":    sum(a.duration_s for a in self.per_attempt_summary),
            "docker_image":        best.get("docker_image", ""),
            "docker_image_layers": best.get("docker_image_layers", {}),
            "cost":                best.get("cost", {}),
            "repo_structure":      best.get("repo_structure", ""),
            "docs":                best.get("docs", ""),
            "setup_commands":      best.get("setup_commands", []),
            "rebuild_commands":    best.get("rebuild_commands", []),
            "test_commands":       best.get("test_commands", []),
            "print_commands":      best.get("print_commands", []),
            "log_parser":          best.get("log_parser", ""),
            "pertest_command":     best.get("pertest_command", {}),
            "test_status":         best.get("test_status", {}),
            "unittest_generator":  best.get("unittest_generator", ""),
            "created_at":          started_at,
            "swe_brew": {
                "gate":            self.validity_gate,
                "fail_to_pass":    sorted(self.fail_to_pass),
                "pass_to_pass":    sorted(self.pass_to_pass),
                "assertion_f2p":   sorted(self.assertion_f2p),
                "compile_f2p":     sorted(self.compile_f2p),
                "gate_reasons":     best.get("validity_notes", []),
                "gold_fail_to_pass": sorted(self.gold_fail_to_pass),
                "gold_recall_measurable": self.gold_recall_measurable,
                "gold_recall_denominator": self.gold_recall_denominator,
                "recall_vs_gold_pct": self.recall_vs_gold_pct,
                "pred_gold_developer_intersection": sorted(self.pred_gold_developer_intersection),
                "passk": {
                    "attempts_total":       self.attempts_total,
                    "attempts_gated":       self.attempts_gated,
                    "provenance":           dict(sorted(self.passk_provenance.items())),
                    "per_attempt_summary":  [a.to_json() for a in self.per_attempt_summary],
                },
            },
        }
        return result


def discover_attempt_dirs(case_dir: Path) -> list[Path]:
    """Return sorted list of ``<case-dir>/enhanced-attempt-NNN/`` dirs."""
    out: list[tuple[int, Path]] = []
    for child in case_dir.iterdir():
        if not child.is_dir():
            continue
        m = _ATTEMPT_DIR_RE.match(child.name)
        if m is None:
            continue
        out.append((int(m.group(1)), child))
    out.sort(key=lambda t: t[0])
    return [p for _, p in out]


def _load_attempt(attempt_dir: Path, index: int) -> AttemptSummary:
    """Read one attempt's verify_report.json / result.json into a summary.

    Missing / malformed files are tolerated — the attempt is recorded
    with gate=False and any parse error captured in ``exception``. The
    merger is designed to survive partial fleet outputs.
    """
    verify_path = attempt_dir / "verify_report.json"
    result_path = attempt_dir / "result.json"
    if not verify_path.exists():
        return AttemptSummary(
            attempt_index=index, path=attempt_dir,
            validity_gate=False,
            exception=f"missing {verify_path.name}",
        )
    try:
        verify = json.loads(verify_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return AttemptSummary(
            attempt_index=index, path=attempt_dir,
            validity_gate=False,
            exception=f"verify_report.json parse: {e}",
        )
    exception = None
    completed = True
    duration_s = 0.0
    if result_path.exists():
        try:
            r = json.loads(result_path.read_text(encoding="utf-8"))
            exception = r.get("exception")
            completed = bool(r.get("completed", True))
            duration_s = float(r.get("duration_seconds") or 0.0)
        except json.JSONDecodeError:
            pass
    gate = bool(verify.get("validity_gate", False))
    gold = list(verify.get("gold_fail_to_pass_developer") or verify.get("gold_fail_to_pass") or [])
    intersection = list(
        verify.get("pred_gold_developer_intersection")
        or verify.get("pred_gold_f2p_intersection")
        or [],
    )
    measurable = bool(verify.get("gold_recall_measurable", False))
    denominator = int(verify.get("gold_recall_denominator", len(gold)))
    recall = None
    if denominator > 0:
        recall = round(100.0 * len(intersection) / denominator, 2)
    return AttemptSummary(
        attempt_index=index,
        path=attempt_dir,
        validity_gate=gate,
        fail_to_pass=list(verify.get("fail_to_pass", [])),
        pass_to_pass=list(verify.get("pass_to_pass", [])),
        assertion_f2p=list(verify.get("assertion_f2p", [])),
        compile_f2p=list(verify.get("compile_f2p", [])),
        gold_fail_to_pass=gold,
        pred_gold_developer_intersection=intersection,
        gold_recall_measurable=measurable,
        gold_recall_denominator=denominator,
        recall_vs_gold_pct=recall,
        exception=exception,
        completed=completed,
        gate_reasons=list(verify.get("validity_notes", [])),
        duration_s=duration_s,
    )


def merge_attempts(
    case_id: str, attempt_dirs: list[Path],
) -> PassKAggregate:
    """Union F2P across gated attempts and P2P across completed attempts.

    F2P is gated credit, including zero-yield gated attempts; P2P is
    retained from every completed attempt.
    Provenance tracks which attempts produced each test.

    Empty input list → aggregate with gate=False, empty sets, and
    ``attempts_total=0``. Non-empty input where NO attempt gated →
    gate=False, empty sets, but per_attempt_summary populated.
    """
    per_attempt: list[AttemptSummary] = []
    for idx, adir in enumerate(attempt_dirs, start=1):
        per_attempt.append(_load_attempt(adir, idx))

    gated = [a for a in per_attempt if a.validity_gate]
    f2p_union: set[str] = set()
    p2p_union: set[str] = set()
    assertion_union: set[str] = set()
    compile_union: set[str] = set()
    gold_union: set[str] = set()
    intersection_union: set[str] = set()
    provenance: dict[str, list[int]] = {}
    for a in gated:
        for t in a.fail_to_pass:
            f2p_union.add(t)
            provenance.setdefault(t, []).append(a.attempt_index)
        for t in a.assertion_f2p:
            assertion_union.add(t)
        for t in a.compile_f2p:
            compile_union.add(t)
        for t in a.gold_fail_to_pass:
            gold_union.add(t)
        for t in a.pred_gold_developer_intersection:
            intersection_union.add(t)

    # P2P is the regression basis, not F2P credit. A completed attempt can
    # still prove that existing tests are stable even when it produced no
    # new F2P, so retain completed P2P sets instead of dropping them with
    # an ungated attempt.
    for a in per_attempt:
        if not a.completed:
            continue
        for t in a.pass_to_pass:
            p2p_union.add(t)
            provenance.setdefault(t, []).append(a.attempt_index)

    recall_basis = next((a for a in gated), None) or next(
        (a for a in per_attempt if a.completed), None
    ) or (per_attempt[0] if per_attempt else None)
    gold_recall_measurable = bool(recall_basis and recall_basis.gold_recall_measurable)
    gold_recall_denominator = int(recall_basis.gold_recall_denominator if recall_basis else 0)
    recall_vs_gold_pct = recall_basis.recall_vs_gold_pct if recall_basis else None

    return PassKAggregate(
        case_id=case_id,
        attempts_total=len(per_attempt),
        attempts_gated=len(gated),
        validity_gate=bool(gated),
        fail_to_pass=sorted(f2p_union),
        pass_to_pass=sorted(p2p_union),
        assertion_f2p=sorted(assertion_union),
        compile_f2p=sorted(compile_union),
        gold_fail_to_pass=sorted(gold_union),
        pred_gold_developer_intersection=sorted(intersection_union),
        gold_recall_measurable=gold_recall_measurable,
        gold_recall_denominator=gold_recall_denominator,
        recall_vs_gold_pct=recall_vs_gold_pct,
        passk_provenance=provenance,
        per_attempt_summary=per_attempt,
    )


def _best_attempt_result_json(
    attempts: list[AttemptSummary],
) -> dict[str, Any]:
    """Pick the best attempt's result.json to source docker_image /
    test_commands / etc. Preference: first gated attempt; else first
    completed attempt; else empty."""
    for a in attempts:
        if a.validity_gate:
            return _try_load_result_json(a.path)
    for a in attempts:
        if a.completed:
            return _try_load_result_json(a.path)
    return {}


def _try_load_result_json(attempt_dir: Path) -> dict[str, Any]:
    p = attempt_dir / "result.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _first_exception(attempts: list[AttemptSummary]) -> str | None:
    """First non-None exception, or None if all attempts were clean.
    Prefixed with attempt index for audit clarity."""
    for a in attempts:
        if a.exception:
            return f"attempt-{a.attempt_index:03d}: {a.exception}"
    return None


# ─── CLI ─────────────────────────────────────────────────────────────────────


@app.command()
def merge(
    case_dir: Path = typer.Argument(..., help="Directory containing case.json and enhanced-attempt-* subdirs"),
    out_dir: Path | None = typer.Option(None, "--out-dir", help="Where to write the merged verify_report.json + result.json (default: <case-dir>/enhanced)"),
) -> None:
    """Merge all enhanced-attempt-* subdirs of a case into a single
    pass@k aggregate under <case-dir>/enhanced/.

    Usage:
        swe-brew-passk-merge <case-dir>

    Assumes ``fire_fleet.sh`` (or equivalent) has already run N
    attempts and moved each ``enhanced/`` to ``enhanced-attempt-NNN/``.
    """
    if not case_dir.exists() or not case_dir.is_dir():
        console.print(f"[red]case_dir does not exist or is not a directory: {case_dir}[/red]")
        raise typer.Exit(code=1)

    case_id = case_dir.name  # matches SWE-bench-Live convention
    attempt_dirs = discover_attempt_dirs(case_dir)
    if not attempt_dirs:
        console.print(f"[yellow]No enhanced-attempt-* subdirs under {case_dir}. Nothing to merge.[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"[bold]Merging {len(attempt_dirs)} attempts for case {case_id}[/bold]")
    for a in attempt_dirs:
        console.print(f"  · {a.name}")

    aggregate = merge_attempts(case_id, attempt_dirs)

    _render_summary_table(aggregate)

    out = out_dir if out_dir is not None else (case_dir / "enhanced")
    out.mkdir(parents=True, exist_ok=True)

    result_json = aggregate.to_result_json(
        started_at=dt.datetime.now(tz=dt.UTC).isoformat(),
    )
    (out / "result.json").write_text(
        json.dumps(result_json, indent=2, sort_keys=False), encoding="utf-8",
    )
    console.print(f"[green]wrote {out}/result.json (pass@k aggregate over {aggregate.attempts_total} attempts)[/green]")


def _render_summary_table(agg: PassKAggregate) -> None:
    t = Table(title=f"pass@{agg.attempts_total} summary — {agg.case_id}")
    t.add_column("field")
    t.add_column("value", overflow="fold")
    t.add_row("attempts_total",  str(agg.attempts_total))
    t.add_row("attempts_gated",  str(agg.attempts_gated))
    t.add_row("merged_gate",     "✅" if agg.validity_gate else "❌")
    t.add_row("|F2P union|",     str(len(agg.fail_to_pass)))
    t.add_row("|P2P union|",     str(len(agg.pass_to_pass)))
    console.print(t)
    if agg.per_attempt_summary:
        pa = Table(title="per-attempt breakdown")
        pa.add_column("i", justify="right")
        pa.add_column("gate")
        pa.add_column("F2P")
        pa.add_column("P2P")
        pa.add_column("duration")
        pa.add_column("exception", overflow="fold")
        for a in agg.per_attempt_summary:
            pa.add_row(
                str(a.attempt_index),
                "✅" if a.validity_gate else "❌",
                str(len(a.fail_to_pass)),
                str(len(a.pass_to_pass)),
                f"{a.duration_s:.1f}s" if a.duration_s else "?",
                (a.exception or "")[:80],
            )
        console.print(pa)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
