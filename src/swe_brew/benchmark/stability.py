"""Compare repeated verify reports for execution-stability screening.

The six-way verify already checks the expected fail-to-pass transition
once. CodeMidas additionally reruns selected solutions in fresh containers
and drops environments whose outcomes are unstable. This module implements
the comparison side of that check so post-run curation can separate stable
behavioral F2P from flaky tests without running Docker itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["StabilityReport", "compare_reruns"]


def _set_from_report(report: dict, key: str) -> set[str]:
    value = report.get(key, [])
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def _new_p2f(report: dict) -> set[str]:
    """Return candidate-induced P2F regressions recorded in a report."""
    swt = report.get("swt_report_pred") or {}
    return {str(item) for item in swt.get("PASS_TO_FAIL", [])}


@dataclass(frozen=True)
class StabilityReport:
    """Comparison of one reference report against rerun reports."""

    reruns: int = 0
    reference_f2p: set[str] = field(default_factory=set)
    reference_p2p: set[str] = field(default_factory=set)
    observed_f2p_sets: tuple[set[str], ...] = ()
    observed_p2p_sets: tuple[set[str], ...] = ()
    stable_f2p: set[str] = field(default_factory=set)
    stable_p2p: set[str] = field(default_factory=set)
    flaky_f2p: set[str] = field(default_factory=set)
    vanished_f2p: set[str] = field(default_factory=set)
    introduced_f2p: set[str] = field(default_factory=set)
    p2f_regressions: set[str] = field(default_factory=set)
    reasons: list[str] = field(default_factory=list)

    @property
    def stable(self) -> bool:
        """True only when every observed set matches the reference."""
        return not self.reasons


def compare_reruns(
    reference: dict,
    reruns: list[dict],
) -> StabilityReport:
    """Compare F2P/P2P sets across repeated executions.

    Parameters
    ----------
    reference:
        Canonical ``verify_report.json`` produced by the original run.
    reruns:
        One or more repeated ``verify_report.json`` dictionaries. Order is
        preserved in the returned observation tuples.
    """
    ref_f2p = _set_from_report(reference, "fail_to_pass")
    ref_p2p = _set_from_report(reference, "pass_to_pass")
    f2p_sets = [_set_from_report(r, "fail_to_pass") for r in reruns]
    p2p_sets = [_set_from_report(r, "pass_to_pass") for r in reruns]
    regressions = {
        test_id
        for r in reruns
        for test_id in _new_p2f(r)
    }

    reasons: list[str] = []
    stable_f2p = ref_f2p.intersection(*f2p_sets) if f2p_sets else ref_f2p
    stable_p2p = ref_p2p.intersection(*p2p_sets) if p2p_sets else ref_p2p
    flaky_f2p = set()
    introduced_f2p = set()
    if f2p_sets:
        observed_union = set().union(*f2p_sets)
        flaky_f2p = (ref_f2p | observed_union) - stable_f2p
        introduced_f2p = observed_union - ref_f2p

    vanished_f2p = ref_f2p - stable_f2p
    if f2p_sets and any(f2p_set != ref_f2p for f2p_set in f2p_sets):
        reasons.append("F2P set changed across repeated runs")
    if p2p_sets and any(p2p_set != ref_p2p for p2p_set in p2p_sets):
        reasons.append("P2P set changed across repeated runs")
    if regressions:
        reasons.append(
            f"{len(regressions)} PASS->FAIL regression(s) observed in reruns"
        )

    return StabilityReport(
        reruns=len(reruns),
        reference_f2p=ref_f2p,
        reference_p2p=ref_p2p,
        observed_f2p_sets=tuple(f2p_sets),
        observed_p2p_sets=tuple(p2p_sets),
        stable_f2p=stable_f2p,
        stable_p2p=stable_p2p,
        flaky_f2p=flaky_f2p,
        vanished_f2p=vanished_f2p,
        introduced_f2p=introduced_f2p,
        p2f_regressions=regressions,
        reasons=reasons,
    )
