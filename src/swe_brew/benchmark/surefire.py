"""Deterministic Maven-Surefire XML parser.

RepoLaunch has no shipped Java parser — its ``parselog`` node
LLM-authors a fresh one every run
(``launch/launch/agent/organize/parselog.py:128-341``). For a *reproducible*
benchmark we need parsing to be a pure function, so we ship this
deterministic parser and skip the ``parselog`` step entirely.

The parser reads Surefire's ``target/surefire-reports/TEST-*.xml`` files
into ``{TestId -> TestStatus}`` maps that ``verify.py`` compares to derive
F2P and P2P sets.

Attribution
-----------
Ported from ``data-factory/datafactory/testset_v2/test_enum.py``
(project-internal test). The status semantics ("any of failure/error/
skipped/rerunFailure ⇒ non-pass") are lifted verbatim; the surrounding
types and API surface adapt to SWE-BREW's ``TestOutcome`` model.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from swe_brew.benchmark.models import TestId, TestOutcome, TestStatus

__all__ = [
    "parse_surefire_reports",
    "surefire_report_paths",
    "clear_surefire_reports",
    "intersect_runs",
    "outcome_map",
]


_REPORT_PATTERNS = (
    "**/target/surefire-reports/**/TEST-*.xml",
    "**/target/failsafe-reports/**/TEST-*.xml",
)


def _all_report_paths(workdir: Path) -> list[Path]:
    """All Surefire/Failsafe XML reports, including TestNG aggregate files."""
    paths: set[Path] = set()
    for pattern in _REPORT_PATTERNS:
        paths.update(workdir.glob(pattern))
    return sorted(paths)


def surefire_report_paths(workdir: Path) -> list[Path]:
    """Preferred report paths, excluding TestNG's duplicate aggregate suite."""
    return [p for p in _all_report_paths(workdir) if p.name != "TEST-TestSuite.xml"]


def _aggregate_report_paths(workdir: Path) -> list[Path]:
    return [p for p in _all_report_paths(workdir) if p.name == "TEST-TestSuite.xml"]


def _classify_testcase(tc: ET.Element) -> tuple[TestStatus, str]:
    """Classify a ``<testcase>`` node.

    Surefire lists rerun failures alongside the final result on the same
    ``<testcase>`` node. Any of ``failure`` / ``error`` / ``skipped`` /
    ``rerunFailure`` ⇒ non-pass (flaky-conservative: prefer under-counting).
    """
    err = tc.find("error")
    fail = tc.find("failure")
    skip = tc.find("skipped")
    rerun = tc.find("rerunFailure")
    if err is not None:
        return TestStatus.ERROR, (err.get("message") or err.text or "")[:400]
    if fail is not None:
        return TestStatus.FAILED, (fail.get("message") or fail.text or "")[:400]
    if rerun is not None:
        return TestStatus.FAILED, "flaky rerun"
    if skip is not None:
        return TestStatus.SKIPPED, (skip.get("message") or "")[:200]
    return TestStatus.PASSED, ""


def _parse_paths(
    paths: list[Path],
    diagnostics: list[dict] | None,
) -> list[TestOutcome]:
    """Parse every Surefire XML under ``workdir`` into ``TestOutcome``s.

    Returns a de-duplicated, order-preserving list (multi-module builds
    can emit the same test in two aggregated reports; we keep the first).
    A malformed / truncated XML is skipped, not fatal — matches Surefire's
    own tolerance and matches the sister-project's behavior.
    """
    outcomes: list[TestOutcome] = []
    seen: set[TestId] = set()
    for xml_path in paths:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            if diagnostics is not None:
                diagnostics.append({
                    "path": str(xml_path),
                    "error": f"ParseError: {exc}",
                })
            continue  # truncated on kill; ignore this file
        for tc in root.iter("testcase"):
            cls = tc.get("classname") or ""
            name = tc.get("name") or ""
            if not cls or not name:
                continue
            tid: TestId = f"{cls}#{name}"
            if tid in seen:
                continue
            seen.add(tid)
            status, msg = _classify_testcase(tc)
            time_attr = tc.get("time")
            try:
                time_s = float(time_attr) if time_attr else None
            except ValueError:
                time_s = None
            outcomes.append(
                TestOutcome(test_id=tid, status=status, suite=cls, time_s=time_s, message=msg),
            )
    return outcomes


def parse_surefire_reports(
    workdir: Path,
    *,
    diagnostics: list[dict] | None = None,
) -> list[TestOutcome]:
    """Parse preferred Surefire/Failsafe reports; fall back to aggregate only."""
    preferred = surefire_report_paths(workdir)
    outcomes = _parse_paths(preferred, diagnostics)
    if not outcomes:
        # Some TestNG-only projects emit only TEST-TestSuite.xml. Use it as
        # a fallback, but never prefer it when clean per-class reports exist.
        outcomes = _parse_paths(_aggregate_report_paths(workdir), diagnostics)
    return outcomes


def outcome_map(outcomes: list[TestOutcome]) -> dict[TestId, TestStatus]:
    """Convenience: outcomes → ``{test_id: status}`` for set-diff logic in verify.py."""
    return {o.test_id: o.status for o in outcomes}


def clear_surefire_reports(workdir: Path) -> int:
    """Remove all Surefire XMLs from ``workdir``.

    Called between runs of the same container to prevent the previous
    round's XMLs from being counted in the next. Returns count removed.
    """
    removed = 0
    for xml_path in _all_report_paths(workdir):
        try:
            xml_path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def intersect_runs(runs: list[list[TestOutcome]]) -> tuple[list[TestOutcome], list[TestId]]:
    """SWE-bench-style flaky filter across N independent runs.

    Returns ``(stable, flaky_ids)``:

    * ``stable`` — the tests that PASSED in *every* run, in first-run order.
    * ``flaky_ids`` — union minus stable, sorted (audit trail).

    Any test not present in every run is dropped — bias is toward reporting
    fewer, more trustworthy tests rather than more, noisier ones.

    Ported from ``testset_v2/test_enum.py:123-136`` with expanded types
    (outcomes instead of raw ``PassedTest``).
    """
    if not runs:
        return [], []
    passing_id_sets: list[set[TestId]] = []
    for run in runs:
        passing_id_sets.append({o.test_id for o in run if o.status is TestStatus.PASSED})
    stable_ids = set.intersection(*passing_id_sets) if passing_id_sets else set()
    all_ids = set.union(*passing_id_sets) if passing_id_sets else set()
    stable = [o for o in runs[0] if o.test_id in stable_ids]
    flaky = sorted(all_ids - stable_ids)
    return stable, flaky
