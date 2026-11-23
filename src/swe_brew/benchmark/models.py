"""Pydantic schemas for the benchmark enhancer.

Kept small and boring on purpose: models describe *facts about a case*
and the *outcome of a verification run* — no logic, no methods with side effects.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, computed_field

# ─── Type aliases ────────────────────────────────────────────────────────────

#: Fully-qualified test identifier, e.g. ``com.example.module.FooTest#testBar``.
TestId = str

#: Set of test IDs that must FAIL on base and PASS after the code patch.
F2P_SET = set[TestId]

#: Set of test IDs that must PASS both before and after the code patch (regression basis).
P2P_SET = set[TestId]


class TestStatus(StrEnum):
    """Per-test outcome in a single container run.

    Aligned with SWT-Bench's ``TestStatus`` enum (``swt-bench/src/constants.py:44``).
    Their gate (``get_eval_report`` in ``grading.py:142-181``) treats any
    state that isn't PASSED / FAILED as "missing" (None) — so ERROR,
    SKIPPED, and MISSING all bucket together in the categorization.
    We preserve their enum + let SWT-Bench's semantics classify.
    """

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"      # framework-level error (missing class, compile fail, runtime exception)
    SKIPPED = "skipped"
    MISSING = "missing"  # test simply wasn't discovered / didn't run


class TestOutcome(BaseModel):
    """One test's outcome in one container run."""

    test_id: TestId
    status: TestStatus
    suite: str = ""
    time_s: float | None = None
    message: str = ""

    def is_pass(self) -> bool:
        return self.status is TestStatus.PASSED

    def is_fail(self) -> bool:
        """SWT-Bench semantic — FAILED and ERROR both count as fail.
        Everything else (PASSED, SKIPPED, MISSING) is not-fail.
        See ``swt-bench/src/grading.py::test_failed`` (line 40-43)."""
        return self.status in (TestStatus.FAILED, TestStatus.ERROR)


# ─── Inputs ──────────────────────────────────────────────────────────────────


class SweBrewCase(BaseModel):
    """A single benchmark task, extracted from ``case.json``.

    Corresponds 1:1 to a directory under
    ``$CASE_ROOT/case_id/``.
    """

    case_id: str = Field(..., description="Used verbatim as SWE-bench-Live's instance_id")
    repo_short: str = Field(..., description="Short repo name from L1_code_diff_raw.patches[0].repo")
    mr_id: str = Field(default="", description="Merge-request number for base_commit recovery (legacy records only)")
    prd_text: str = Field(..., description="L4_prd_original.description — the human problem statement (evaluation-only)")
    gold_patch: str = Field(..., description="L1_code_diff_raw.patches[0].patch — the full gold diff (unified format)")
    changed_files: list[str] = Field(default_factory=list, description="Enumerated changed paths from patches[0].changed_files")
    setup_cmds: list[str] = Field(default_factory=list, description="Prior successful setup commands, if available")
    test_cmds: list[str] = Field(default_factory=list, description="Prior successful test commands, if available")
    avoid_commands: list[str] = Field(default_factory=list, description="Prior failed/timed-out commands to avoid")

    # Recovered / derived (optional, filled in by later stages)
    base_commit: str | None = Field(default=None, description="Real base commit SHA, once recovered")
    remote_url: str | None = Field(default=None, description="Full clone URL for the repo (the configured Git host)")
    local_repo: "Path | None" = Field(default=None, description="Explicit local Git clone path (simple release schema)")
    head_commit: str | None = Field(default=None, description="Post-change commit SHA (simple release schema)")

    @computed_field
    @property
    def instance_id(self) -> str:
        """Alias to make SWE-bench-Live's ``instance_id`` semantics explicit."""
        return self.case_id


class SplitPatches(BaseModel):
    """Result of splitting the gold diff into code vs test."""

    code_patch: str = Field(..., description="Non-test hunks — the answer we withhold from the agent")
    test_patch: str = Field(..., description="Test hunks — may be empty for the majority of cases")
    code_files: list[str] = Field(default_factory=list)
    test_files: list[str] = Field(default_factory=list)
    classification: str = "git-native"

    @computed_field
    @property
    def has_gold_tests(self) -> bool:
        return bool(self.test_files)


# ─── Outputs ─────────────────────────────────────────────────────────────────

Configuration = Literal[
    "base_pre",  "base_post",   # no test patch                  vs no test patch + code patch
    "gold_pre",  "gold_post",   # gold test patch                vs gold test patch + code patch
    "pred_pre",  "pred_post",   # candidate test patch           vs candidate test patch + code patch
]


class RunOutcome(BaseModel):
    """One container run in the 6-way matrix."""

    configuration: Configuration
    tests: list[TestOutcome] = Field(default_factory=list)
    ran_seconds: float = 0.0
    exit_code: int | None = None
    stdout_tail: str = ""       # last ~4KB of stdout for auditing
    stderr_tail: str = ""

    def pass_set(self) -> set[TestId]:
        return {t.test_id for t in self.tests if t.is_pass()}

    def fail_set(self) -> set[TestId]:
        return {t.test_id for t in self.tests if t.is_fail()}


class VerifyReport(BaseModel):
    """The full auditable trace of one case's benchmark enhancement.

    Written to ``<case_dir>/enhanced/verify_report.json``. The lighter
    ``test_results.json`` is a projection of this.
    """

    case_id: str
    image_tag: str
    test_command: str = ""
    runs: dict[Configuration, RunOutcome] = Field(default_factory=dict)

    # Candidate acceptance gate: candidate-added tests execute when present,
    # preserve base F2P, and introduce no regressions. Zero new F2P is accepted.
    validity_gate: bool = False
    validity_notes: list[str] = Field(default_factory=list)

    # Derived sets we ship downstream (sorted lists, JSON-friendly).
    # All fields are execution-verified — computed from the six live Maven
    # runs in `runs`, never from static parsing.
    #
    # pred_* — the candidate's earned credit.
    fail_to_pass: list[TestId] = Field(default_factory=list)
    pass_to_pass: list[TestId] = Field(default_factory=list)
    assertion_f2p: list[TestId] = Field(default_factory=list)
    compile_f2p: list[TestId] = Field(default_factory=list)
    gold_recall_measurable: bool = False
    gold_recall_denominator: int = 0

    # gold_* — the developer's own tests, run under the same 6-way matrix
    # as pred. This is the "reasonable ceiling proxy" for the case:
    # tests the human wrote that genuinely go fail→pass with their patch.
    # Preserved for backward compatibility — aliases gold_fail_to_pass_developer.
    gold_fail_to_pass: list[TestId] = Field(default_factory=list)
    gold_pass_to_pass: list[TestId] = Field(default_factory=list)
    pred_gold_f2p_intersection: list[TestId] = Field(default_factory=list)

    # Three-way gold scoping:
    #   developer  — case.json.test_files subset (SWT-Bench canonical)
    #   discovered — organize agent's full pre-existing test suite
    #   local      — tests in organize-validated Maven modules
    # ``gold_fail_to_pass`` above is the developer-scope default.
    gold_fail_to_pass_developer:      list[TestId] = Field(default_factory=list)
    gold_fail_to_pass_discovered:     list[TestId] = Field(default_factory=list)
    gold_fail_to_pass_local:          list[TestId] = Field(default_factory=list)
    gold_pass_to_pass_developer:      list[TestId] = Field(default_factory=list)
    gold_pass_to_pass_discovered:     list[TestId] = Field(default_factory=list)
    gold_pass_to_pass_local:          list[TestId] = Field(default_factory=list)
    pred_gold_developer_intersection:  list[TestId] = Field(default_factory=list)
    pred_gold_discovered_intersection: list[TestId] = Field(default_factory=list)
    pred_gold_local_intersection:      list[TestId] = Field(default_factory=list)

    # SWT-Bench-canonical 3×3 categorization reports, one per candidate
    # source (base / gold / pred). Emitted verbatim from
    # :func:`swe_brew.benchmark.swt_report.get_eval_report` (grounded 1:1
    # on ``swt-bench/src/grading.py:142-181``).
    #
    # Each report is a dict with 5 keys: FAIL_TO_PASS, PASS_TO_PASS,
    # FAIL_TO_FAIL, PASS_TO_FAIL, UNMATCHED. F2P bucket is x2p + f2p
    # per SWT-Bench's semantic (tests missing/failed at pre AND
    # passed at post).
    swt_report_base:   dict[str, list[TestId]] = Field(default_factory=dict)
    swt_report_gold:   dict[str, list[TestId]] = Field(default_factory=dict)
    swt_report_pred:   dict[str, list[TestId]] = Field(default_factory=dict)

    # SWE-bench-Live-compatible resolved report over the same six runs.
    # Populated by ``swebench_live_resolved_report``. Default expected sets
    # are the gold six-way ``gold_f2p`` / ``gold_p2p``.
    swebench_live_resolved: dict[str, object] = Field(default_factory=dict)

    # JaCoCo executable-line coverage delta over the gold code patch.
    # Empty when ``collect_coverage=False`` (the normal six-way run).
    coverage_delta: dict[str, object] = Field(default_factory=dict)

    # Bookkeeping
    generated_at: str = ""
    llm_model: str = ""
    llm_stub: bool = False
    raw_gold_patch_parse: str = "unidiff-parsed"
    code_patch_source: str = "git-native"
    notes: list[str] = Field(default_factory=list)

    # Reproducibility manifest. Records the exact container environment
    # used for the six-way matrix — image tag, platform, Maven version,
    # JDK version, settings.xml hash, mount points. Mirrors the shape
    # of SWE-bench-Live/launch/run.py's ``result.json.docker_image_layers``
    # so a fleet-mode runner can rebuild the same env byte-for-byte.
    docker_layers: dict[str, str] = Field(default_factory=dict)


class TestResults(BaseModel):
    """The lean, scorecard-facing artefact.

    Written to ``<case_dir>/enhanced/test_results.json`` and consumed by
    ``code_ai_evaluator``'s ``compile_pass`` / ``test_code_pass_rate`` dims.
    """

    case_id: str
    image_tag: str
    test_command: str
    FAIL_TO_PASS: list[TestId] = Field(default_factory=list)
    PASS_TO_PASS: list[TestId] = Field(default_factory=list)
    ASSERTION_F2P: list[TestId] = Field(default_factory=list)
    COMPILE_F2P: list[TestId] = Field(default_factory=list)
    validity_gate: bool = False
    meta: dict[str, str] = Field(default_factory=dict)


# ─── Filesystem convention ───────────────────────────────────────────────────


def case_dir(corpus_dir: Path, case_id: str) -> Path:
    """Given ``$CASE_ROOT`` and a case_id, return the on-disk directory."""
    return corpus_dir / case_id


def enhanced_dir(case_dir_: Path) -> Path:
    """Where we write our artefacts. Never overwrites anything shipped in case.json."""
    out = case_dir_ / "enhanced"
    out.mkdir(parents=True, exist_ok=True)
    return out
