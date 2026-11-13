"""Six-way execution matrix + candidate acceptance gate.

For each of the six configurations
``{base | gold | pred} × {pre | post}`` we

1. reset the working tree to base_commit,
2. apply the test_patch of the row (base⇒none, gold⇒gold_test, pred⇒candidate),
3. optionally apply the gold code_patch (``post`` variants),
4. run ``mvn test`` (or the passed ``test_command``),
5. parse Surefire XMLs → ``{test_id: status}`` map.

Then we apply a candidate acceptance gate:

    candidate-added tests execute when present    AND
    pred_f2p ⊇ base_f2p                AND
    no candidate-induced P2F / F2F regressions

Where ``f2p(cfg_pre, cfg_post)`` = tests that FAIL in ``cfg_pre`` and PASS
in ``cfg_post``.

This deliberately does NOT inherit SWT-Bench's benchmark-only requirement
that a candidate must add at least one new F2P. Zero-yield but structurally
clean attempts are accepted and collected; new F2P is a preferred signal,
not a rejection condition.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from swe_brew.benchmark.container import TestSandbox
from swe_brew.benchmark.coverage import (
    augment_test_command_for_paths,
    build_jacoco_test_command,
    coverage_delta,
    find_jacoco_xmls,
    merge_jacoco_xmls,
)
from swe_brew.benchmark.models import (
    Configuration,
    RunOutcome,
    SplitPatches,
    SweBrewCase,
    TestId,
    TestStatus,
    VerifyReport,
)
from swe_brew.benchmark.runtime import runtime_spec
from swe_brew.benchmark.surefire import (
    parse_surefire_reports,
)
from swe_brew.benchmark.swt_report import get_eval_report
from swe_brew.benchmark.test_discovery import expected_outcomes_from_test_files

__all__ = [
    "verify_case",
    "SWTBenchGateResult",
    "SweBenchLiveResolvedReport",
    "swebench_live_resolved_report",
    "swt_gate",
]


# Default Maven invocation. Callers can override with a repo-specific command.
#
# Flag choices — each addresses a known Java-benchmark failure mode:
#   -B                                : batch mode, no interactive prompts.
#   -q                                : quiet; only errors + Surefire summaries.
#   -Dmaven.test.failure.ignore=true  : test *failures* (assertion fails) don't
#                                       abort the build — F2P/P2P computation
#                                       needs to see both pre-state failures
#                                       AND post-state passes.
#   -Dmaven.compiler.failOnError=false: **test-compile* errors* don't poison
#                                       an entire module. When a conflict
#                                       cascade leaves ONE test file in a
#                                       module referencing an orphaned symbol,
#                                       the OTHER (compile-clean) test files
#                                       in the same module can still run.
#                                       This is what upstream SWT-Bench gets
#                                       for free from Python (pytest treats
#                                       missing imports as per-test ERROR);
#                                       we synthesise the same effect for
#                                       Maven at the module granularity.
#   -Dmaven.javadoc.skip=true         : skip javadoc generation (irrelevant
#                                       and slow).
#
# Timeout ownership: a single outer sandbox.exec ceiling plus a
# per-trial watchdog. We deliberately avoid granular Surefire flags
# (``-Dsurefire.timeout``, ``-Dsurefire.exitTimeout``) — they split
# timeout ownership across three layers (tool subprocess, Surefire,
# JVM) and false-fire on legitimately slow integration tests.
# Kept for tests/backward compatibility. Runtime callers should use
# ``runtime_spec().test_command`` so environment overrides loaded after
# module import are honoured.
DEFAULT_TEST_COMMAND = runtime_spec().test_command


# ─── Candidate acceptance gate ─────────────────────────────────────────────


@dataclass(frozen=True)
class SWTBenchGateResult:
    """Result of the candidate acceptance gate.

    F2P sets are computed via ``swt_report.get_eval_report`` so the
    bucket math remains SWT-Bench-compatible. The gate accepts zero new
    F2P when the candidate still executed cleanly and introduced no
    regressions.
    """

    valid: bool
    reasons: list[str]
    pred_f2p: set[TestId]
    base_f2p: set[TestId]
    p2p: set[TestId]
    gold_f2p: set[TestId] = None                           # type: ignore[assignment]
    gold_p2p: set[TestId] = None                           # type: ignore[assignment]

    def __post_init__(self) -> None:
        for attr in ("gold_f2p", "gold_p2p"):
            if getattr(self, attr) is None:
                object.__setattr__(self, attr, set())


def swt_gate(runs: dict[Configuration, RunOutcome]) -> SWTBenchGateResult:
    """Apply the candidate acceptance rule to the six runs.

    Every bucket computation routes through
    ``swt_report.get_eval_report`` — the byte-verbatim port of
    ``swt-bench/src/grading.py`` — so ``validity_notes`` and the
    parallel ``swt_report_*`` fields in the emitted verify_report
    always agree on the same six runs. Historical note: a prior
    version of this function had its own inline bucket math with
    subtly different semantics (``ERROR`` treated as ``FAILED``
    rather than coerced to ``None``); the two graders produced
    contradictory numbers on real fleet data. That inline math is
    gone.

    Returns a rich result with the F2P set that should be shipped and a
    list of human-readable blocking reasons (empty when ``valid``).
    """
    from swe_brew.benchmark.swt_report import (
        FAIL_TO_FAIL,
        FAIL_TO_PASS,
        PASS_TO_FAIL,
        PASS_TO_PASS,
        get_eval_report,
    )

    def m(cfg: Configuration) -> dict[TestId, TestStatus]:
        r = runs.get(cfg)
        return {o.test_id: o.status for o in r.tests} if r else {}

    base_pre, base_post = m("base_pre"), m("base_post")
    gold_pre, gold_post = m("gold_pre"), m("gold_post")
    pred_pre, pred_post = m("pred_pre"), m("pred_post")

    # Single source of truth: SWT-Bench-canonical bucket computation.
    base_report = get_eval_report(base_pre, base_post)
    pred_report = get_eval_report(pred_pre, pred_post)
    gold_report = get_eval_report(gold_pre, gold_post)

    base_f2p = set(base_report[FAIL_TO_PASS])
    pred_f2p = set(pred_report[FAIL_TO_PASS])
    pred_p2p = set(pred_report[PASS_TO_PASS])
    gold_f2p = set(gold_report[FAIL_TO_PASS])
    gold_p2p = set(gold_report[PASS_TO_PASS])

    reasons: list[str] = []
    executed_statuses = {TestStatus.PASSED, TestStatus.FAILED, TestStatus.ERROR}

    def has_executed(outcomes: dict[TestId, TestStatus]) -> bool:
        return any(status in executed_statuses for status in outcomes.values())

    if not pred_pre or not has_executed(pred_pre):
        reasons.append("pred_pre run produced no executed test outcomes (compile/runner failure?)")
    if not pred_post or not has_executed(pred_post):
        reasons.append("pred_post run produced no executed test outcomes")

    candidate_test_ids = (
        (set(pred_pre) | set(pred_post))
        - (set(base_pre) | set(base_post))
    )
    if candidate_test_ids:
        candidate_executed_post = any(
            tid in candidate_test_ids and status in executed_statuses
            for tid, status in pred_post.items()
        )
        if not candidate_executed_post:
            reasons.append(
                f"candidate added {len(candidate_test_ids)} test ids but none "
                "executed in pred_post (compile/scope/runner failure?)"
            )

    if not (pred_f2p >= base_f2p):
        missing = base_f2p - pred_f2p
        reasons.append(f"pred_f2p missing {len(missing)} tests from base_f2p (e.g. {sorted(missing)[:3]})")

    # Regression + fail-to-fail via the SAME grader. Compares to the
    # base_report's corresponding bucket so we surface CANDIDATE-INDUCED
    # regressions, not pre-existing project flakes.
    new_p2f = set(pred_report[PASS_TO_FAIL]) - set(base_report[PASS_TO_FAIL])
    if new_p2f:
        reasons.append(f"candidate introduces {len(new_p2f)} new PASS→FAIL regressions (e.g. {sorted(new_p2f)[:3]})")
    new_f2f = set(pred_report[FAIL_TO_FAIL]) - set(base_report[FAIL_TO_FAIL])
    if new_f2f:
        reasons.append(f"candidate introduces {len(new_f2f)} new FAIL→FAIL tests (e.g. {sorted(new_f2f)[:3]})")

    return SWTBenchGateResult(
        valid=not reasons,
        reasons=reasons,
        pred_f2p=pred_f2p,
        base_f2p=base_f2p,
        p2p=pred_p2p,
        gold_f2p=gold_f2p,
        gold_p2p=gold_p2p,
    )


@dataclass(frozen=True)
class SweBenchLiveResolvedReport:
    """Resolved report over our six-way runs.

    The resolved report produces a ``report.json`` with
    ``resolved``, ``PASS_TO_PASS.success/failure``, and
    ``FAIL_TO_PASS.success/failure``. This adapter emits the same semantic
    shape from ``pred_post`` and the gold-derived expected test sets. It does
    not replace the richer SWT-Bench gate; it is the comparable resolved
    summary for reporting against external baselines.
    """

    resolved: bool
    pass_to_pass_success: list[TestId]
    pass_to_pass_failure: list[TestId]
    fail_to_pass_success: list[TestId]
    fail_to_pass_failure: list[TestId]

    def as_dict(self) -> dict[str, object]:
        return {
            "resolved": self.resolved,
            "PASS_TO_PASS": {
                "success": sorted(self.pass_to_pass_success),
                "failure": sorted(self.pass_to_pass_failure),
            },
            "FAIL_TO_PASS": {
                "success": sorted(self.fail_to_pass_success),
                "failure": sorted(self.fail_to_pass_failure),
            },
        }


def swebench_live_resolved_report(
    runs: dict[Configuration, RunOutcome],
    *,
    expected_pass_to_pass: set[TestId] | None = None,
    expected_fail_to_pass: set[TestId] | None = None,
) -> SweBenchLiveResolvedReport:
    """Emit resolved semantics from the six-way run map.

    Default expected sets come from the gold six-way rows
    (``gold_pre`` -> ``gold_post``) via ``swt_gate``, so no separate test
    metadata is required. ``pred_post`` is the candidate's post-patch result.
    """

    def outcomes(cfg: Configuration) -> dict[TestId, TestStatus]:
        run = runs.get(cfg)
        return {o.test_id: o.status for o in run.tests} if run else {}

    pred_post = outcomes("pred_post")
    gold = swt_gate(runs)
    expected_p2p = set(expected_pass_to_pass if expected_pass_to_pass is not None else gold.gold_p2p)
    expected_f2p = set(expected_fail_to_pass if expected_fail_to_pass is not None else gold.gold_f2p)

    passed = {tid for tid, status in pred_post.items() if status == TestStatus.PASSED}
    p2p_success = expected_p2p & passed
    p2p_failure = expected_p2p - passed
    f2p_success = expected_f2p & passed
    f2p_failure = expected_f2p - passed
    resolved = not p2p_failure and not f2p_failure

    return SweBenchLiveResolvedReport(
        resolved=resolved,
        pass_to_pass_success=list(p2p_success),
        pass_to_pass_failure=list(p2p_failure),
        fail_to_pass_success=list(f2p_success),
        fail_to_pass_failure=list(f2p_failure),
    )


def new_fail_to_fail_ids(runs: dict[Configuration, RunOutcome]) -> set[TestId]:
    """Return candidate-added FAIL→FAIL test IDs from the six-way runs."""
    from swe_brew.benchmark.swt_report import (
        FAIL_TO_FAIL,
        get_eval_report,
    )

    def m(cfg: Configuration) -> dict[TestId, TestStatus]:
        r = runs.get(cfg)
        return {o.test_id: o.status for o in r.tests} if r else {}

    base_report = get_eval_report(m("base_pre"), m("base_post"))
    pred_report = get_eval_report(m("pred_pre"), m("pred_post"))
    return set(pred_report[FAIL_TO_FAIL]) - set(base_report[FAIL_TO_FAIL])


def new_pass_to_fail_ids(runs: dict[Configuration, RunOutcome]) -> set[TestId]:
    """Return candidate-added PASS→FAIL test IDs from the six-way runs.

    These are candidate tests that pass against the pre-patch tree but fail
    against the post-patch tree. They assert the old behavior, so the
    precision stage should remove them rather than reject the remaining
    candidate harvest.
    """
    from swe_brew.benchmark.swt_report import (
        PASS_TO_FAIL,
        get_eval_report,
    )

    def m(cfg: Configuration) -> dict[TestId, TestStatus]:
        r = runs.get(cfg)
        return {o.test_id: o.status for o in r.tests} if r else {}

    base_report = get_eval_report(m("base_pre"), m("base_post"))
    pred_report = get_eval_report(m("pred_pre"), m("pred_post"))
    return set(pred_report[PASS_TO_FAIL]) - set(base_report[PASS_TO_FAIL])


# ─── the runner ──────────────────────────────────────────────────────────────


def _run_config(
    sandbox: TestSandbox,
    cfg: Configuration,
    split: SplitPatches,
    candidate_test_patch: str,
    test_command: str,
    *,
    pipeline_patch: str = "",
    pred_code_patch: str | None = None,
) -> RunOutcome:
    """Materialise + run one of the six configurations."""
    # 1. clean state
    sandbox.reset_working_tree()
    sandbox.clear_surefire_reports()

    # 2. select the test_patch for this row
    if cfg.startswith("base"):
        test_patch = ""
    elif cfg.startswith("gold"):
        test_patch = split.test_patch
    elif cfg.startswith("pred"):
        test_patch = candidate_test_patch
    else:
        raise ValueError(f"unknown configuration {cfg}")

    # 3. Reapply setup/organize build fixes before grading. Verify
    # resets each leg to the base commit, so without this patch any POM
    # or settings work performed by setup/organize is lost and tests can
    # fail for reasons the pipeline already diagnosed and fixed.
    pipeline_apply = sandbox.apply_patch(pipeline_patch) if pipeline_patch else None

    # 4. apply test_patch first, then code_patch iff *_post.
    # For pred legs, ``pred_code_patch`` overrides the withheld gold code patch
    # so a solver-produced code patch can be graded against the same test set.
    test_apply = sandbox.apply_patch(test_patch) if test_patch else None
    if cfg.endswith("_post"):
        code_patch = (
            pred_code_patch
            if cfg.startswith("pred") and pred_code_patch is not None
            else split.code_patch
        )
        code_apply = sandbox.apply_patch(code_patch)
    else:
        code_apply = None

    # 5. run tests
    started = dt.datetime.now(tz=dt.UTC)
    exec_res = sandbox.exec(
        test_command,
        timeout=runtime_spec().command_timeout_seconds,
    )
    ran_s = (dt.datetime.now(tz=dt.UTC) - started).total_seconds()

    # 6. parse surefire (real outcomes for tests that actually compiled + ran)
    parse_diagnostics: list[dict] = []
    outcomes = parse_surefire_reports(
        sandbox.snapshot.work_dir,
        diagnostics=parse_diagnostics,
    )

    # Java infra localization of SWT-Bench's grading semantic via
    # source-tree walk. We enumerate ``@Test`` methods from the APPLIED
    # SOURCE TREE (post-``git apply``, the .java files exist on disk
    # with their annotations intact); walking is more robust than any
    # diff-parser and immune to strict unidiff rejections.
    from swe_brew.benchmark.test_discovery import enumerate_test_files_in_tree

    all_test_files = enumerate_test_files_in_tree(sandbox.snapshot.work_dir)
    if all_test_files:
        try:
            # Backfill outcomes for every ``@Test`` in the source tree
            # that Surefire didn't report. Default status is ``MISSING``
            # — most of these tests are simply outside the mvn ``-pl``
            # scope, not compile-failing. The message is chosen so an
            # operator reading the verify_report can distinguish the
            # two cases; both coerce to ``None`` in the SWT-Bench
            # grader so F2P math is unaffected.
            expected = expected_outcomes_from_test_files(
                sandbox.snapshot.work_dir,
                all_test_files,
                message="test not present in Surefire reports "
                        "(likely outside mvn -pl scope or compile-failed)",
            )
            seen = {o.test_id for o in outcomes}
            for e in expected:
                if e.test_id not in seen:
                    outcomes.append(e)
        except Exception as e:                     # noqa: BLE001
            # Defensive — enumerate should not raise, but if it does
            # (e.g. permission denied on a subdir), don't kill the run.
            # Surefire outcomes are still valid.
            pass

    # 7. Notes: annotate with apply-time errors (audit trail)
    notes = [
        f"surefire_parse_error={d['path']}: {d['error']}"
        for d in parse_diagnostics
    ]
    if pipeline_apply is not None and not pipeline_apply.ok:
        notes.append(f"pipeline_patch apply exit={pipeline_apply.exit_code}: {pipeline_apply.stderr[-200:] or pipeline_apply.stdout[-200:]}")
    if test_apply is not None and not test_apply.ok:
        notes.append(f"test_patch apply exit={test_apply.exit_code}: {test_apply.stderr[-200:] or test_apply.stdout[-200:]}")
    if code_apply is not None and not code_apply.ok:
        notes.append(f"code_patch apply exit={code_apply.exit_code}: {code_apply.stderr[-200:] or code_apply.stdout[-200:]}")

    return RunOutcome(
        configuration=cfg,
        tests=outcomes,
        ran_seconds=ran_s,
        exit_code=exec_res.exit_code,
        stdout_tail=exec_res.stdout[-4000:],
        stderr_tail=" | ".join(notes),
    )


def _run_coverage_config(
    sandbox: TestSandbox,
    cfg: Configuration,
    split: SplitPatches,
    candidate_test_patch: str,
    test_command: str,
    *,
    pipeline_patch: str = "",
    pred_code_patch: str | None = None,
) -> dict[str, object]:
    """Run one configuration under JaCoCo and return its line coverage.

    This is a focused twin of :func:`_run_config`: same patch materialisation
    order, but the Maven invocation wraps the test goal with JaCoCo and the
    result is executable/covered line maps instead of Surefire outcomes.
    """
    sandbox.reset_working_tree()
    sandbox.clear_surefire_reports()

    if cfg.startswith("base"):
        test_patch = ""
    elif cfg.startswith("gold"):
        test_patch = split.test_patch
    elif cfg.startswith("pred"):
        test_patch = candidate_test_patch
    else:
        raise ValueError(f"unknown configuration {cfg}")

    if pipeline_patch:
        sandbox.apply_patch(pipeline_patch)
    if test_patch:
        sandbox.apply_patch(test_patch)
    if cfg.endswith("_post"):
        code_patch = (
            pred_code_patch
            if cfg.startswith("pred") and pred_code_patch is not None
            else split.code_patch
        )
        sandbox.apply_patch(code_patch)

    scoped_test_command = augment_test_command_for_paths(
        test_command,
        split.code_files,
    )
    coverage_command = build_jacoco_test_command(scoped_test_command)
    exec_res = sandbox.exec(
        coverage_command,
        timeout=runtime_spec().command_timeout_seconds,
    )

    xml_paths = find_jacoco_xmls(sandbox.snapshot.work_dir)
    return {
        "exit_code": exec_res.exit_code,
        "coverage": merge_jacoco_xmls(xml_paths),
        "xml_reports": [str(p.relative_to(sandbox.snapshot.work_dir)) for p in xml_paths],
    }


def _capture_docker_layers(sandbox: TestSandbox) -> dict[str, str]:
    """Snapshot the container environment for the reproducibility manifest.

    Mirrors the intent of the standard benchmark runner's
    ``result.json.docker_image_layers`` — record everything a
    fleet-mode runner would need to rebuild byte-for-byte.
    """
    import hashlib

    layers: dict[str, str] = {
        "image_tag": sandbox.image_tag,
        "container_short_id": sandbox.container.short_id,
    }
    # Capture java + mvn versions from inside the container. We
    # prepend the full PATH the container sets in its environment
    # (see :func:`start_sandbox` for the canonical value) so that
    # ``java -version`` resolves even without ``/etc/profile`` sourcing.
    # eclipse-temurin images install the JDK under /opt/java/openjdk, but
    # the container PATH used by ``start_sandbox`` only prepends Maven.
    _PATH = (
        "/opt/java/openjdk/bin:"
        "/usr/lib/jvm/java-21-openjdk-amd64/bin:"
        "/opt/maven/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    for label, cmd in (("java_version", "java -version"), ("mvn_version", "mvn --version")):
        try:
            r = sandbox.exec(f"PATH={_PATH} {cmd}", timeout=30)
            # `java -version` writes to STDERR; our sandbox.exec merges
            # streams so it lands in stdout.
            if r.stdout.strip():
                layers[label] = r.stdout.strip().splitlines()[0]
            else:
                layers[label] = f"exit={r.exit_code} (no output)"
        except Exception as e:                     # noqa: BLE001
            layers[label] = f"error: {e!s}"
    # settings.xml is bind-mounted from ~/.swe-brew-m2-cache — hash on
    # the host side.
    try:
        settings_path = Path.home() / ".swe-brew-m2-cache" / "settings.xml"
        if settings_path.exists():
            layers["settings_xml_sha256"] = hashlib.sha256(settings_path.read_bytes()).hexdigest()[:16]
        else:
            layers["settings_xml_sha256"] = "missing"
    except Exception as e:  # noqa: BLE001
        layers["settings_xml_sha256"] = f"error: {e!s}"
    # Mount points are fixed by ``start_sandbox`` — record for the manifest.
    layers["mounts"] = "/testbed=work_dir; /root/.m2=~/.swe-brew-m2-cache; /opt/maven=~/.swe-brew-tools/apache-maven-*"
    return layers


def _run_swt_report(
    runs: dict[Configuration, RunOutcome],
    source: str,
) -> dict[str, list[TestId]]:
    """SWT-Bench-canonical 3×3 report for one candidate source
    (base / gold / pred). Extracts <source>_pre and <source>_post from
    the runs dict and hands them to ``get_eval_report`` verbatim.

    Returns the 5-bucket dict per ``swt_report.get_eval_report``:
    FAIL_TO_PASS / PASS_TO_PASS / FAIL_TO_FAIL / PASS_TO_FAIL / UNMATCHED.
    """
    pre_key  = f"{source}_pre"                   # type: ignore[assignment]
    post_key = f"{source}_post"                  # type: ignore[assignment]
    pre_run  = runs.get(pre_key)                 # type: ignore[arg-type]
    post_run = runs.get(post_key)                # type: ignore[arg-type]
    pre_map:  dict[TestId, TestStatus] = {o.test_id: o.status for o in (pre_run.tests  if pre_run else [])}
    post_map: dict[TestId, TestStatus] = {o.test_id: o.status for o in (post_run.tests if post_run else [])}
    return get_eval_report(pre_map, post_map)


def classify_f2p(
    runs: dict[Configuration, RunOutcome],
    f2p_ids: set[TestId],
) -> tuple[set[TestId], set[TestId]]:
    """Split earned F2P into assertion-driven and compile-driven buckets.

    Java needs a compile-fail-at-pre analogue because a generated test can
    reference an API added by the reference patch. The canonical gate treats
    both as F2P, but reporting them separately keeps recall audits honest.
    """
    pred_pre = {o.test_id: o.status for o in runs["pred_pre"].tests}
    pred_post = {o.test_id: o.status for o in runs["pred_post"].tests}
    assertion: set[TestId] = set()
    compile_only: set[TestId] = set()
    for tid in f2p_ids:
        if pred_post.get(tid) is not TestStatus.PASSED:
            continue
        if pred_pre.get(tid) is TestStatus.FAILED:
            assertion.add(tid)
        else:
            compile_only.add(tid)
    return assertion, compile_only


def _test_ids_in_scope(scope_files: list[str], test_ids: set[TestId]) -> set[TestId]:
    """Filter ``test_ids`` (of form ``fqcn#method``) to those whose file
    path is in ``scope_files``. Matching is by class simple-name
    heuristic: a test file ``…/src/test/java/com/foo/BarTest.java`` matches
    test ids of form ``com.foo.BarTest#…``.

    Handles the case where scope_files use POSIX-slash paths and test_ids
    use dot-separated FQCNs.
    """
    marker = runtime_spec().test_path_marker
    scope_classes: set[str] = set()
    for p in scope_files:
        # Extract FQCN from <module>/<test-marker>/<pkg-as-dirs>/<Class>.java
        i = p.find(marker)
        if i < 0:
            # Also try the case where the path *starts* with the marker.
            i = p.find(marker.lstrip("/"))
            if i == 0:
                pkg_and_file = p[len(marker.lstrip("/")):]
            else:
                continue
        else:
            pkg_and_file = p[i + len(marker):]
        pkg_and_file = pkg_and_file.rsplit(".java", 1)[0]
        fqcn = pkg_and_file.replace("/", ".")
        scope_classes.add(fqcn)
    return {tid for tid in test_ids if tid.rsplit("#", 1)[0] in scope_classes}


def verify_case(
    case: SweBrewCase,
    split: SplitPatches,
    candidate_test_patch: str,
    sandbox: TestSandbox,
    *,
    test_command: str = "",
    is_stub: bool = False,
    llm_model: str = "",
    test_scope_developer: list[str] | None = None,
    test_scope_discovered: list[str] | None = None,
    test_scope_local: list[str] | None = None,
    raw_gold_patch_parse: str = "unidiff-parsed",
    code_patch_source: str = "git-native",
    pipeline_patch: str = "",
    pred_code_patch: str | None = None,
    expected_pass_to_pass: set[TestId] | None = None,
    expected_fail_to_pass: set[TestId] | None = None,
    collect_coverage: bool = False,
) -> VerifyReport:
    """Run the six-way matrix and package the report — with dual gold scoping.

    Order is:
        base_pre → gold_pre → pred_pre → base_post → gold_post → pred_post

    Scoping arguments (all optional):
      test_scope_developer:  case.json.test_files subset (SWT-Bench canonical)
                             Defaults to split.test_files.
      test_scope_discovered: organize-agent's full pre-existing test suite
                             (repo-relative paths). If None, defaults to
                             ``all @Test methods actually observed running``
                             — effectively no-op filter.
      test_scope_local:      tests in organize-validated Maven modules.
                             If None, defaults to test_scope_discovered.

    The SWT-Bench canonical gate is always computed against
    ``test_scope_developer`` (preserves upstream comparability).
    ``discovered`` + ``local`` are additional recall proxies emitted
    on top.
    """
    test_command = test_command or runtime_spec().test_command

    order: list[Configuration] = [
        "base_pre", "gold_pre", "pred_pre",
        "base_post", "gold_post", "pred_post",
    ]
    runs: dict[Configuration, RunOutcome] = {}
    for cfg in order:
        runs[cfg] = _run_config(
            sandbox,
            cfg,
            split,
            candidate_test_patch,
            test_command,
            pipeline_patch=pipeline_patch,
            pred_code_patch=pred_code_patch,
        )

    gate = swt_gate(runs)
    swebench_live = swebench_live_resolved_report(
        runs,
        expected_pass_to_pass=expected_pass_to_pass,
        expected_fail_to_pass=expected_fail_to_pass,
    )
    assertion_f2p, compile_f2p = classify_f2p(runs, gate.pred_f2p)

    # 3-way scope defaults.
    dev_files: list[str] = list(test_scope_developer or split.test_files or [])
    disc_files: list[str] = list(test_scope_discovered or [])
    local_files: list[str] = list(test_scope_local or disc_files)

    # If no discovered scope was provided, don't over-filter: fall back to
    # "all test IDs observed" so ``gold_f2p_discovered = gold_f2p_developer``.
    if disc_files:
        gold_f2p_developer   = _test_ids_in_scope(dev_files, gate.gold_f2p) if dev_files else set(gate.gold_f2p)
        gold_p2p_developer   = _test_ids_in_scope(dev_files, gate.gold_p2p) if dev_files else set(gate.gold_p2p)
        gold_f2p_discovered  = _test_ids_in_scope(disc_files, gate.gold_f2p)
        gold_p2p_discovered  = _test_ids_in_scope(disc_files, gate.gold_p2p)
        gold_f2p_local       = _test_ids_in_scope(local_files, gate.gold_f2p)
        gold_p2p_local       = _test_ids_in_scope(local_files, gate.gold_p2p)
    else:
        # No organize output → all scopes equal the developer scope.
        gold_f2p_developer   = _test_ids_in_scope(dev_files, gate.gold_f2p) if dev_files else set(gate.gold_f2p)
        gold_p2p_developer   = _test_ids_in_scope(dev_files, gate.gold_p2p) if dev_files else set(gate.gold_p2p)
        gold_f2p_discovered  = set(gate.gold_f2p)
        gold_p2p_discovered  = set(gate.gold_p2p)
        gold_f2p_local       = set(gate.gold_f2p)
        gold_p2p_local       = set(gate.gold_p2p)

    coverage_delta_report: dict[str, object] = {}
    if collect_coverage:
        pred_pre_cov = _run_coverage_config(
            sandbox,
            "pred_pre",
            split,
            candidate_test_patch,
            test_command,
            pipeline_patch=pipeline_patch,
        )
        pred_post_cov = _run_coverage_config(
            sandbox,
            "pred_post",
            split,
            candidate_test_patch,
            test_command,
            pipeline_patch=pipeline_patch,
            pred_code_patch=pred_code_patch,
        )
        coverage_delta_report = coverage_delta(
            split.code_patch,
            pred_pre_cov["coverage"],
            pred_post_cov["coverage"],
        )
        coverage_delta_report["exit_codes"] = {
            "pred_pre": pred_pre_cov["exit_code"],
            "pred_post": pred_post_cov["exit_code"],
        }
        coverage_delta_report["xml_reports"] = {
            "pred_pre": pred_pre_cov["xml_reports"],
            "pred_post": pred_post_cov["xml_reports"],
        }

    return VerifyReport(
        case_id=case.case_id,
        image_tag=sandbox.image_tag,
        test_command=test_command,
        runs=runs,
        validity_gate=gate.valid,
        validity_notes=gate.reasons,
        fail_to_pass=sorted(gate.pred_f2p),
        pass_to_pass=sorted(gate.p2p),
        assertion_f2p=sorted(assertion_f2p),
        compile_f2p=sorted(compile_f2p),
        gold_recall_measurable=split.has_gold_tests,
        gold_recall_denominator=len(gold_f2p_developer),
        # Developer-scope aliases (canonical SWT-Bench view).
        gold_fail_to_pass=sorted(gold_f2p_developer),
        gold_pass_to_pass=sorted(gold_p2p_developer),
        pred_gold_f2p_intersection=sorted(gate.pred_f2p & gold_f2p_developer),
        # Three-way gold scoping (developer / discovered / local).
        gold_fail_to_pass_developer=sorted(gold_f2p_developer),
        gold_fail_to_pass_discovered=sorted(gold_f2p_discovered),
        gold_fail_to_pass_local=sorted(gold_f2p_local),
        gold_pass_to_pass_developer=sorted(gold_p2p_developer),
        gold_pass_to_pass_discovered=sorted(gold_p2p_discovered),
        gold_pass_to_pass_local=sorted(gold_p2p_local),
        pred_gold_developer_intersection=sorted(gate.pred_f2p & gold_f2p_developer),
        pred_gold_discovered_intersection=sorted(gate.pred_f2p & gold_f2p_discovered),
        pred_gold_local_intersection=sorted(gate.pred_f2p & gold_f2p_local),
        # SWT-Bench 3×3 categorization reports, one per candidate source
        # (verbatim from :func:`get_eval_report`).
        swt_report_base=_run_swt_report(runs, "base"),
        swt_report_gold=_run_swt_report(runs, "gold"),
        swt_report_pred=_run_swt_report(runs, "pred"),
        swebench_live_resolved=swebench_live.as_dict(),
        coverage_delta=coverage_delta_report,
        # Metadata
        generated_at=dt.datetime.now(tz=dt.UTC).isoformat(),
        llm_model=llm_model,
        llm_stub=is_stub,
        raw_gold_patch_parse=raw_gold_patch_parse,
        code_patch_source=code_patch_source,
        docker_layers=_capture_docker_layers(sandbox),
    )
