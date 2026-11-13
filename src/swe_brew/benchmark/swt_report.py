"""SWT-Bench-canonical 3×3 test-outcome categorization.

# PROVENANCE — 1:1 verbatim port
# ================================
# ``get_eval_report`` is a byte-for-byte port of
# ``swt-bench/src/grading.py:142-181`` (Mündler et al., 2024). Same
# variable names, same 3×3 bucket table, same result-dict shape, same
# F2P/P2P constants. The only substantive change is our TestStatus enum
# has PASSED/FAILED/ERROR/SKIPPED/MISSING (theirs has just
# PASSED/FAILED); their ``in [PASSED, FAILED]`` filter naturally maps
# anything else (ERROR / SKIPPED / MISSING) into the "None" bucket, per
# their canonical semantic.
#
# The 3×3 table:
#
#              post=PASSED   post=FAILED   post=None
#   pre=PASSED    p2p           p2f           p2x
#   pre=FAILED    f2p           f2f           f2x
#   pre=None      x2p           x2f           x2x
#
# Derived buckets:
#   FAIL_TO_PASS = x2p + f2p        (line 175 in upstream)
#   PASS_TO_PASS = p2p              (line 176)
#   FAIL_TO_FAIL = f2f + x2f        (line 177)
#   PASS_TO_FAIL = p2f + p2x        (line 178)
#   UNMATCHED    = f2x + x2x        (line 179)
#
# ``FAIL_TO_PASS = x2p + f2p`` is the load-bearing definition: a test
# that was MISSING (None) at pre and PASSED at post counts as F2P.
# This is what makes SWT-Bench-canonical PERMISSIVE about
# compile-fail-at-pre — a compile-fail lands in None, and if the test
# passes at post, it becomes an x2p → f2p.
"""

from __future__ import annotations

from swe_brew.benchmark.models import TestId, TestStatus

__all__ = [
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "FAIL_TO_FAIL",
    "PASS_TO_FAIL",
    "UNMATCHED",
    "get_eval_report",
]


# Constants — 1:1 with ``swt-bench/src/constants.py``. Preserving their
# string values so downstream tooling that already knows SWT-Bench can
# consume our reports directly.
FAIL_TO_PASS = "FAIL_TO_PASS"
PASS_TO_PASS = "PASS_TO_PASS"
FAIL_TO_FAIL = "FAIL_TO_FAIL"
PASS_TO_FAIL = "PASS_TO_FAIL"
UNMATCHED    = "UNMATCHED"


def get_eval_report(
    eval_pre:  dict[TestId, TestStatus],
    eval_post: dict[TestId, TestStatus],
) -> dict[str, list[TestId]]:
    """Categorize every test observed in either eval into the 3×3 table
    (pre × post × {PASSED, FAILED, None}) and return the 5 derived buckets.

    Port of ``swt-bench/src/grading.py:get_eval_report`` (lines 142-181).
    Behaviour is preserved:

    - A test's pre-state is None if it wasn't in ``eval_pre`` OR if its
      status isn't in ``{PASSED, FAILED}``. So ERROR / SKIPPED / MISSING
      all funnel into the None bucket.
    - Same rule for post-state.
    - FAIL_TO_PASS = x2p + f2p (tests missing OR failed at pre, passed
      at post).
    - PASS_TO_FAIL = p2f + p2x (regressions).
    """
    f2p: list[TestId] = []
    f2f: list[TestId] = []
    p2p: list[TestId] = []
    p2f: list[TestId] = []
    x2p: list[TestId] = []
    x2f: list[TestId] = []
    f2x: list[TestId] = []
    p2x: list[TestId] = []
    x2x: list[TestId] = []

    PASSED = TestStatus.PASSED
    FAILED = TestStatus.FAILED

    results_dict: dict[TestStatus | None, dict[TestStatus | None, list[TestId]]] = {
        PASSED: {PASSED: p2p, FAILED: p2f, None: p2x},
        FAILED: {PASSED: f2p, FAILED: f2f, None: f2x},
        None:   {PASSED: x2p, FAILED: x2f, None: x2x},
    }

    all_tests = list(set(eval_pre.keys()).union(set(eval_post.keys())))

    for test in all_tests:
        pre_result = eval_pre.get(test)
        if pre_result not in (PASSED, FAILED):
            pre_result = None

        post_result = eval_post.get(test)
        if post_result not in (PASSED, FAILED):
            post_result = None

        results_dict[pre_result][post_result].append(test)

    return {
        FAIL_TO_PASS: sorted(set(x2p + f2p)),
        PASS_TO_PASS: sorted(set(p2p)),
        FAIL_TO_FAIL: sorted(set(f2f + x2f)),
        PASS_TO_FAIL: sorted(set(p2f + p2x)),
        UNMATCHED:    sorted(set(f2x + x2x)),
    }
