"""SWE-BREW benchmark enhancer.

The first wheel: takes a benchmark task (problem statement + gold patch + repo)
and produces validated F2P / P2P test sets that let downstream evaluations
report ground-truth ``resolution``.

Pipeline (see ``docs/DESIGN.md §4``):

    case.json
       │
       ├─► case_adapter    → SweBrewCase   (extract PRD / gold diff / repo / mr_id)
       │
       ├─► patch_split     → (code_patch, test_patch)
       │
       ├─► snapshot        → checkout base_commit into a working dir
       │
       ├─► container       → Docker image  (temurin JDK + maven cache)
       │
       ├─► generate        → candidate test_patch (LLM: PRD + code_patch)
       │
       └─► verify          → 6-way execution matrix + candidate acceptance gate
                                    │
                                    ▼
                              VerifyReport
                              test_results.json
"""

from swe_brew.benchmark.models import (
    F2P_SET,
    P2P_SET,
    SweBrewCase,
    TestOutcome,
    TestStatus,
    VerifyReport,
)

__all__ = [
    "SweBrewCase",
    "VerifyReport",
    "TestOutcome",
    "TestStatus",
    "F2P_SET",
    "P2P_SET",
]
