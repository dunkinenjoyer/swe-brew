"""Split a unified gold diff into ``code_patch`` and ``test_patch``.

Rationale
---------
SWE-bench-Live's evaluator applies ``test_patch`` first and then the
solution ``patch`` (evaluation/evaluation.py:115-126). The split is the
*caller's* job — we do it here so that:

* ``code_patch`` is the *answer we withhold* from the agent under
  evaluation (it's what the human developer wrote to *solve* the requirement),
* ``test_patch`` is the *fail-to-pass* evidence set that lands *before*
  the code patch is applied,
* both round-trip cleanly with ``git apply``.

Test-file convention (Java, Maven multi-module)
-------------------------------------------------------
A hunk is a *test hunk* iff its target path lives under a Maven test
source root — i.e. the path contains an ``src/test/`` segment.

Filename patterns like ``*Test.java`` / ``TestFoo.java`` are *not*
sufficient on their own: real-world Java code has production classes
called ``TestService``, ``TestDataDao``, etc. — services related to
"test features" (A/B testing, feature-flag experiments), not JUnit
classes. Example: a ``src/main/java/.../service/test/TestService.java``
production class is not a JUnit test despite its name.

Rule: **``src/test/`` is authoritative**, name conventions only matter
inside that segment.
"""

from __future__ import annotations

import re

from unidiff import PatchSet, UnidiffParseError

from swe_brew.benchmark.models import SplitPatches
from swe_brew.benchmark.runtime import runtime_spec

__all__ = ["split_gold_patch", "is_test_path", "count_hunks"]

# Authoritative: anything under a Maven test source root.
def is_test_path(path: str) -> bool:
    """True iff ``path`` lives under a Maven test source root.

    Only ``str`` in, ``bool`` out — deliberately no filesystem access.
    """
    if not path or path in ("/dev/null",):
        return False
    marker = runtime_spec().test_path_marker
    return marker in f"/{path.lstrip('/')}"


def _hunk_target_path(patched_file) -> str:  # unidiff.PatchedFile
    """Return the "target" path for a patched file, robust to added/deleted files.

    For an added file the source is ``/dev/null`` and the target is the real
    path; the inverse for a deletion. We classify by whichever side has the
    real path — a test is a test regardless of the direction.
    """
    target = getattr(patched_file, "target_file", "") or ""
    source = getattr(patched_file, "source_file", "") or ""
    # unidiff prefixes with 'a/' and 'b/' — strip.
    target = target.removeprefix("b/") if target.startswith("b/") else target
    source = source.removeprefix("a/") if source.startswith("a/") else source
    real = target if target and target != "/dev/null" else source
    return real.lstrip("/")


def split_gold_patch(gold_patch: str) -> SplitPatches:
    """Split a unified gold diff into (code_patch, test_patch).

    Preserves each PatchedFile's original textual representation so
    ``git apply`` sees exactly what the developer wrote — no reserialization
    quirks. Empty sides are returned as ``""``, not ``None``.
    """
    if not gold_patch.strip():
        return SplitPatches(code_patch="", test_patch="", classification="empty")

    code_files: list[str] = []
    test_files: list[str] = []
    code_chunks: list[str] = []
    test_chunks: list[str] = []

    try:
        patchset = PatchSet(gold_patch)
    except UnidiffParseError:
        # Some OSS case.json patches are data-factory-normalized diffs
        # without the trailing newline that unidiff requires. This split
        # only needs file classification; derive_patches later uses
        # git-native diffs. Parse `diff --git` headers directly and leave
        # the raw text chunks empty so malformed input cannot poison it.
        for match in re.finditer(
            r"^diff --git a/(\S+) b/(\S+)$",
            gold_patch,
            re.MULTILINE,
        ):
            left, right = match.groups()
            path = right if right != "/dev/null" else left
            if not path or path == "/dev/null":
                continue
            if is_test_path(path):
                test_files.append(path)
            else:
                code_files.append(path)
        return SplitPatches(
            code_patch="",
            test_patch="",
            code_files=code_files,
            test_files=test_files,
            classification="malformed-header-classified",
        )

    for pf in patchset:
        path = _hunk_target_path(pf)
        text = str(pf)  # unidiff preserves the original hunk text on __str__
        if is_test_path(path):
            test_files.append(path)
            test_chunks.append(text)
        else:
            code_files.append(path)
            code_chunks.append(text)

    return SplitPatches(
        code_patch="".join(code_chunks),
        test_patch="".join(test_chunks),
        code_files=code_files,
        test_files=test_files,
        classification="unidiff-parsed",
    )


def count_hunks(patch_text: str) -> int:
    """Sum of hunks across all files. Convenience for verify-side stats."""
    if not patch_text.strip():
        return 0
    return sum(len(pf) for pf in PatchSet(patch_text))
