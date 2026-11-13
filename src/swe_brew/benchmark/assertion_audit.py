"""Static assertion-leakage audit for generated Java test candidates.

This is the lightweight, offline analogue of the CodeMidas adversarial
assertion review. It does not run Maven or Docker; it reads a candidate
unified diff and the gold code patch and reports whether the added test
code is grounded in concrete behavior or merely pins private symbols,
reflection shape, or incidental string literals.

The audit is deliberately conservative. It produces evidence for a human
curation pass, not a replacement for the executable six-way gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from unidiff import PatchSet, UnidiffParseError

__all__ = ["AssertionAudit", "audit_candidate"]


_REFLECTION_MARKERS = re.compile(
    r"\bClass\.forName\b|\bgetDeclaredMethod\b|\bgetDeclaredField\b"
    r"|\bgetDeclaredConstructor\b|\bEnum\.valueOf\b|\bModifier\.\b"
    r"|\bisAnnotationPresent\b|\bgetAnnotations\b|\bgetSuperclass\b"
)

_PRIVATE_REACH = re.compile(
    r"\bgetDeclaredMethod\b|\bgetDeclaredField\b|\bgetDeclaredConstructor\b"
    r"|\bsetAccessible\b|\bgetModifiers\b|\bgetSuperclass\b"
)

_DIRECT_CONSTRUCTION = re.compile(r"\bnew\s+[A-Z][A-Za-z0-9_$]*\s*\(")

_CONCRETE_ASSERTION = re.compile(
    r"\b(?:assert\w+)\s*\(|\bassertThat\s*\(|\bverify\s*\(",
)

_EXACT_LITERAL_ASSERTION = re.compile(
    r"\bassertEquals\s*\(\s*"
    r"(?:\"[^\"\n]*\"\s*,\s*[^;\n]*\.getMessage\s*\(\s*\)"
    r"|[^;\n]*\.getMessage\s*\(\s*\)\s*,\s*\"[^\"\n]*\")"
    r"|\b(?:hasMessage|containsString|startsWith|endsWith)\s*"
    r"\([^;\n]*\"[^\"\n]*\"",
    re.IGNORECASE,
)


def _is_test_path(path: str, marker: str = "/src/test/") -> bool:
    if not path or path == "/dev/null":
        return False
    return marker in f"/{path.lstrip('/')}"


def _strip_git_prefix(path: str) -> str:
    path = path.removeprefix("a/")
    return path.removeprefix("b/")


def _code_patch_is_deletion_only(code_patch: str) -> bool:
    """Return True when the gold code patch removes lines and adds none.

    This mirrors the test-generation preflight exception: for a pure
    deletion/rename contract, reflection is the only valid executable
    oracle for the removed API shape.
    """
    added = len(re.findall(r"^\+(?!\+\+).*$", code_patch or "", re.MULTILINE))
    removed = len(re.findall(r"^-(?!--).*$", code_patch or "", re.MULTILINE))
    return added == 0 and removed > 0


@dataclass(frozen=True)
class AssertionAudit:
    """Outcome of a candidate test-diff review."""

    test_files: int = 0
    added_test_methods: int = 0
    reflection_markers: int = 0
    private_symbol_reaches: int = 0
    direct_constructions: int = 0
    concrete_assertions: int = 0
    exact_literal_assertions: int = 0
    deletion_only_code_patch: bool = False
    deletes_existing_tests: bool = False
    structural_only_risk: bool = False
    overrestriction_risk: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def leakage_risk(self) -> bool:
        """True when this candidate should be flagged for manual review."""
        return bool(
            self.structural_only_risk
            or self.overrestriction_risk
            or self.deletes_existing_tests
        )


def _deletes_existing_test_source(diff: str) -> bool:
    """Conservative unified-diff check for deletion inside existing tests."""
    in_test_file = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            in_test_file = _is_test_path(line)
            continue
        if in_test_file and line.startswith("-") and not line.startswith("---"):
            return True
    return False


def audit_candidate(
    candidate_diff: str,
    *,
    code_patch: str = "",
    test_path_marker: str = "/src/test/",
) -> AssertionAudit:
    """Review added candidate test source for behavior leakage.

    Parameters
    ----------
    candidate_diff:
        The generated unified test patch, normally
        ``stages/generate/trial-*/candidate.diff``.
    code_patch:
        Optional gold code patch. Used only to permit reflection for a
        pure deletion/rename contract.
    test_path_marker:
        Runtime-specific path marker, defaulting to Maven ``src/test``.
    """
    if not candidate_diff.strip():
        return AssertionAudit(
            reasons=["empty candidate diff"],
        )

    reflection_markers = len(_REFLECTION_MARKERS.findall(candidate_diff))
    private_reaches = len(_PRIVATE_REACH.findall(candidate_diff))
    direct_constructions = len(_DIRECT_CONSTRUCTION.findall(candidate_diff))
    concrete_assertions = len(_CONCRETE_ASSERTION.findall(candidate_diff))
    exact_literals = len(_EXACT_LITERAL_ASSERTION.findall(candidate_diff))
    added_test_methods = len(re.findall(r"@Test\b", candidate_diff))
    deletes_existing_tests = _deletes_existing_test_source(candidate_diff)
    deletion_only = _code_patch_is_deletion_only(code_patch)

    test_files = 0
    try:
        patchset = PatchSet(candidate_diff)
    except UnidiffParseError:
        patchset = None

    if patchset is None:
        test_files = len(
            {
                _strip_git_prefix(right)
                for left, right in re.findall(
                    r"^diff --git a/(\S+) b/(\S+)$",
                    candidate_diff,
                    re.MULTILINE,
                )
                if _is_test_path(right, test_path_marker)
            }
        )
    else:
        test_files = sum(
            1
            for patched_file in patchset
            if _is_test_path(
                _strip_git_prefix(
                    patched_file.target_file
                    or patched_file.source_file
                    or ""
                ),
                test_path_marker,
            )
        )

    structural_only_risk = (
        not deletion_only
        and reflection_markers > 0
        and direct_constructions == 0
    )
    overrestriction_risk = (
        private_reaches > 0
        and direct_constructions == 0
    ) or exact_literals > 0

    reasons: list[str] = []
    if structural_only_risk:
        reasons.append(
            "candidate relies on reflection/signature checks without "
            "direct construction for a non-deletion patch"
        )
    if overrestriction_risk:
        reasons.append(
            "candidate reaches private symbols or asserts incidental "
            "string literals"
        )
    if deletes_existing_tests:
        reasons.append("candidate deletes lines from an existing test file")

    return AssertionAudit(
        test_files=test_files,
        added_test_methods=added_test_methods,
        reflection_markers=reflection_markers,
        private_symbol_reaches=private_reaches,
        direct_constructions=direct_constructions,
        concrete_assertions=concrete_assertions,
        exact_literal_assertions=exact_literals,
        deletion_only_code_patch=deletion_only,
        deletes_existing_tests=deletes_existing_tests,
        structural_only_risk=structural_only_risk,
        overrestriction_risk=overrestriction_risk,
        reasons=reasons,
    )
