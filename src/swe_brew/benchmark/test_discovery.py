"""Enumerate the tests a Java test-file *declares*, without compiling it.

Why
---
Maven's ``mvn test`` aborts an entire module's test phase if a single test
source fails to compile — no Surefire XMLs get written for the tests that
*would* have run. In SWT-Bench's Python world, an ``ImportError`` inside a
test file surfaces as a per-test ``ERROR`` outcome that F2P set-diff can
read directly; Java has no such analogue at the Maven level.

To close that semantic gap, we *pre-declare* what a test file promises
to run: parse its source, extract every ``@Test``-annotated method, and
build a ``TestOutcome(status=ERROR)`` for each. ``verify.py`` unions
these synthesized outcomes with Surefire's real outcomes: a synthesized
ERROR wins iff Surefire didn't emit anything for that test ID.

This is a straight port of SWT-Bench's grading semantic, not a duct-tape
fallback: any Java case where the expected tests genuinely didn't run
(compile fail, discovery fail, module skipped) now looks the same as it
would in the Python reference.

Scope
-----
Supports **JUnit 4** (``org.junit.Test``) and **JUnit 5**
(``org.junit.jupiter.api.Test``). Both use the annotation spelling
``@Test`` — we don't disambiguate at parse time. TestNG's ``@Test`` is
identical in spelling and works accidentally; if we ever hit a repo that
uses ``@ParameterizedTest`` / ``@TestFactory`` heavily we'll extend.

We deliberately do *not* pull in a JavaParser library — a Java parser
here is a heavy transitive-dep answer to a light regex-and-brace-scanning
question. The parser below is ~120 lines, is exercised by tests against
real gold test fixtures, and refuses
gracefully (``[]``) on anything it can't parse — the F2P downstream
treats "expected but not seen" symmetrically to "not expected at all".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from swe_brew.benchmark.models import TestId, TestOutcome, TestStatus
from swe_brew.benchmark.runtime import runtime_spec

__all__ = [
    "TestDeclaration",
    "parse_test_declarations",
    "enumerate_test_files_in_tree",
    "expected_outcomes_from_test_files",
]


@dataclass(frozen=True)
class TestDeclaration:
    """One test method declared in Java source."""

    fqcn: str          # e.g. com.example.module.util.CountUtilsTest
    method: str        # e.g. testAdd
    source_line: int   # 1-based line of the `@Test` annotation

    @property
    def test_id(self) -> TestId:
        return f"{self.fqcn}#{self.method}"


# ─── Java lexical helpers ────────────────────────────────────────────────────

# Strip //-line comments AND /* … */ block comments (multiline). Order matters:
# strip block first (so //-inside-block doesn't leak), then //-line.
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")

# Strip string + char literals so an inner `@Test` in a string doesn't fool us.
# This matches the *body* of the literal too, so quote-boundary offsets are lost;
# fine — we only use the stripped text for annotation-and-signature scanning,
# never for reporting positions in the ORIGINAL text (we track those from a
# *separate* pass over the raw source, below).
_STRING_LITERAL_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_CHAR_LITERAL_RE = re.compile(r"'(?:\\.|[^'\\])*'")


def _strip_java_noise(src: str) -> str:
    """Remove comments and string/char literals so annotation scans are safe.

    Keeps whitespace + newlines intact so line numbers still map roughly to
    the original — we don't rely on exact col numbers, but line numbers should
    align within a comment's height.
    """
    def _blank_keep_newlines(m: re.Match[str]) -> str:
        s = m.group(0)
        return "".join(c if c == "\n" else " " for c in s)

    src = _BLOCK_COMMENT_RE.sub(_blank_keep_newlines, src)
    src = _LINE_COMMENT_RE.sub("", src)
    src = _STRING_LITERAL_RE.sub('""', src)
    src = _CHAR_LITERAL_RE.sub("''", src)
    return src


# Package: `package a.b.c ;`  (Java's rule: exactly one at top of file).
_PACKAGE_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.MULTILINE)

# Top-level class/enum/record/interface. Java allows only one *public* type
# per file matching the filename, but a file can host multiple package-private
# types. We capture ALL top-level type names so nested `@Test`s in a
# secondary type still get attributed to the right owner.
#
# We are deliberately loose about modifiers: matches any legal java modifier
# sequence + class/enum/record/interface. Generics on record/class parameters
# are handled by non-greedy consumption up to `{`.
_TYPE_DECL_RE = re.compile(
    r"^\s*(?:@[A-Za-z_][\w.]*(?:\s*\([^)]*\))?\s*)*"     # any leading annotations
    r"(?:public\s+|protected\s+|private\s+|abstract\s+|final\s+|"
    r"static\s+|sealed\s+|non-sealed\s+)*"
    r"(?:class|interface|enum|record)\s+"
    r"([A-Za-z_]\w*)\b",
    re.MULTILINE,
)

# @Test annotation. Matches @Test, @Test(...), and @org.junit.Test / fully-qualified.
_TEST_ANNOTATION_RE = re.compile(
    r"@(?:[A-Za-z_][\w.]*\.)?"
    r"(?:Test|ParameterizedTest|TestFactory|RepeatedTest)\b"
    r"(?:\s*\([^)]*\))?",
)

# The method signature that follows a @Test annotation. We accept any modifier
# order, any return type (void or otherwise — some JUnit tests are annotated
# but non-void; we still enumerate them), any generic parameters on the return
# type. We do NOT consume the argument list — we stop at the first `(`.
_METHOD_HEAD_RE = re.compile(
    r"""
    \s*
    (?:@[A-Za-z_][\w.]*(?:\s*\([^)]*\))?\s*)*     # possible extra annotations
    (?:public\s+|private\s+|protected\s+|
       static\s+|final\s+|synchronized\s+|
       default\s+|native\s+|strictfp\s+)*
    (?:<[^>]{0,200}>\s+)?                         # optional generic type params
    (?:[\w.$<>?\[\],\s]+?)\s+                     # return type (non-greedy)
    ([A-Za-z_$]\w*)                                # method name (captured)
    \s*\(
    """,
    re.VERBOSE,
)


# ─── Top-level API ───────────────────────────────────────────────────────────


def parse_test_declarations(source: str, *, default_fqcn: str | None = None) -> list[TestDeclaration]:
    """Return every ``@Test`` declaration found in a Java source string.

    Parameters
    ----------
    source :
        Raw Java source. Comments, strings, and char literals are stripped
        internally before annotation matching so they can't trigger false
        positives.
    default_fqcn :
        Used only when the source omits ``package …;`` (rare in production
        code but common in ad-hoc tests). If a package IS declared, the
        parsed value wins.

    Returns
    -------
    list[TestDeclaration]
        Sorted by ``source_line``. Empty on any malformed input — this is
        best-effort discovery, not compilation.
    """
    if not source or "@Test" not in source and "@org.junit" not in source:
        return []

    stripped = _strip_java_noise(source)

    # Package
    pkg_match = _PACKAGE_RE.search(stripped)
    package = pkg_match.group(1) if pkg_match else ""

    # Enumerate top-level type declarations *and their opening-brace offsets*.
    # This lets us decide, for each @Test at offset X, which enclosing class it
    # belongs to (linear-scan: the last type whose { came before X and whose }
    # comes after X).
    types: list[tuple[str, int, int]] = []   # (name, open_brace_offset, close_brace_offset)
    for tm in _TYPE_DECL_RE.finditer(stripped):
        name = tm.group(1)
        # Find the opening brace after the type header.
        brace_idx = stripped.find("{", tm.end())
        if brace_idx < 0:
            continue
        close = _matching_brace(stripped, brace_idx)
        if close < 0:
            close = len(stripped)
        types.append((name, brace_idx, close))

    if not types:
        # No class header found — fall back to source-file-basename heuristic
        # via default_fqcn, so we still return SOMETHING useful for weird
        # inputs. If default_fqcn is also None, we can't attribute; bail.
        if not default_fqcn:
            return []
        types = [(default_fqcn.rsplit(".", 1)[-1], 0, len(stripped))]

    # Enumerate @Test annotations
    decls: list[TestDeclaration] = []
    for am in _TEST_ANNOTATION_RE.finditer(stripped):
        ann_end = am.end()
        # Advance past any additional annotations between @Test and the method
        # head. _METHOD_HEAD_RE tolerates them at the start too, so we just try
        # to match from ann_end onward.
        mm = _METHOD_HEAD_RE.match(stripped, ann_end)
        if not mm:
            continue
        method_name = mm.group(1)
        # Attribute to the innermost enclosing type
        owner_idx = _find_enclosing_type(types, am.start())
        if owner_idx is None:
            # @Test outside any type declaration — malformed source, skip.
            continue
        owner_name = types[owner_idx][0]
        fqcn = f"{package}.{owner_name}" if package else (default_fqcn or owner_name)
        line = stripped.count("\n", 0, am.start()) + 1
        decls.append(TestDeclaration(fqcn=fqcn, method=method_name, source_line=line))

    # Sort + dedupe by (fqcn, method) — real code sometimes has a @Test method
    # overloaded across parameterizations; we keep the first.
    seen: set[tuple[str, str]] = set()
    unique: list[TestDeclaration] = []
    for d in sorted(decls, key=lambda d: d.source_line):
        key = (d.fqcn, d.method)
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def enumerate_test_files_in_tree(
    workdir: Path,
    *,
    include_suffixes: tuple[str, ...] | None = None,
) -> list[str]:
    """Walk the sandbox worktree, return every JUnit-shaped test file
    as a repo-relative path (POSIX slashes).

    Used by verify.py's D1 fix to replace the unidiff-based
    `split_gold_patch(pred_patch)` call on malformed patches. Walking the applied
    source tree post-`git apply` is more robust: no diff parser to
    choke on malformed LLM output; we see the actual on-disk state.

    ``include_suffixes`` matches the JUnit / TestNG / IT-style
    conventions we support. Add more here if the corpus surfaces
    other patterns.
    """
    workdir = Path(workdir)
    if not workdir.is_dir():
        return []
    spec = runtime_spec()
    suffixes = include_suffixes or spec.test_file_suffixes
    path_marker = spec.test_path_marker
    rels: list[str] = []
    for p in workdir.rglob("*.java"):
        # Only walk the configured test-source subtree.
        # Anything under production/generated sources is ignored,
        # even if the filename ends in Test.java (mostly avoids fixtures).
        try:
            rel = p.relative_to(workdir).as_posix()
        except ValueError:
            continue
        if path_marker not in "/" + rel:
            continue
        if not any(rel.endswith(sfx) for sfx in suffixes):
            continue
        rels.append(rel)
    return sorted(rels)


def expected_outcomes_from_test_files(
    workdir: Path,
    test_files: list[str],
    *,
    message: str = "expected but not discovered in Surefire reports",
    status: TestStatus = TestStatus.MISSING,
) -> list[TestOutcome]:
    """Read the test files under ``workdir``, return synthesized outcomes
    for every ``@Test`` we find. Used by ``verify.py`` to backfill
    Surefire's blindspot when a test file didn't produce a report (either
    the module was outside the mvn ``-pl`` scope, or its compile failed).

    The default status is ``MISSING``. Both ``MISSING`` and ``ERROR``
    coerce to ``None`` in ``swt_report.get_eval_report`` so the F2P math
    is unchanged either way, but ``MISSING`` is honest for the audit
    trail — a test that wasn't in mvn's scope is genuinely absent from
    the run, not erroring at test-time. Callers that specifically know
    a module compile-failed can pass ``status=TestStatus.ERROR`` for
    a more precise audit signal.

    When Surefire *does* emit a real outcome for the same ``test_id``,
    the caller unions with real-outcome-wins, so these synthesized
    outcomes are always fallbacks, never overrides.

    Parameters
    ----------
    workdir : the snapshot's on-disk working copy (already contains the test
        files after ``apply_patch(test_patch)`` has been called).
    test_files : repo-relative paths (from ``SplitPatches.test_files``).
    message : populates ``TestOutcome.message`` — audit trail only.
    status : the status to synthesize. Default ``MISSING``.
    """
    outcomes: list[TestOutcome] = []
    for rel in test_files:
        p = workdir / rel
        if not p.is_file():
            continue  # deleted / renamed in the patch — skip
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Default FQCN in case package parsing fails — file basename minus .java.
        stem = Path(rel).stem
        default_fqcn = stem  # bare class name only; better than nothing
        for d in parse_test_declarations(src, default_fqcn=default_fqcn):
            outcomes.append(
                TestOutcome(
                    test_id=d.test_id,
                    status=status,
                    suite=d.fqcn,
                    message=message,
                ),
            )
    return outcomes


# ─── Internal helpers ────────────────────────────────────────────────────────


def _matching_brace(src: str, open_pos: int) -> int:
    """Return the index of the ``}`` matching the ``{`` at ``open_pos``; -1 on error."""
    if open_pos >= len(src) or src[open_pos] != "{":
        return -1
    depth = 0
    for i in range(open_pos, len(src)):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _find_enclosing_type(types: list[tuple[str, int, int]], offset: int) -> int | None:
    """Return the index into ``types`` of the *innermost* type enclosing ``offset``.

    ``types`` is (name, open_brace_offset, close_brace_offset) tuples in
    the order they were discovered. Innermost = the type whose interval
    (open, close) tightly contains ``offset``.
    """
    best: int | None = None
    best_span = float("inf")
    for i, (_name, o, c) in enumerate(types):
        if o < offset < c:
            span = c - o
            if span < best_span:
                best = i
                best_span = span
    return best
