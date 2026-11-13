"""JaCoCo-based executable-line coverage for the gold code patch.

SWT-Bench computes a *coverage delta* over the executable added/removed
lines of the golden code patch (``swt-bench/src/grading.py``). This module
is the Java-native analogue:

1. ``build_jacoco_test_command`` runs the normal Maven test command with the
   JaCoCo Maven plugin invoked directly (no ``pom.xml`` mutation).
2. ``parse_jacoco_xml`` reads each module's ``jacoco.xml`` and returns the
   executable/covered line sets.
3. ``executable_changed_lines`` derives the added/removed lines from a
   unified patch.
4. ``coverage_delta`` reports the fraction of those executable lines covered
   by a given pre/post coverage pair.

It deliberately does not use repo-specific parser logic. JaCoCo XML is the
single source of truth for "executable" and "covered".
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from unidiff import PatchSet

JACOCO_PLUGIN_VERSION = "0.8.12"


@dataclass(frozen=True)
class LineCoverage:
    """Executable/covered line numbers for one source file."""

    executable: frozenset[int]
    covered: frozenset[int]


@dataclass(frozen=True)
class ChangedLine:
    path: str
    line_number: int
    added: bool
    content: str = ""


def build_jacoco_test_command(
    test_command: str,
    *,
    data_file: str = "target/jacoco.exec",
    report_dir: str = "target/site/jacoco",
) -> str:
    """Wrap a Maven test command with JaCoCo prepare-agent + report goals.

    The goals are invoked by coordinates directly, so no repo ``pom.xml`` is
    modified. ``-Dmaven.test.failure.ignore=true`` must already be present
    in ``test_command`` or the report goal will not run after assertion
    failures.
    """
    cmd = test_command.strip()
    goals = (
        f"org.jacoco:jacoco-maven-plugin:{JACOCO_PLUGIN_VERSION}:prepare-agent "
        "test "
        f"org.jacoco:jacoco-maven-plugin:{JACOCO_PLUGIN_VERSION}:report"
    )
    props = (
        f" -Djacoco.destFile={data_file} "
        f"-Djacoco.dataFile={data_file} "
        f"-Djacoco.outputDirectory={report_dir}"
    )
    if not cmd.startswith("mvn "):
        return f"mvn {goals}{props}"
    # Avoid running `test` twice when the caller's test command already ends
    # with the test goal; the appended goals contain their own `test`.
    cmd_without_goal = re.sub(r"\btest\s*$", "", cmd).rstrip()
    return cmd_without_goal + props + f" {goals}"


def augment_test_command_for_paths(
    test_command: str,
    paths: Iterable[str],
) -> str:
    """Expand a Maven ``-pl`` list to include modules touched by ``paths``.

    JaCoCo only reports modules Maven actually builds. If the test command is
    narrowed to validated test modules but the gold patch touches production
    code in other modules, those files never appear in ``jacoco.xml``. Adding
    their Maven modules keeps the reactor scoped while instrumenting the
    changed production files.
    """
    cmd = test_command.strip()
    if not cmd.startswith("mvn "):
        return cmd

    changed_modules = {
        path.split("/", 1)[0]
        for path in paths
        if "/" in path and "/src/main/" in path
    }
    if not changed_modules:
        return cmd

    pl_match = re.search(r"(-pl\s+)(\S+)", cmd)
    if pl_match is None:
        # Reactor-wide command already builds every module.
        return cmd

    existing = {
        token.strip()
        for token in pl_match.group(2).split(",")
        if token.strip()
    }
    missing = sorted(changed_modules - existing)
    if not missing:
        return cmd

    union = sorted(existing | changed_modules)
    new_pl = f"{pl_match.group(1)}{','.join(union)}"
    expanded = cmd[:pl_match.start()] + new_pl + cmd[pl_match.end():]
    if not re.search(r"(?:^|\s)-am(?:\s|$)", expanded):
        insert_at = pl_match.start() + len(new_pl)
        expanded = expanded[:insert_at] + " -am" + expanded[insert_at:]
    return expanded


def parse_jacoco_xml(path: Path) -> dict[str, LineCoverage]:
    """Parse a JaCoCo report into ``{relative_source_path: LineCoverage}``.

    JaCoCo omits the Maven module prefix from ``sourcefile``, so the returned
    path is package-relative, e.g. ``com/foo/Bar.java``. The caller is
    responsible for suffix-matching against a unified-diff path.
    """
    if not path.exists():
        return {}

    root = ElementTree.parse(path).getroot()
    out: dict[str, LineCoverage] = {}
    for pkg in root.findall(".//package"):
        package_name = pkg.get("name", "")
        for sourcefile in pkg.findall("sourcefile"):
            filename = sourcefile.get("name", "")
            rel = f"{package_name}/{filename}".lstrip("/")
            executable: set[int] = set()
            covered: set[int] = set()
            for line in sourcefile.findall("line"):
                nr = line.get("nr")
                mi = int(line.get("mi") or 0)
                ci = int(line.get("ci") or 0)
                if nr is None or mi + ci == 0:
                    continue
                executable.add(int(nr))
                if ci > 0:
                    covered.add(int(nr))
            if executable:
                out[rel] = LineCoverage(
                    executable=frozenset(executable),
                    covered=frozenset(covered),
                )
    return out


def _patch_path(path: str) -> str:
    return path.removeprefix("a/").removeprefix("b/")


def executable_changed_lines(patch_text: str) -> list[ChangedLine]:
    """Return added/removed non-test lines from a unified patch."""
    lines: list[ChangedLine] = []
    for patched_file in PatchSet(patch_text):
        path = _patch_path(patched_file.path)
        if "/src/test/" in f"/{path.lstrip('/')}":
            continue
        for hunk in patched_file:
            for line in hunk:
                if line.is_added and line.target_line_no is not None:
                    lines.append(
                        ChangedLine(path, line.target_line_no, added=True, content=line.value)
                    )
                elif line.is_removed and line.source_line_no is not None:
                    lines.append(
                        ChangedLine(path, line.source_line_no, added=False, content=line.value)
                    )
    return lines


def _match_line(
    line: ChangedLine,
    pre: dict[str, LineCoverage],
    post: dict[str, LineCoverage],
) -> tuple[bool, bool] | None:
    """Return ``(is_executable, is_covered)`` for a changed line.

    Added lines are checked against post coverage; removed lines against pre
    coverage. Matching uses the longest suffix to tolerate JaCoCo's missing
    Maven-module prefix.
    """
    pool = post if line.added else pre
    suffix = line.path.replace("\\", "/")
    candidates = [suffix]
    for i, ch in enumerate(suffix):
        if ch == "/":
            candidates.append(suffix[i + 1:])
    best: LineCoverage | None = None
    for candidate in candidates:
        if candidate in pool:
            best = pool[candidate]
            break
    if best is None:
        return None
    if line.line_number not in best.executable:
        return None
    return True, line.line_number in best.covered


def coverage_delta(
    patch_text: str,
    pre: dict[str, LineCoverage],
    post: dict[str, LineCoverage],
) -> dict[str, int | float | None]:
    """Compute SWT-Bench-style executable-line coverage delta.

    Returns ``{total, covered, pct, skipped, fallback}``. ``fallback`` counts
    changed lines absent from JaCoCo but treated as executable from source
    syntax, so an uninstrumented module is counted as executable-but-uncovered
    rather than silently dropped.
    """
    changed = executable_changed_lines(patch_text)
    total = 0
    covered = 0
    skipped = 0
    fallback = 0
    for line in changed:
        matched = _match_line(line, pre, post)
        if matched is None:
            if _fallback_executable(line.content):
                fallback += 1
                total += 1
            else:
                skipped += 1
            continue
        is_executable, is_covered = matched
        if not is_executable:
            skipped += 1
            continue
        total += 1
        if is_covered:
            covered += 1
    return {
        "total": total,
        "covered": covered,
        "pct": round(100.0 * covered / total, 2) if total else None,
        "skipped": skipped,
        "fallback": fallback,
    }


_COMMENT_OR_DECL_ONLY = re.compile(
    r"^\s*(?://|/\*|\*|/\*\*|@|\}|\{|package\s+|import\s+)"
)


def _fallback_executable(content: str) -> bool:
    """Conservative source fallback for lines JaCoCo did not instrument."""
    stripped = content.strip()
    if not stripped:
        return False
    if _COMMENT_OR_DECL_ONLY.match(stripped):
        return False
    if stripped in {";", "}", "{", "});", ")"}:
        return False
    return True


def merge_jacoco_xmls(paths: Iterable[Path]) -> dict[str, LineCoverage]:
    """Merge multiple module-level JaCoCo reports into one map."""
    merged: dict[str, set[int]] = {}
    covered_merged: dict[str, set[int]] = {}
    for path in paths:
        for rel, lc in parse_jacoco_xml(path).items():
            merged.setdefault(rel, set()).update(lc.executable)
            covered_merged.setdefault(rel, set()).update(lc.covered)
    return {
        rel: LineCoverage(
            executable=frozenset(merged[rel]),
            covered=frozenset(covered_merged.get(rel, set())),
        )
        for rel in merged
    }


_JACOCO_XML_GLOB = "**/target/site/jacoco/jacoco.xml"


def find_jacoco_xmls(work_dir: Path) -> list[Path]:
    """Find every module-level JaCoCo report under a Maven working tree."""
    return sorted(work_dir.glob(_JACOCO_XML_GLOB))


def _looks_like_java_path(path: str) -> bool:
    return path.endswith(".java")


def validate_patch_paths(patch_text: str) -> list[str]:
    """Return malformed patch paths for defensive logging."""
    bad: list[str] = []
    for line in re.findall(r"^(?:\+\+\+|--\-) ([ab]/.*)$", patch_text, re.MULTILINE):
        if not _looks_like_java_path(line):
            bad.append(line)
    return bad
