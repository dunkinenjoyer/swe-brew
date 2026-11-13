"""Derive git-native, apply-clean patches from real (base, head) commit pairs.

Motivation
----------
``case.json`` ships a ``L1_code_diff_raw.patches[0].patch`` field that is a
*data-factory-normalized* diff — semantically correct but **not
byte-reversible against any real commit** in the git history. Attempting
``git apply --check`` on it exits non-zero at every ancestor we probed
(15 commits deep), and ``--3way`` fails because the pre-image blobs it
references aren't in the local object database.

SWE-Bench / SWT-Bench avoid this entire class of problem by taking the
raw patch **from git itself** (`git diff <base> <head>`) — those diffs
are apply-clean by construction because git generates them from the
actual commit states.

We do the same here: given ``(base_commit, head_commit)`` from git, plus
the *scope* of paths that count for this case (from ``case.json``), we
compute two patches by intersecting:

* ``code_patch = git diff base head -- <case.changed_files ∖ test_paths>``
* ``test_patch = git diff base head -- <case.test_paths>``

This uses git for the STATE, ``case.json`` for the SCOPE, and combines
them into a diff that is both semantically-focused AND apply-clean.

Why the intersection matters
----------------------------
A raw ``git diff base head`` between a branch point ~5 weeks before the
MR and the MR head can include 1000+ unrelated files (other MRs that
landed on the developer's branch via sync-merge cascades). Restricting
the pathspec to files listed in ``case.json`` isolates *this MR's*
semantic scope while still producing a git-canonical (apply-clean) diff.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from swe_brew.benchmark.models import SplitPatches
from swe_brew.benchmark.patch_split import is_test_path

__all__ = ["derive_patches", "GitDiffError"]


class GitDiffError(RuntimeError):
    pass


def _pathspec_for(paths: list[str]) -> list[str]:
    """Turn a list of file paths into a git pathspec argv suffix.

    Git accepts each path literally; no globbing needed since we already
    know the exact set. Prepending ``--`` is caller's responsibility.
    """
    return [p for p in paths if p and p not in ("/dev/null",)]


def derive_patches(
    local_clone: Path,
    base_commit: str,
    head_commit: str,
    test_scope: list[str],
    *,
    code_scope: list[str] | None = None,
    context_lines: int = 3,
) -> SplitPatches:
    """Return SplitPatches computed from ``git diff base..head`` with a
    *two-tier* scope model — compilation vs grading.

    Motivation
    ----------
    In a typical change request, ``case.json.changed_files`` is a *grading* scope
    (which files count for F2P/P2P), not a *compilation* scope (all files
    a Maven module needs to compile). We hit this empirically: the POC's
    92-file scope missed a util referenced by a changed class that lives
    in a sibling package. Scoped
    code_patch → base_post compile failure → no test outcomes.

    SWT-Bench upstream sidesteps this by taking the whole ``git diff``
    (compilation-complete by construction). We do the same for
    ``code_patch`` (default: no code_scope filter → full raw diff over
    non-test paths), while keeping ``test_patch`` scoped to the case's
    declared test files (so F2P/P2P grading remains focused on the MR's
    intent).

    Parameters
    ----------
    local_clone : path to the local git clone
    base_commit : pre-state SHA
    head_commit : post-state SHA
    test_scope :  paths that count as "gold tests" for grading (usually
        ``case.json.test_files``). Only these are included in ``test_patch``.
    code_scope :  paths that count for the code patch. ``None`` (default) →
        use ALL non-test files that differ between base and head. Non-None
        restricts the diff — but you probably want None for compile
        completeness.
    context_lines : ``--unified=N`` for git-diff.

    Returns
    -------
    SplitPatches
        ``code_patch`` = git-diff over the code_scope (or full raw non-test
                        diff if scope is None) — guaranteed apply-clean
                        against base_commit AND compilation-complete at
                        head_commit.
        ``test_patch`` = git-diff over the test_scope only.
        ``code_files`` / ``test_files`` = the paths that ended up in each.
    """
    # code_patch scope
    if code_scope is None:
        # Full raw diff, minus tests (isolated below).
        argv = [
            "git", "-C", str(local_clone), "-c", "core.quotepath=false",
            "diff", "--name-only", f"{base_commit}..{head_commit}",
        ]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=60)
        except subprocess.CalledProcessError as e:
            raise GitDiffError(
                f"git diff --name-only {base_commit[:8]}..{head_commit[:8]} failed "
                f"(exit={e.returncode}): {e.stderr[:300]}",
            ) from e
        except subprocess.TimeoutExpired as e:
            raise GitDiffError("git diff --name-only timed out") from e
        all_changed = [p for p in r.stdout.splitlines() if p]
        code_files = sorted({p for p in all_changed if not is_test_path(p)})
    else:
        code_files = sorted({p for p in code_scope if not is_test_path(p)})

    # test_patch scope
    test_files = sorted({p for p in (test_scope or []) if is_test_path(p)})

    code_patch = _git_diff(local_clone, base_commit, head_commit, code_files, context_lines) if code_files else ""
    test_patch = _git_diff(local_clone, base_commit, head_commit, test_files, context_lines) if test_files else ""

    return SplitPatches(
        code_patch=code_patch,
        test_patch=test_patch,
        code_files=code_files,
        test_files=test_files,
        classification="git-native",
    )


def _git_diff(
    clone: Path,
    base: str,
    head: str,
    paths: list[str],
    context_lines: int,
) -> str:
    """Run ``git diff <base> <head> -- <paths>`` and return stdout.

    Uses ``--no-color`` and ``--binary`` off (we're passing to
    ``git apply`` inside a container — text-only is fine for Java).
    """
    if not paths:
        return ""
    argv = [
        "git", "-C", str(clone),
        "-c", "core.quotepath=false",       # unicode paths verbatim
        "diff",
        f"--unified={context_lines}",
        "--no-color",
        "--no-renames",                     # keep {add, delete} pairs stable across apply
        f"{base}..{head}",
        "--",
        *paths,
    ]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=120)
    except subprocess.CalledProcessError as e:
        raise GitDiffError(
            f"git diff {base[:8]}..{head[:8]} on {len(paths)} paths failed "
            f"(exit={e.returncode}): {e.stderr[:300]}",
        ) from e
    except subprocess.TimeoutExpired as e:
        raise GitDiffError(f"git diff timed out after 120s on {len(paths)} paths") from e
    return r.stdout
