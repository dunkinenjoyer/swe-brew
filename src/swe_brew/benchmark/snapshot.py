"""Take a fresh, isolated working-copy of a repo at a specific commit.

Two responsibilities:

1. **Locate** a local clone of the target repo (avoiding a network hit).
2. **Materialise** a working-copy at a specific ``base_commit`` inside our
   staging area, without mutating the original clone.

Design notes
------------
- **Local-clone-first.** ``$CASE_ROOT`` and the target repos live
  under a configured local clone root. We refuse to network-clone from an
  anonymous session; if a clone is missing, the caller prints the exact
  ``git clone`` invocation needed and exits.

- **Isolation via `git worktree`.** Working copies live in
  ``<workspace>/snapshots/<case_id>/``. We use ``git worktree add`` so
  the checkout is isolated but shares object storage with the source clone
  (fast and cheap). ``git worktree remove`` cleans up.

- **base_commit recovery.** The case.json ships ``mr_id`` but
  ``merge_base == "local_snapshot"``. For the POC we accept two
  fallbacks (with an audit note):

  1. If the local clone has an obvious *"Merge ... !<mr_id>"* commit
     in its master history, use that merge commit's *first parent*
     (which is the pre-merge tip of master, i.e. the base_commit).
  2. Otherwise fall back to ``master`` HEAD with a warning; document
     the imprecision in the ``VerifyReport``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from swe_brew.benchmark import base_commit as _bc
from swe_brew.benchmark.models import SweBrewCase

__all__ = [
    "SnapshotError",
    "Snapshot",
    "local_clone_path",
    "resolve_base_commit",
    "create_snapshot",
    "cleanup_snapshot",
]


class SnapshotError(RuntimeError):
    """Raised when a snapshot cannot be created (missing clone, git failure, ...)."""


@dataclass(frozen=True)
class Snapshot:
    """A materialised working-copy at a known commit."""

    case_id: str
    repo_short: str
    base_commit: str
    work_dir: Path
    # Where the *source* clone lives (used by cleanup + audit).
    source_clone: Path
    # How base_commit was determined ("mr-search", "master-fallback", "explicit").
    resolution_method: str


# ─── locate a local clone ────────────────────────────────────────────────────

def _local_clone_roots() -> tuple[Path, ...]:
    """Return configured local clone roots, first match wins."""
    raw = os.environ.get("SWE_BREW_LOCAL_CLONE_ROOTS", "").strip()
    if raw:
        return tuple(Path(p.strip()).expanduser() for p in raw.split(",") if p.strip())
    return (Path.home() / "swe-brew-clones",)


def local_clone_path(repo_short: str) -> Path:
    """Return the local clone path for ``repo_short`` or raise."""
    for root in _local_clone_roots():
        candidate = (root / repo_short).resolve()
        if (candidate / ".git").is_dir():
            return candidate
    roots = [str((root / repo_short).resolve()) for root in _local_clone_roots()]
    raise SnapshotError(
        f"local clone for {repo_short!r} not found under {roots}. "
        f"Clone it first:\n"
        f"  git clone https://github.com/<org>/{repo_short}.git "
        f"{roots[0]}",
    )


# ─── base_commit recovery ────────────────────────────────────────────────────


def _git(clone: Path, *args: str, check: bool = True) -> str:
    """Run ``git`` inside ``clone`` and return stdout (stripped)."""
    result = subprocess.run(
        ["git", "-C", str(clone), *args],
        capture_output=True, text=True, check=check,
    )
    return result.stdout.strip()


def resolve_base_commit(case: SweBrewCase) -> tuple[str, str]:
    """Return ``(base_commit_sha, resolution_method)`` — thin wrapper.

    Delegates to ``base_commit.resolve``. Kept as a stable interface so the
    rest of the codebase doesn't have to import ``Resolution``.
    """
    clone = local_clone_path(case.repo_short)
    r = _bc.resolve(case, clone)
    return r.sha, r.method


# ─── materialise a snapshot ──────────────────────────────────────────────────


def create_snapshot(
    case: SweBrewCase,
    workspace: Path,
    *,
    base_commit: str | None = None,
    suffix: str = "",
) -> Snapshot:
    """Create a working-copy of ``case.repo_short`` at ``base_commit``.

    Uses ``git worktree add`` for cheap isolation. If a stale worktree
    already exists at the target path it is removed first.

    Parameters
    ----------
    case: the SweBrewCase (provides ``repo_short`` and audit info)
    workspace: root directory for all SWE-BREW working state
    base_commit: override recovered SHA (mostly for tests)

    Returns
    -------
    A ``Snapshot`` dataclass — pass this to ``container.build_image``.
    """
    clone = local_clone_path(case.repo_short)
    if base_commit is None:
        commit, method = resolve_base_commit(case)
    else:
        commit, method = base_commit, "override"

    work_root = workspace / "snapshots"
    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = work_root / (f"{case.case_id}-{suffix}" if suffix else case.case_id)

    # Prune ALL stale worktree registrations for this clone (crashed runs
    # leave orphaned registrations that make `worktree add` exit 128), then
    # drop any leftover dir at the target path.
    _git(clone, "worktree", "prune", check=False)
    if work_dir.exists():
        cleanup_snapshot(clone, work_dir)

    try:
        _git(clone, "worktree", "add", "--detach", str(work_dir), commit)
    except Exception:
        # A concurrent/earlier run may hold a registration for this exact
        # path or commit; force-prune and retry once before giving up.
        _git(clone, "worktree", "prune", check=False)
        if work_dir.exists():
            cleanup_snapshot(clone, work_dir)
        _git(clone, "worktree", "add", "--detach", str(work_dir), commit)

    return Snapshot(
        case_id=case.case_id,
        repo_short=case.repo_short,
        base_commit=commit,
        work_dir=work_dir,
        source_clone=clone,
        resolution_method=method,
    )


def cleanup_snapshot(clone: Path, work_dir: Path) -> None:
    """Remove a worktree cleanly (best-effort — leaves nothing behind)."""
    try:
        _git(clone, "worktree", "remove", "--force", str(work_dir), check=False)
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
            if work_dir.exists():
                subprocess.run(
                    ["sudo", "-n", "rm", "-rf", str(work_dir)],
                    check=False,
                )
        _git(clone, "worktree", "prune", check=False)
