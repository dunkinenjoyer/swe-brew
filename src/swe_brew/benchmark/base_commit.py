"""Pluggable strategies for recovering the real ``base_commit`` of a case.

A case.json may ship ``L1_code_diff_raw.patches[0].commit
.merge_base = "local_snapshot"`` — an explicit placeholder — so every
snapshot invocation needs to *resolve* the real SHA before it can
faithfully materialise the pre-merge state.

Design
------
We support four resolver strategies, tried in this precedence order:

1. **Explicit override.** ``SweBrewCase.base_commit`` is already
   populated (unit tests, replays of a curated dataset).
2. **JSON sidecar.** A flat mapping ``{case_id → sha}`` on disk. The
   universal escape hatch: any strategy that yields ``(case_id, sha)``
   tuples out-of-band populates this file and everything works.
3. **GitLab MR API.** ``GET /api/v4/projects/:proj/merge_requests/:iid``
   returns ``diff_refs.base_sha`` — the exact ancestor that the MR was
   forked from. Works iff the ``PRIVATE-TOKEN`` has ``api`` scope for
   this project. Fails cleanly on 302 / 401 / 403 and falls through.
4. **Master fallback.** ``git rev-parse master`` on the local clone.
   Imprecise (master has moved past the MR); flagged so callers can
   downgrade validity_gate reasoning appropriately.

The chain is *transparent*: ``resolve()`` returns ``(sha, method)`` so
callers can audit which link fired.

Configuration
-------------
* ``SWE_BREW_BASE_COMMIT_SIDECAR`` — path to the sidecar JSON. Defaults
  to ``$SWE_BREW_WORKSPACE/base_commits.json`` (i.e. ``_work/base_commits.json``).
* ``GIT_BASE`` — API base URL (defaults to ``https://git.example.com``).
* ``GIT_TOKEN`` — personal access token with ``api`` scope.

None of these are strictly required; the fallback works without any of
them, it's just imprecise.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from swe_brew.benchmark.models import SweBrewCase

__all__ = [
    "ResolverError",
    "Resolution",
    "resolve",
    "sidecar_path",
    "load_sidecar",
    "save_sidecar",
    "fetch_via_gitlab_api",
    "guess_gitlab_project_slug",
    "fetch_mr_head_via_git_refs",
    "compute_base_by_test_first_add",
    "resolve_dual",
    "DualResolution",
]


class ResolverError(RuntimeError):
    """Terminal failure of a specific resolver strategy (surfaces in `method`)."""


@dataclass(frozen=True)
class Resolution:
    """Result of a *single-SHA* resolver run (legacy interface)."""

    sha: str
    method: str            # 'explicit' | 'sidecar' | 'gitlab-api' | 'master-fallback'
    is_precise: bool       # False iff master-fallback (or explicit-but-unverified)


@dataclass(frozen=True)
class DualResolution:
    """Result of a *dual-SHA* resolver run.

    ``base_commit`` is the pre-state (before the MR's test files existed);
    ``head_commit`` is the MR's tip (post-state). Together with
    ``git_diff.derive_patches`` these produce apply-clean SWT-Bench-style
    ``code_patch`` and ``test_patch`` scoped to the case's own file set.
    """

    base_commit: str
    head_commit: str
    method: str            # 'explicit' | 'sidecar' | 'git-refs' | 'gitlab-api'
    is_precise: bool


# ─── sidecar ─────────────────────────────────────────────────────────────────


def sidecar_path(workspace: Path | None = None) -> Path:
    """Return the sidecar path, honouring env override or ``$SWE_BREW_WORKSPACE``."""
    override = os.environ.get("SWE_BREW_BASE_COMMIT_SIDECAR")
    if override:
        return Path(override).expanduser()
    if workspace is None:
        workspace = Path(os.environ.get("SWE_BREW_WORKSPACE", "_work"))
    return workspace / "base_commits.json"


def load_sidecar(path: Path | None = None) -> dict[str, object]:
    """Load the sidecar mapping. Empty dict if absent/malformed.

    Values may be either a plain SHA (legacy) or a dict with keys
    ``base_commit`` and ``head_commit`` (dual). Callers check the shape.
    """
    p = path or sidecar_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, object] = {}
    for k, v in data.items():
        if not v:
            continue
        if isinstance(v, str):
            out[str(k)] = v
        elif isinstance(v, dict) and v.get("base_commit") and v.get("head_commit"):
            out[str(k)] = {
                "base_commit": str(v["base_commit"]),
                "head_commit": str(v["head_commit"]),
            }
    return out


def save_sidecar(mapping: dict[str, str], path: Path | None = None) -> Path:
    """Atomically write the sidecar. Returns the path written."""
    p = path or sidecar_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


# ─── GitLab MR API ───────────────────────────────────────────────────────────


def gitlab_namespace_hints() -> list[str]:
    """Return explicit namespace hints from the environment.

    The old implementation probed hardcoded namespaces. That is
    repo-specific routing and is intentionally gone: a fleet operator must
    supply ``SWE_BREW_GIT_NAMESPACE`` or comma-separated
    ``SWE_BREW_GIT_NAMESPACES``, or provide a base-commit sidecar.
    """
    namespaces: list[str] = []
    single = os.environ.get("SWE_BREW_GIT_NAMESPACE", "").strip()
    if single:
        namespaces.append(single)
    multiple = os.environ.get("SWE_BREW_GIT_NAMESPACES", "").strip()
    if multiple:
        namespaces.extend(n.strip() for n in multiple.split(",") if n.strip())
    return list(dict.fromkeys(namespaces))


def guess_gitlab_project_slug(repo_short: str) -> list[str]:
    """Return explicit candidate ``namespace/repo_short`` slugs."""
    namespaces = gitlab_namespace_hints()
    if not namespaces:
        raise ResolverError(
            "no GitLab namespace configured; set SWE_BREW_GIT_NAMESPACE "
            "or SWE_BREW_GIT_NAMESPACES, or use SWE_BREW_BASE_COMMIT_SIDECAR",
        )
    return [f"{ns}/{repo_short}" for ns in namespaces]


def fetch_via_gitlab_api(
    case: SweBrewCase,
    *,
    base_url: str | None = None,
    token: str | None = None,
    cookie: str | None = None,
    timeout: float = 15.0,
    # Injectable for tests
    http_client: Callable[..., httpx.Response] | None = None,
) -> str:
    """Query GitLab for the MR's ``diff_refs.base_sha`` — the true base_commit.

    Two auth modes are supported (either one is enough):

    * **PRIVATE-TOKEN** (``$GIT_TOKEN``) — the clean path when the token
      has ``api`` scope. On some GitLab hosts, deploy tokens grant
      only ``read_repository`` and 302 on API endpoints — check first with
      ``curl -sI $GIT_BASE/api/v4/projects``.

    * **Session cookie** (``$SWE_BREW_GITLAB_COOKIE`` or the ``cookie=``
      kwarg) — copy from an SSO'd browser (DevTools → Network →
      ``Cookie:`` request header for any /api/v4 call). Value is passed
      verbatim as the ``Cookie:`` request header.

    Raises ``ResolverError`` on any non-200 response, missing field, or
    exhausted namespace candidates.
    """
    base_url = (base_url or os.environ.get("GIT_BASE", "https://git.example.com")).rstrip("/")
    token = token or os.environ.get("GIT_TOKEN", "")
    cookie = cookie or os.environ.get("SWE_BREW_GITLAB_COOKIE", "")
    if not token and not cookie:
        raise ResolverError("neither GIT_TOKEN nor SWE_BREW_GITLAB_COOKIE set")
    if not case.mr_id:
        raise ResolverError(f"case {case.case_id!r} has no mr_id")

    if http_client is None:
        def http_client(url: str, headers: dict) -> httpx.Response:  # noqa: ANN001
            with httpx.Client(timeout=timeout, follow_redirects=False) as c:
                return c.get(url, headers=headers)

    headers = {"Accept": "application/json"}
    if token:
        headers["PRIVATE-TOKEN"] = token
    if cookie:
        headers["Cookie"] = cookie

    tried: list[str] = []
    for slug in guess_gitlab_project_slug(case.repo_short):
        proj = slug.replace("/", "%2F")
        url = f"{base_url}/api/v4/projects/{proj}/merge_requests/{case.mr_id}"
        tried.append(url)
        resp = http_client(url, headers)
        if resp.status_code != 200:
            # 302 (SSO), 401, 404 — try next namespace
            continue
        try:
            data = resp.json()
        except Exception:
            continue
        # GitLab returns diff_refs = {base_sha, head_sha, start_sha}
        # base_sha is the merge-base at the moment the MR was created.
        diff_refs = data.get("diff_refs") or {}
        base_sha = diff_refs.get("base_sha")
        if isinstance(base_sha, str) and len(base_sha) >= 8:
            return base_sha
        # Some GitLab versions only surface merge_commit_sha; fall back to
        # its first parent if we have to (documented as approximate).
        merge_commit_sha = data.get("merge_commit_sha")
        if isinstance(merge_commit_sha, str) and len(merge_commit_sha) >= 8:
            # Not the base — actual result depends on local git; we return
            # a marker the caller can consult, prefixed with 'merge:'.
            return f"merge:{merge_commit_sha}"
        raise ResolverError(
            f"GitLab MR {case.mr_id} at {slug} returned 200 but no usable "
            f"sha (keys={list(data)})",
        )
    raise ResolverError(
        f"GitLab API returned no matching MR {case.mr_id} across "
        f"{len(tried)} candidate slugs (all non-200)",
    )


# ─── the chain ───────────────────────────────────────────────────────────────


def _local_master_sha(local_clone: Path) -> str:
    """Return ``master`` HEAD SHA from a local clone; empty on failure."""
    import subprocess
    try:
        r = subprocess.run(
            ["git", "-C", str(local_clone), "rev-parse", "master"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return r.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def resolve(
    case: SweBrewCase,
    local_clone: Path,
    *,
    sidecar: Path | None = None,
    use_gitlab_api: bool = True,
) -> Resolution:
    """Run the strategy chain; return (sha, method, is_precise).

    Precedence:
      1. ``case.base_commit`` (explicit)
      2. Sidecar JSON lookup
      3. GitLab MR API (if ``GIT_TOKEN`` present and ``use_gitlab_api``)
      4. Local ``master`` HEAD (imprecise; ``is_precise=False``)

    Raises ``ResolverError`` iff *every* strategy fails (including master
    HEAD lookup, e.g. no local clone).
    """
    # 1. explicit
    if case.base_commit:
        return Resolution(sha=case.base_commit, method="explicit", is_precise=True)

    # 2. sidecar
    mapping = load_sidecar(sidecar)
    if case.case_id in mapping:
        return Resolution(sha=mapping[case.case_id], method="sidecar", is_precise=True)

    # 3. GitLab API
    if use_gitlab_api:
        try:
            sha = fetch_via_gitlab_api(case)
            if sha.startswith("merge:"):
                # We only have merge_commit_sha; ask git for its first parent
                # (== pre-merge master tip == our base).
                mc = sha[len("merge:"):]
                import subprocess
                try:
                    r = subprocess.run(
                        ["git", "-C", str(local_clone), "rev-parse", f"{mc}^1"],
                        capture_output=True, text=True, check=True, timeout=10,
                    )
                    parent = r.stdout.strip()
                    if parent:
                        return Resolution(sha=parent, method="gitlab-api:merge^1", is_precise=True)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    pass
                # Merge sha unresolvable locally — return the sha, imprecise
                return Resolution(sha=mc, method="gitlab-api:merge", is_precise=False)
            return Resolution(sha=sha, method="gitlab-api", is_precise=True)
        except ResolverError:
            pass  # fall through to master-fallback

    # 4. master fallback
    sha = _local_master_sha(local_clone)
    if not sha:
        raise ResolverError(
            f"all resolvers failed for {case.case_id!r}: "
            f"no explicit base_commit, no sidecar entry, no GitLab MR match, "
            f"and no readable master at {local_clone!s}",
        )
    return Resolution(sha=sha, method="master-fallback", is_precise=False)


# ─── git-refs strategy (Blocker B — the clean path) ─────────────────────────

def fetch_mr_head_via_git_refs(local_clone: Path, mr_id: str, *, remote: str = "origin") -> str:
    """Return the MR-head SHA via ``refs/merge-requests/<iid>/head``.

    GitLab exposes every merge request as a fetchable git ref. This uses
    only ``git ls-remote`` (and optionally ``git fetch``) — no REST API,
    no ``api`` scope, no SSO cookie. Works iff the operator's git
    credentials (ssh key / cached creds) grant read access to the remote,
    which they do by definition since the local clone already exists.

    Falls back to a local lookup first (in case the ref was previously
    fetched into a local namespace like ``refs/swe_brew/mr/<iid>``).
    """
    if not mr_id:
        raise ResolverError("empty mr_id")
    # 1. Fast local hit
    local_ref = f"refs/swe_brew/mr/{mr_id}"
    r = subprocess.run(
        ["git", "-C", str(local_clone), "rev-parse", "--verify", "--quiet", local_ref],
        capture_output=True, text=True,
    )
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()

    # 2. ls-remote for the MR head
    r = subprocess.run(
        ["git", "-C", str(local_clone), "ls-remote", remote, f"refs/merge-requests/{mr_id}/head"],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        raise ResolverError(
            f"ls-remote {remote} refs/merge-requests/{mr_id}/head "
            f"produced no ref (exit={r.returncode}, stderr={r.stderr.strip()[:200]!r})",
        )
    sha = r.stdout.split(None, 1)[0]
    if len(sha) < 40:
        raise ResolverError(f"unexpected ls-remote output: {r.stdout[:200]!r}")

    # 3. Cache locally: fetch the object graph if not already present, and
    #    always write the local ref so the next call hits the fast path.
    have = subprocess.run(
        ["git", "-C", str(local_clone), "cat-file", "-e", f"{sha}^{{commit}}"],
        capture_output=True,
    )
    if have.returncode != 0:
        fetch = subprocess.run(
            ["git", "-C", str(local_clone), "fetch", "--no-tags", "--no-recurse-submodules",
             remote, f"refs/merge-requests/{mr_id}/head:{local_ref}"],
            capture_output=True, text=True, timeout=180,
        )
        if fetch.returncode != 0:
            raise ResolverError(f"fetch of MR {mr_id} failed: {fetch.stderr.strip()[:300]}")
    else:
        # SHA already local; just plant the ref so the fast-path finds it later.
        subprocess.run(
            ["git", "-C", str(local_clone), "update-ref", local_ref, sha],
            capture_output=True,
        )
    return sha


def compute_base_by_test_first_add(
    local_clone: Path, head_commit: str, test_files: list[str],
) -> str:
    """Return the parent of the earliest add-commit of any ``test_files``.

    Strategy: for each test file the case declares, find the first commit
    on ``head_commit``'s ancestry that ADDED it (``git log --diff-filter=A``,
    oldest entry). The commit right before *any* of those adds is a valid
    base — a pre-state where at least one gold test file is genuinely
    absent, which is exactly what F2P needs to see the "ERROR at pre" signal.

    If ``test_files`` is empty or none of them can be traced, raises
    ``ResolverError``.
    """
    if not test_files:
        raise ResolverError("no test_files given; cannot compute base")

    candidates: list[str] = []
    for tf in test_files:
        r = subprocess.run(
            ["git", "-C", str(local_clone), "log", "--diff-filter=A",
             "--format=%H", head_commit, "--", tf],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            continue
        adds = [ln for ln in r.stdout.splitlines() if ln.strip()]
        if not adds:
            continue
        first_add = adds[-1]  # log lists newest-first; oldest is last
        parent = subprocess.run(
            ["git", "-C", str(local_clone), "rev-parse", "--verify", f"{first_add}^1"],
            capture_output=True, text=True,
        )
        if parent.returncode == 0 and parent.stdout.strip():
            candidates.append(parent.stdout.strip())

    if not candidates:
        raise ResolverError(
            f"no add-commit found for any of the {len(test_files)} test files "
            f"in ancestry of {head_commit[:12]}",
        )

    # Pick the OLDEST candidate — earliest commit where at least one test
    # file is guaranteed absent. Older = stronger F2P signal.
    def _committer_time(sha: str) -> int:
        r = subprocess.run(
            ["git", "-C", str(local_clone), "log", "-1", "--format=%ct", sha],
            capture_output=True, text=True,
        )
        try:
            return int(r.stdout.strip())
        except ValueError:
            return 0

    candidates.sort(key=_committer_time)
    return candidates[0]


def resolve_dual(
    case: SweBrewCase,
    local_clone: Path,
    test_files: list[str],
    *,
    sidecar: Path | None = None,
) -> DualResolution:
    """Return ``(base_commit, head_commit, method)`` — the SWT-Bench dual.

    Strategy chain:

    1. **Sidecar (dual)**: sidecar value is ``{"base_commit": …, "head_commit": …}``.
    2. **Sidecar (legacy)**: sidecar value is a plain SHA string. Treated as
       ``base_commit`` only; ``head_commit`` still resolved via git-refs.
    3. **git-refs**: ``fetch_mr_head_via_git_refs`` for head; ``compute_base_by
       _test_first_add`` for base. Zero external dependencies beyond the
       user's own git auth. This is the primary path.

    Raises ``ResolverError`` iff every strategy fails.
    """
    # Explicit short-circuit: the simple release schema carries both SHAs.
    if getattr(case, "base_commit", None) and getattr(case, "head_commit", None):
        return DualResolution(
            base_commit=case.base_commit,
            head_commit=case.head_commit,
            method="explicit",
            is_precise=True,
        )

    mapping = load_sidecar(sidecar)
    entry = mapping.get(case.case_id)

    # 1. Sidecar (dual dict)
    if isinstance(entry, dict) and entry.get("base_commit") and entry.get("head_commit"):
        return DualResolution(
            base_commit=str(entry["base_commit"]),
            head_commit=str(entry["head_commit"]),
            method="sidecar",
            is_precise=True,
        )

    # 2/3. git-refs (also handles sidecar-legacy: known base + fetched head)
    head_commit = fetch_mr_head_via_git_refs(local_clone, case.mr_id)
    if isinstance(entry, str) and entry:
        return DualResolution(
            base_commit=entry, head_commit=head_commit,
            method="sidecar-legacy+git-refs", is_precise=True,
        )
    base_commit = compute_base_by_test_first_add(local_clone, head_commit, test_files)
    return DualResolution(
        base_commit=base_commit, head_commit=head_commit,
        method="git-refs", is_precise=True,
    )
