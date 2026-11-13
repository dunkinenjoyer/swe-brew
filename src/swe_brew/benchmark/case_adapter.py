"""Read a case.json into a ``SweBrewCase``.

The corpus lives at ``$CASE_ROOT`` and looks like::

    case_id/
        case.json                     (2–10 MB per case, 24 cases total)
        ...

The *actual* data we care about lives under nested keys — not the
convenient-looking top-level ``repo`` / ``mr_id`` fields which are all
``None`` per the empirical inspection (see ``docs/DESIGN.md §4``).

Real schema:

    case_id                                      = "case_id"
    L4_prd_original.description                  = str  (21,410 chars — the problem statement)
    L4_prd_original.prd_type                     = "human_prd" or "commit_message"
    L1_code_diff_raw.patches[0].repo             = "<repo_short>"
    L1_code_diff_raw.patches[0].patch            = str  (93,633 chars — the gold diff)
    L1_code_diff_raw.patches[0].changed_files    = list[str] (32 java files)
    L1_code_diff_raw.patches[0].commit.mr_id     = "27779161"
    L1_code_diff_raw.patches[0].commit.merge_base = "local_snapshot"   # 🔴 base_commit blocker

Explicit non-goals
------------------
This adapter does not touch the network / GitLab / disk beyond reading
the JSON. base_commit recovery is ``snapshot.py``'s job.
"""

from __future__ import annotations

import json
from pathlib import Path

from swe_brew.benchmark.models import SweBrewCase

__all__ = ["load_case", "load_case_from_dir", "CaseSchemaError"]

_ALLOWED_PRD_TYPES = {"human_prd", "commit_message"}


class CaseSchemaError(ValueError):
    """Raised when case.json is missing a required field."""


def _require(d: dict, path: str, kind: type = object):
    """Traverse ``d`` by dotted path, raising a helpful error if any hop is missing."""
    cur = d
    for i, key in enumerate(path.split(".")):
        if not isinstance(cur, dict):
            raise CaseSchemaError(f"expected object at {'.'.join(path.split('.')[:i])!r}, got {type(cur).__name__}")
        if key not in cur:
            raise CaseSchemaError(f"missing required key {path!r}")
        cur = cur[key]
    if cur is None:
        raise CaseSchemaError(f"field {path!r} is null (case.json field left unfilled)")
    if kind is not object and not isinstance(cur, kind):
        raise CaseSchemaError(f"field {path!r} expected {kind.__name__}, got {type(cur).__name__}")
    return cur


def load_case(case_json_path: Path) -> SweBrewCase:
    """Load a single ``case.json`` into a ``SweBrewCase``.

    Raises ``CaseSchemaError`` for missing / typed-wrong fields.
    """
    if not case_json_path.exists():
        raise FileNotFoundError(f"case.json not found at {case_json_path}")
    data = json.loads(case_json_path.read_text(encoding="utf-8"))

    case_id: str = _require(data, "case_id", str)

    prd_text: str = _require(data, "L4_prd_original.description", str)
    prd_type: str = _require(data, "L4_prd_original.prd_type", str)
    if prd_type not in _ALLOWED_PRD_TYPES:
        # Per PLAN-ZH §3.4: we evaluate only on known original PRD labels.
        # ``human_prd`` covers the hand-authored pool; ``commit_message``
        # covers the OSS code-diff pool. Any future label must be audited
        # here rather than slipping through a substring-based guard.
        raise CaseSchemaError(
            f"{case_id}: L4_prd_original.prd_type is {prd_type!r}, not in "
            f"allowed set {sorted(_ALLOWED_PRD_TYPES)}; "
            "LLM-generated PRDs must not be used for evaluation.",
        )

    patches = _require(data, "L1_code_diff_raw.patches", list)
    if not patches:
        raise CaseSchemaError(f"{case_id}: L1_code_diff_raw.patches is empty")
    p0 = patches[0]
    for req in ("repo", "patch", "changed_files"):
        if req not in p0:
            raise CaseSchemaError(f"{case_id}: L1_code_diff_raw.patches[0].{req} missing")

    repo_short_raw: str = p0["repo"]
    # OSS code-diff cases may carry a namespace-qualified GitLab project path.
    # Local snapshot discovery expects only the repository basename.
    repo_short = repo_short_raw.rsplit("/", 1)[-1] or repo_short_raw
    gold_patch: str = p0["patch"]
    changed_files: list[str] = list(p0.get("changed_files") or [])
    setup_cmds: list[str] = list(data.get("setup_cmds") or [])
    test_cmds: list[str] = list(data.get("test_cmds") or [])
    avoid_commands: list[str] = list(data.get("avoid_commands") or [])

    commit = p0.get("commit") or {}
    mr_id: str = commit.get("mr_id") or ""
    merge_base_raw: str = commit.get("merge_base") or ""
    # Merge_base == "local_snapshot" is the known blocker; leave base_commit=None
    # and let snapshot.py recover from mr_id.
    base_commit = merge_base_raw if merge_base_raw and merge_base_raw != "local_snapshot" else None

    return SweBrewCase(
        case_id=case_id,
        repo_short=repo_short,
        mr_id=mr_id,
        prd_text=prd_text,
        gold_patch=gold_patch,
        changed_files=changed_files,
        setup_cmds=setup_cmds,
        test_cmds=test_cmds,
        avoid_commands=avoid_commands,
        base_commit=base_commit,
    )


def load_case_from_dir(case_dir: Path) -> SweBrewCase:
    """Convenience: point at the case directory instead of case.json."""
    case = load_case(case_dir / "case.json")
    prior = case_dir / "enhanced" / "result.json"
    if not prior.exists():
        return case

    try:
        prior_data = json.loads(prior.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return case
    if not prior_data.get("completed"):
        return case

    setup_cmds = list(prior_data.get("setup_commands") or case.setup_cmds)
    test_cmds = list(prior_data.get("test_commands") or case.test_cmds)
    avoid_commands = list(case.avoid_commands)
    gate = prior_data.get("swe_brew", {}).get("gate")
    if prior_data.get("organize_completed") is False or gate is False:
        avoid_commands = list(test_cmds)

    return case.model_copy(
        update={
            "setup_cmds": setup_cmds,
            "test_cmds": test_cmds,
            "avoid_commands": avoid_commands,
        },
    )
