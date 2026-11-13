"""RepoLaunch-parity ``result.json`` manifest builder.

# PROVENANCE
# ==========
# The canonical top-level shape is 1:1 with
# ``SWE-bench-Live/launch/data/examples/playground/*/result.json``.
# I inspected ``rayon-rs__rayon-5142c8d/result.json`` (a completed
# RepoLaunch example) — it has these 20+ top-level fields:
#
#     completed, organize_completed, cost, docker_image, docker_image_layers,
#     docs, duration, exception, instance_id, log_parser, pertest_command,
#     print_commands, rebuild_commands, repo_structure, setup_commands,
#     test_commands, test_status, unittest_generator
#
# We reproduce these where our data supports it, and nest our F2P
# additions under ``swe_brew.*`` so any existing SWE-bench-Live tooling
# can consume our reports without modification. (Option A from the
# design discussion — user green-lit.)
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from swe_brew.benchmark.runtime import runtime_spec

__all__ = ["build_result_json"]


def build_result_json(
    *,
    case_id: str,
    completed: bool = True,
    organize_completed: bool = False,
    exception: str | None = None,
    duration_seconds: float = 0.0,
    docker_image: str = "",
    docker_image_layers: dict[str, Any] | None = None,
    setup_output: Any | None = None,
    organize_output: Any | None = None,
    verify_report: Any | None = None,
    repo_structure: str = "",
    docs: str = "",
    cost: dict[str, dict[str, int]] | None = None,
    started_at: dt.datetime | None = None,
) -> dict[str, Any]:
    """Assemble the canonical result.json manifest.

    Fields mirror RepoLaunch's ``result.json`` where our data maps
    directly; our F2P extensions live under ``result["swe_brew"]``.
    Fields with no direct analog (e.g. ``log_parser``, which their
    parselog agent writes — we don't need one since Surefire XML is
    universal) get sensible defaults or empty values.
    """
    setup_output_json = _extract_json(setup_output)
    organize_output_json = _extract_json(organize_output)
    verify_report_json = _extract_json(verify_report)

    # RepoLaunch's ``cost`` is a dict per phase; we produce the same
    # shape from the per-role cost dicts of our three agents.
    if cost is None:
        cost = {}
        if setup_output_json:
            cost["setup"] = setup_output_json.get("cost", {})
        if organize_output_json:
            cost["organize"] = organize_output_json.get("cost", {})

    manifest: dict[str, Any] = {
        # RepoLaunch canonical fields (Option A — flat top-level).
        "instance_id":         case_id,
        "case_id":             case_id,               # alias for our tooling
        "completed":           completed,
        "organize_completed":  organize_completed,
        "exception":           exception,
        "duration":            int(duration_seconds / 60) if duration_seconds else 0,   # theirs is minutes
        "duration_seconds":    round(duration_seconds, 1),
        "docker_image":        docker_image,
        "docker_image_layers": docker_image_layers or {},
        "cost":                cost,
        "repo_structure":      repo_structure,
        "docs":                docs,
        # From setup output
        "setup_commands":      _get(setup_output_json, "setup_commands", default=[]),
        # From organize output — RepoLaunch's field names
        "rebuild_commands":    [],  # we don't do a separate rebuild phase; empty for parity
        "test_commands":       [_get(organize_output_json, "test_command", default="")]
                                if organize_output_json else [],
        "print_commands":      [],  # Surefire is our print — no separate command
        "log_parser":          "",  # we use surefire.parse_surefire_reports directly
        "pertest_command":     _pertest_dict(organize_output_json, verify_report_json),
        "test_status":         _test_status_dict(verify_report_json),
        "unittest_generator":  "",  # not applicable — LLM writes tests, not test-runners
        "created_at":          started_at.isoformat() if started_at else "",
        # Our F2P + benchmark-specific extensions (Option A — nested).
        "swe_brew": _swe_brew_extension(verify_report_json, organize_output_json),
    }
    return manifest


def _extract_json(obj: Any) -> dict[str, Any] | None:
    """Convert obj to a dict via to_json / model_dump / dict / passthrough."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_json"):
        return obj.to_json()
    return None


def _get(d: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    if d is None:
        return default
    return d.get(key, default)


def _pertest_dict(organize: dict[str, Any] | None, verify: dict[str, Any] | None) -> dict[str, str]:
    """Build ``{test_id → shell command}`` mirroring RepoLaunch's
    ``pertest_command`` field.

    Uses the discovered ``pertest_template`` if organize gave us one,
    filled in with each test_id we ever observed running.
    """
    if not organize:
        return {}
    template = organize.get("pertest_template", "")
    if not template:
        return {}
    # Enumerate test IDs we've seen — union across configs.
    seen: set[str] = set()
    if verify and "runs" in verify:
        for run in verify["runs"].values():
            for t in run.get("tests", []):
                seen.add(t.get("test_id", ""))
    out: dict[str, str] = {}
    for tid in sorted(seen):
        if "#" not in tid:
            continue
        cls, method = tid.rsplit("#", 1)
        module = _module_for_test_class(organize, cls)
        command = (
            template.replace("{module}", module)
            .replace("{cls}", cls)
            .replace("{method}", method)
        )
        # Never emit a command with unresolved template slots. It is not
        # executable, so downstream tooling must not mistake it for one.
        if "{" not in command:
            out[tid] = command
    return out


def _module_for_test_class(organize: dict[str, Any], test_class: str) -> str:
    """Map an FQCN to the Maven module from organize's test inventory."""
    for path in organize.get("test_scope_full", []) or []:
        fqcn = _fqcn_from_test_path(path)
        if fqcn == test_class:
            return path.split("/", 1)[0]
    return ""


def _fqcn_from_test_path(path: str) -> str:
    """Convert a repo-relative Java test path to its declared FQCN."""
    marker = runtime_spec().test_path_marker
    idx = path.find(marker)
    if idx < 0:
        bare_marker = marker.lstrip("/")
        if path.startswith(bare_marker):
            pkg_and_file = path[len(bare_marker):]
        else:
            return ""
    else:
        pkg_and_file = path[idx + len(marker):]
    if not pkg_and_file.endswith(".java"):
        return ""
    return pkg_and_file[:-len(".java")].replace("/", ".")


def _test_status_dict(verify: dict[str, Any] | None) -> dict[str, str]:
    """Return ``{test_id → 'pass' | 'fail' | 'skip'}`` from the most-
    representative run (prefer pred_post if the run succeeded, else
    gold_post, else base_post, else base_pre — matches RepoLaunch's
    "final test_status" semantics)."""
    if not verify or "runs" not in verify:
        return {}
    priority = ["pred_post", "gold_post", "base_post", "pred_pre", "gold_pre", "base_pre"]
    for cfg in priority:
        run = verify["runs"].get(cfg)
        if not run:
            continue
        tests = run.get("tests", [])
        if not tests:
            continue
        out: dict[str, str] = {}
        for t in tests:
            status = t.get("status", "").lower()
            # RepoLaunch uses pass/fail/skip; map ours.
            if status == "passed":
                out[t["test_id"]] = "pass"
            elif status in ("failed", "error"):
                out[t["test_id"]] = "fail"
            elif status == "skipped":
                out[t["test_id"]] = "skip"
            else:
                out[t["test_id"]] = status
        return out
    return {}


def _swe_brew_extension(
    verify: dict[str, Any] | None,
    organize: dict[str, Any] | None,
) -> dict[str, Any]:
    """Our F2P-specific fields, nested so RepoLaunch tooling can ignore
    the ``swe_brew`` key entirely if it wants to."""
    if not verify:
        return {"gate": False, "note": "no verify_report available"}
    v = verify
    # Deliverable = deduped superset of F2P + P2P that safely built and
    # ran in the executable env. Execution is the ground-truth filter;
    # scope splits and intersections are preserved as diagnostics.
    pred_f2p = v.get("fail_to_pass") or []
    pred_p2p = v.get("pass_to_pass") or []
    # Recall vs SWT-Bench-canonical gold denominator (developer scope).
    gold_f2p = v.get("gold_fail_to_pass_developer") or v.get("gold_fail_to_pass") or []
    intersect = (
        v.get("pred_gold_developer_intersection")
        or v.get("pred_gold_f2p_intersection")
        or []
    )
    recall_pct = _pct(len(intersect), len(gold_f2p))

    return {
        "gate":                        v.get("validity_gate", False),
        "gate_reasons":                v.get("validity_notes", []),
        # DELIVERABLE: deduped superset that safely built + ran.
        "fail_to_pass":                pred_f2p,
        "pass_to_pass":                pred_p2p,
        "assertion_f2p":               v.get("assertion_f2p", []),
        "compile_f2p":                 v.get("compile_f2p", []),
        # Recall vs SWT-Bench-canonical gold (case.json.test_files).
        "gold_fail_to_pass":           gold_f2p,
        "gold_recall_measurable":      bool(v.get("gold_recall_measurable", False)),
        "gold_recall_denominator":     int(v.get("gold_recall_denominator", len(gold_f2p))),
        "recall_vs_gold_pct":          recall_pct,
        # SWT-Bench-canonical 3×3 categorization, one report per candidate
        # source. Verbatim ``get_eval_report`` output (see swt_report.py).
        # Downstream analytics slice this for finer views.
        "swt_reports": {
            "base": v.get("swt_report_base") or {},
            "gold": v.get("swt_report_gold") or {},
            "pred": v.get("swt_report_pred") or {},
        },
        # Additional scope splits + audit counts (our contribution —
        # organize agent's broader test-suite discovery).
        "detail": {
            "raw_gold_patch_parse":          v.get("raw_gold_patch_parse", "unidiff-parsed"),
            "code_patch_source":              v.get("code_patch_source", "git-native"),
            "gold_fail_to_pass_developer":       gold_f2p,
            "gold_fail_to_pass_discovered":      v.get("gold_fail_to_pass_discovered", []),
            "gold_fail_to_pass_local":           v.get("gold_fail_to_pass_local", []),
            "pred_gold_developer_intersection":  intersect,
            "pred_gold_discovered_intersection": v.get("pred_gold_discovered_intersection", []),
            "pred_gold_local_intersection":      v.get("pred_gold_local_intersection", []),
            "n_tests_scope_full":                len(_get(organize, "test_scope_full", []) or []),
            "n_tests_scope_local":               len(_get(organize, "test_scope_local", []) or []),
            "n_tests_scope_developer":           len(_get(organize, "test_scope_developer", []) or []),
            "validated_modules":                 _get(organize, "validated_modules", []) or [],
        },
    }


def _pct(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(100.0 * numerator / denominator, 2)
