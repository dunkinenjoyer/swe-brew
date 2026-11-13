"""OrganizeAgent — discovers the correct ``mvn test`` invocation + full test inventory.

# PROVENANCE
# ==========
# Grounded on ``SWE-bench-Live/launch/launch/agent/organize/testall.py``
# (~500 LOC — their multi-language test-command-discovery agent) and
# ``testone.py`` (~283 LOC — per-test targeting). Our version is
# Java-native and doesn't need parselog (Surefire XML is universal) so
# we collapse their 3-agent chain into one agent that emits a single
# JSON manifest.
#
# **Write-allowed** — same rationale as SetupAgent.
#
# **Novel over both upstreams**: three-way test scope
# (test_scope_full / test_scope_local / test_scope_developer). No prior
# SWE-bench-family benchmark distinguishes between these scopes;
# `test_scope_developer` is SWT-Bench-canonical (from case.json), and
# the other two are our contribution — see
# docs/DESIGN-ENHANCED-2026-09-16.md §9.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from swe_brew.agent.action_parser import OrganizeActionParser, TestGenAction
from swe_brew.agent.framework import AgentLoop
from swe_brew.agent.prompt import (
    ORGANIZE_SYSTEM_MESSAGE,
    ORGANIZE_TASK_TEMPLATE,
    ORGANIZE_TOOL_DESCRIPTIONS,
    REACT_PROMPT,
    safe_render,
)
from swe_brew.agent.state import ChatMessage
from swe_brew.agent.tools import ToolResult, execute_action_write_allowed
from swe_brew.benchmark.runtime import runtime_spec

__all__ = ["OrganizeAgent", "OrganizeOutput", "parse_organize_manifest"]


@dataclass
class OrganizeOutput:
    """Structured output of the organize stage. Persisted to
    ``enhanced/organize.json`` per docs/DESIGN-ENHANCED-2026-09-16.md §2.1.
    """

    success:               bool
    duration_s:            float
    submitted:             bool
    test_command:          str = field(default_factory=lambda: runtime_spec().test_command)
    modules:               list[str] = field(default_factory=list)
    validated_modules:     list[str] = field(default_factory=list)
    test_scope_full:       list[str] = field(default_factory=list)      # repo-relative file paths
    test_scope_local:      list[str] = field(default_factory=list)      # scoped to validated Maven modules
    test_scope_developer:  list[str] = field(default_factory=list)      # from case.json.test_files
    pertest_template:      str = field(default_factory=lambda: runtime_spec().pertest_template)
    trials_used:           int = 0
    steps_used:            int = 0
    cost:                  dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    fallback_used:         bool = False
    notes:                 list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "success":              self.success,
            "duration_s":           self.duration_s,
            "submitted":            self.submitted,
            "test_command":         self.test_command,
            "modules":              self.modules,
            "validated_modules":    self.validated_modules,
            "test_scope_full":      self.test_scope_full,
            "test_scope_local":     self.test_scope_local,
            "test_scope_developer": self.test_scope_developer,
            "pertest_template":     self.pertest_template,
            "trials_used":          self.trials_used,
            "steps_used":           self.steps_used,
            "cost":                 self.cost,
            "fallback_used":        self.fallback_used,
            "notes":                self.notes,
        }


def parse_organize_manifest(submit_payload: str) -> dict[str, Any]:
    """Extract the JSON manifest from a submit payload.

    Accepts three forms:
      - fenced ```json { ... } ```
      - bare JSON `{ ... }`
      - "failure: <reason>" (returns {"failure": reason})
    """
    payload = submit_payload.strip()
    if payload.lower().startswith("failure"):
        _, _, reason = payload.partition(":")
        return {"failure": reason.strip() or "unspecified"}

    # Fenced block. Extract the whole fence first; then decode the first
    # balanced JSON object with ``json.JSONDecoder.raw_decode`` so string
    # literals such as ``-Dtest={cls}#{method}`` cannot terminate the
    # non-greedy ``.*?`` match early.
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)```", payload, re.DOTALL):
        obj = _decode_first_json_object(m.group(1))
        if obj is not None:
            return obj

    # Bare JSON (older agent output).
    obj = _decode_first_json_object(payload)
    if obj is not None:
        return obj
    return {"failure": f"could not parse manifest: {payload[:200]}"}


def _decode_first_json_object(text: str) -> dict[str, Any] | None:
    """Return the first balanced JSON object in ``text``, if any."""
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx = text.find("{", idx + 1)
            continue
        return obj if isinstance(obj, dict) else None
    return None


# ─── the agent ───────────────────────────────────────────────────────────────


@dataclass
class OrganizeAgent(AgentLoop):
    """Organize ReAct loop.

    ``run(case_id, repo_short, base_commit, changed_files, setup_summary)``
    → LoopResult with a JSON manifest in submit_payload. Caller wraps
    into OrganizeOutput.
    """

    role: str = field(default="organize")

    def build_prefix(
        self,
        repo_short: str = "",
        base_commit: str = "",
        changed_files: list[str] | None = None,
        setup_summary: str = "(no setup output available)",
        history_hints: str = "",
        **_ignored: Any,
    ) -> list[ChatMessage]:
        changed_files = changed_files or []
        # Feed the full changed-file set; local-scope derivation is
        # proportional to the touched modules.
        changed_list = "\n".join(f"- {p}" for p in changed_files)
        return [
            ChatMessage(role="system", content=ORGANIZE_SYSTEM_MESSAGE),
            ChatMessage(
                role="user",
                content=(
                    safe_render(REACT_PROMPT, tools=ORGANIZE_TOOL_DESCRIPTIONS)
                    + "\n"
                    + safe_render(
                        ORGANIZE_TASK_TEMPLATE,
                        repo_short=repo_short,
                        base_commit=base_commit[:12] if base_commit else "?",
                        changed_count=len(changed_files),
                        changed_files_list=changed_list or "(none)",
                        setup_summary=setup_summary,
                        history_hints=history_hints,
                    )
                ),
            ),
        ]

    def parse_action(self, response: str) -> TestGenAction | None:
        return OrganizeActionParser().parse(response)

    def execute_action(self, action: TestGenAction) -> ToolResult:
        return execute_action_write_allowed(self.sandbox, action)

    def extract_submit_payload(self, action: TestGenAction) -> str:
        # For organize, submit body contains a JSON manifest (possibly fenced).
        return action.args.strip()

    def parser_drift_help(self) -> str:
        return (
            "No valid action tag found. Reply with exactly one of: "
            "<bash>…</bash>, <apply_diff>…</apply_diff>, "
            "<mvn_test_compile></mvn_test_compile>, <mvn_test></mvn_test>, "
            "<read_surefire>…</read_surefire>, <git_grep>…</git_grep>, "
            "<search>…</search>, or <submit>…</submit>."
        )

    def trial_retry_nudge(self, trial: int) -> str:
        return (
            f"Trial {trial}/{self.config.max_trials}. Your previous "
            "submission wasn't accepted (either no submit, or the JSON "
            "manifest was malformed). Ensure your final submit is "
            "``<submit>```json { \"test_command\": ..., \"modules\": [...], "
            "\"test_scope_full_files\": [...], \"pertest_template\": ... "
            "}```</submit>`` with all required fields."
        )


# ─── validated-scope derivation (used by orchestrator after submit) ──────────


def derive_test_scope_validated(
    all_test_files: list[str],
    validated_modules: list[str],
) -> list[str]:
    """Return only reported tests whose Maven module was validated.

    The allowed set is ``validated_modules``, which is the intersection of
    organize's reported reactor modules and the modules named by its
    executed test command. Changed-file prefixes are intentionally not used.
    """
    allowed = set(validated_modules)
    return [t for t in all_test_files if t.split("/", 1)[0] in allowed]
