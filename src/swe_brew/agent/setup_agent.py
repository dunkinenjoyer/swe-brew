"""SetupAgent — the write-allowed ReAct agent that validates the project builds.

# PROVENANCE
# ==========
# Grounded on ``SWE-bench-Live/launch/launch/agent/setup/setup.py::setup()``
# lines 20-30 (system_msg), 36-80 (SetupAction.prompt), 222-326 (loop).
#
# The prompt-level provenance is documented in ``prompt.py`` next to
# each constant (SETUP_SYSTEM_MESSAGE, SETUP_TOOL_DESCRIPTIONS,
# SETUP_TASK_TEMPLATE). Their Docker/testbed clauses are byte-preserved;
# the objective is substituted for our narrower purpose ("validate this
# specific project builds against the pre-baked image").
#
# **Write-allowed** per user directive 2026-09-16: this agent can run
# state-mutating commands. Rationale: `mvn install` is sometimes required
# to prime the dependency cache before `mvn compile`
# succeeds; settings.xml overrides may be needed for projects that
# declare their own repositories. This matches RepoLaunch's setup agent
# which runs `apt install`, `pip install`, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from swe_brew.agent.action_parser import SetupActionParser, TestGenAction
from swe_brew.agent.framework import AgentLoop
from swe_brew.agent.prompt import (
    REACT_PROMPT,
    SETUP_SYSTEM_MESSAGE,
    SETUP_TASK_TEMPLATE,
    SETUP_TOOL_DESCRIPTIONS,
    safe_render,
)
from swe_brew.agent.state import ChatMessage
from swe_brew.agent.tools import ToolResult, execute_action_write_allowed
from swe_brew.benchmark.runtime import runtime_spec

__all__ = ["SetupAgent", "SetupOutput"]


@dataclass
class SetupOutput:
    """Structured output of the setup stage. Persisted to
    ``enhanced/setup.json`` per docs/DESIGN-ENHANCED-2026-09-16.md §2.1.
    """

    success:        bool
    duration_s:     float
    submitted:      bool
    commands_run:   list[str] = field(default_factory=list)
    modifications:  list[dict] = field(default_factory=list)
    submit_notes:   str = ""
    trials_used:    int = 0
    steps_used:     int = 0
    cost:           dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    transcript:     str = ""
    notes:          list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "success":        self.success,
            "duration_s":     self.duration_s,
            "submitted":      self.submitted,
            "commands_run":   self.commands_run,
            "modifications":  self.modifications,
            "submit_notes":   self.submit_notes,
            "trials_used":    self.trials_used,
            "steps_used":     self.steps_used,
            "cost":           self.cost,
            "notes":          self.notes,
        }


@dataclass
class SetupAgent(AgentLoop):
    """Setup ReAct loop.

    ``run(case_id, repo_short, base_commit, changed_files)`` → LoopResult.
    Caller (enhance.py orchestrator) is responsible for wrapping the
    result into a ``SetupOutput`` + writing setup.json.
    """

    role: str = field(default="setup")

    def build_prefix(
        self,
        repo_short: str = "",
        base_commit: str = "",
        changed_files: list[str] | None = None,
        history_hints: str = "",
        **_ignored: Any,
    ) -> list[ChatMessage]:
        changed_files = changed_files or []
        base_image = runtime_spec().base_image
        # Feed the full changed-file set; setup needs to see every module
        # the patch touches to decide the ``-pl`` scope.
        changed_list = "\n".join(f"- {p}" for p in changed_files)
        return [
            ChatMessage(role="system", content=safe_render(SETUP_SYSTEM_MESSAGE, base_image=base_image)),
            ChatMessage(
                role="user",
                content=(
                    safe_render(REACT_PROMPT, tools=SETUP_TOOL_DESCRIPTIONS)
                    + "\n"
                    + safe_render(
                        SETUP_TASK_TEMPLATE,
                        repo_short=repo_short,
                        base_commit=base_commit[:12] if base_commit else "?",
                        changed_count=len(changed_files),
                        changed_files_list=changed_list or "(none)",
                        history_hints=history_hints,
                    )
                ),
            ),
        ]

    def parse_action(self, response: str) -> TestGenAction | None:
        return SetupActionParser().parse(response)

    def execute_action(self, action: TestGenAction) -> ToolResult:
        return execute_action_write_allowed(self.sandbox, action)

    def extract_submit_payload(self, action: TestGenAction) -> str:
        # Setup's submit body is free-form notes about what changed.
        # No structured payload — the modifications are already captured
        # by the sandbox's git working tree, and the orchestrator will
        # diff before/after.
        return action.args.strip()

    def parser_drift_help(self) -> str:
        return (
            "No valid action tag found. Reply with exactly one of: "
            "<bash>…</bash>, <apply_diff>…</apply_diff>, <mvn_install></mvn_install>, "
            "<mvn_test_compile></mvn_test_compile>, <mvn_test></mvn_test>, "
            "<read_surefire>…</read_surefire>, <git_grep>…</git_grep>, "
            "<search>…</search>, or <submit>…</submit>."
        )

    def trial_retry_nudge(self, trial: int) -> str:
        return (
            f"Trial {trial}/{self.config.max_trials}. The build hasn't "
            "succeeded yet. Start by re-diagnosing: is `mvn compile` still "
            "failing? If so, why? Adjust settings.xml, run `mvn install "
            "-DskipTests` only if required, or narrow `-pl` based on the "
            "actual reactor. When the build works, submit."
        )
