"""Agentic pipeline: setup, organize, test-generation.

Each role is a thin subclass of :class:`AgentLoop` that plugs in its own
prompt, action parser, and tool executor. All three share the ReAct loop
machinery (message threading, telemetry, LLM retry, trial-level wall-
clock watchdog) via :mod:`swe_brew.agent.framework`.

    framework.py       shared AgentLoop + LLMClient
    setup_agent.py     validates the project builds against the sandbox
    organize_agent.py  discovers the mvn test_command + full test inventory
    loop.py            TestGenAgent + AgentGenerator adapter
    action_parser.py   tag-based action parsing shared across all roles
    tools.py           tool executors (read-only vs. write-allowed variants)
    state.py           shared AgentState
    prompt.py          prompt constants (per-role + shared ReAct scaffold)
    progress.py        rich.Progress + JSONL telemetry sink
    watch.py           fleet monitor CLI (``swe-brew-watch``)
"""

from swe_brew.agent.action_parser import (
    ActionParser,
    OrganizeActionParser,
    SetupActionParser,
    TestGenAction,
    TestGenActionParser,
)
from swe_brew.agent.framework import (
    AgentConfig,
    AgentLoop,
    AgentRole,
    LLMClient,
    LoopResult,
)
from swe_brew.agent.loop import AgentGenerator, TestGenAgent
from swe_brew.agent.organize_agent import (
    OrganizeAgent,
    OrganizeOutput,
    derive_test_scope_validated,
    parse_organize_manifest,
)
from swe_brew.agent.progress import TelemetryLog, make_progress, stage
from swe_brew.agent.setup_agent import SetupAgent, SetupOutput
from swe_brew.agent.state import ChatMessage, TestGenAgentState
from swe_brew.agent.tools import (
    ToolResult,
    execute_action,
    execute_action_write_allowed,
    extract_diff_from_submit,
)

__all__ = [
    # Framework
    "AgentConfig",
    "AgentLoop",
    "AgentRole",
    "LLMClient",
    "LoopResult",
    # State + messages
    "ChatMessage",
    "TestGenAgentState",
    # Parsers
    "ActionParser",
    "SetupActionParser",
    "OrganizeActionParser",
    "TestGenActionParser",
    "TestGenAction",
    # Agents
    "SetupAgent",
    "SetupOutput",
    "OrganizeAgent",
    "OrganizeOutput",
    "TestGenAgent",
    "AgentGenerator",
    # Tools
    "ToolResult",
    "execute_action",
    "execute_action_write_allowed",
    "extract_diff_from_submit",
    # Helpers
    "derive_test_scope_validated",
    "parse_organize_manifest",
    # Progress / telemetry
    "TelemetryLog",
    "make_progress",
    "stage",
]
