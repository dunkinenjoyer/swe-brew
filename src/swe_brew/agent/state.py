"""Agent state container for the test-generation workflow.

**Shape ported from** ``launch/agent/state.py:AgentState``. Ours is
smaller because our tool set is narrower — we don't need Tavily search,
LangChain messages, PyPI mirror etc. What we keep: the ``messages`` list
threaded through the loop, ``trials`` counter, cost accumulation,
``success`` flag, and the ``@auto_catch`` decorator idea (renamed
``guarded`` since we're not in a graph node context).
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Literal

__all__ = ["TestGenAgentState", "ChatMessage", "guarded"]


ChatRole = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    """One OpenAI-compat chat message. We roll our own tiny type instead
    of pulling in ``langchain_core.messages`` — it's just role + content.
    """

    role: ChatRole
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class TestGenAgentState:
    """Everything one test-gen run needs to remember.

    Direct parallels to ``launch.agent.state.AgentState``:
    - ``messages`` ← ``messages``
    - ``trials`` ← ``trials``
    - ``success`` ← ``success``
    - ``exception`` ← ``exception``
    - ``cost`` ← ``cost`` (structure preserved: dict of dicts, keyed by
      loop stage)
    - ``commands`` ← ``commands`` / ``setup_commands`` (a log of tool
      invocations for the audit trail)

    Fields we drop from RepoLaunch's state (not applicable to our narrower
    task): ``search_tool``, ``pypiserver``, ``docs``, ``date``,
    ``base_image``, ``platform``, ``preparation_commands``,
    ``parser`` / ``test_output`` / ``test_status``, ``pertest_command``.
    """

    case_id: str
    messages: list[ChatMessage] = field(default_factory=list)
    trials: int = 0
    step: int = 0
    submitted_diff: str = ""
    success: bool | None = None
    exception: Exception | None = None
    # Audit log of tool invocations: e.g. ("mvn_test_compile", exit_code, elapsed_s).
    commands: list[tuple[str, int, float]] = field(default_factory=list)
    # Cost accumulator, mirroring RepoLaunch's ``{stage: {input_tokens, output_tokens}}``.
    cost: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "test_gen": {"input_tokens": 0, "output_tokens": 0},
        },
    )
    # Structured metrics for reporting.
    metrics: dict[str, Any] = field(default_factory=dict)


def guarded(func: Callable[..., dict]) -> Callable[..., dict]:
    """Route uncaught exceptions to ``state.exception`` instead of blowing up.

    Direct port of ``launch.agent.state.auto_catch``. Renamed only to
    avoid conflating "auto-caught in a langgraph node" with our
    imperative loop step.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> dict:
        try:
            return func(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            return {
                "exception": Exception(f"{e}\n\n{traceback.format_exc()}"),
                "success": False,
            }

    return wrapper
