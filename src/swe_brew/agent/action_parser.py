"""Common action parsing utilities for agent interactions.

Grounded on ``launch/agent/action_parser.py`` — same ``<tag>…</tag>``
regex extraction, same ``<think>…</think>`` stripping, same abstract
``parse()`` interface. The concrete parser at the bottom
(``TestGenActionParser``) is our Java localisation — it knows about
our seven tools plus ``<submit>``.

Preserving the base-class shape means anyone reading RepoLaunch's
code can point at line-parallel behaviour here.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["ActionParser", "TestGenAction", "TestGenActionParser"]


class ActionParser(ABC):
    """Base class for parsing LLM responses into structured actions.

    Grounded on ``launch/agent/action_parser.py`` (the abstract shape +
    two static helpers). This class copies the base 1:1 because there
    is no Java-specific behaviour to introduce at this level.
    """

    @abstractmethod
    def parse(self, response: str) -> Any | None:
        """Parse response string into action object."""
        raise NotImplementedError

    @staticmethod
    def extract_tag_content(response: str, tag: str) -> str | None:
        """Extract content between XML-style tags."""
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1) if match else None

    @staticmethod
    def clean_response(response: str) -> str:
        """Remove reasoning tags from response if present.

        Handles two shapes:
        - ``<think>…</think>rest`` — return rest.
        - ``<reasoning>…</reasoning>rest`` — same.
        - No think tags — return as-is.
        """
        if "<think>" in response and "</think>" in response:
            return response.split("</think>", 1)[1]
        if "<reasoning>" in response and "</reasoning>" in response:
            return response.split("</reasoning>", 1)[1]
        return response


# ─── Java-specific action shape ──────────────────────────────────────────────


ActionKind = Literal[
    "apply_diff",         # apply a unified diff to /testbed on the host
    "mvn_test_compile",   # run `mvn test-compile` and return stdout+errors
    "mvn_test",           # run the full mvn test command; returns Surefire summary
    "mvn_install",        # setup-only: `mvn install -DskipTests`
    "read_surefire",      # dump one Surefire XML by path (for debugging failures)
    "git_grep",           # search HEAD for a symbol/class/method
    "bash",               # bash — allowlisted for test_gen; full for setup + organize
    "search",             # Tavily web search — public knowledge only; repo source out of scope
    "submit",             # terminal action — ship the final diff / manifest / build-ok signal
]


@dataclass
class TestGenAction:
    """One action emitted by the LLM in a single turn."""

    kind: ActionKind
    args: str                 # tag body — semantics depend on ``kind``


# Alias: same shape works for setup + organize agents; the role changes
# what the executor DOES, not the parsed structure.
AgentAction = TestGenAction


class TestGenActionParser(ActionParser):
    """Parse ``<tag>…</tag>`` actions for the test-gen loop.

    Precedence: submit > apply_diff > mvn_test > mvn_test_compile > read_surefire > git_grep.
    Only the *first* matching tag wins per response, matching RepoLaunch's
    "one action per turn" contract.
    """

    def parse(self, response: str) -> TestGenAction | None:
        response = self.clean_response(response)

        # Terminal first — if the LLM decides to submit, that's the whole turn.
        submit = self.extract_tag_content(response, "submit")
        if submit is not None:
            return TestGenAction(kind="submit", args=submit)

        for tag, kind in (
            ("apply_diff", "apply_diff"),
            ("mvn_test_compile", "mvn_test_compile"),
            ("mvn_test", "mvn_test"),
            ("mvn_install", "mvn_install"),
            ("read_surefire", "read_surefire"),
            ("git_grep", "git_grep"),
            ("bash", "bash"),
            ("search", "search"),
        ):
            body = self.extract_tag_content(response, tag)
            if body is not None:
                return TestGenAction(kind=kind, args=body)  # type: ignore[arg-type]

        return None


class SetupActionParser(TestGenActionParser):
    """Same precedence as test-gen. Present as a named class so the
    setup agent's construction reads clean: SetupAgent uses
    SetupActionParser, OrganizeAgent uses OrganizeActionParser, etc.
    """


class OrganizeActionParser(TestGenActionParser):
    """Same precedence as test-gen."""
