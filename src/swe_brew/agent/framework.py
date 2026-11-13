"""Shared ReAct framework for the setup, organize, and test-gen agents.

Grounded on RepoLaunch's core agent loop
(``SWE-bench-Live/launch/launch/agent/setup/setup.py::setup()``,
lines 222-326). Each role is a thin subclass of :class:`AgentLoop`
that plugs in its own prompt + action parser + tool executor and
inherits the message threading, cost accounting, telemetry, submit
handling, LLM retry, and trial-retry machinery.

Full-history conversation window: we do not summarise older turns for
the LLM's view; we send everything and rely on the model's large
context. RepoLaunch's ``VERIFY_CONVERSATION_WINDOW`` capped at 40-80
turns for smaller models; ours is unbounded
(``AgentConfig.conversation_window = None``).
"""

from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from swe_brew.agent.progress import TelemetryLog
from swe_brew.agent.prompt import render_history_hints
from swe_brew.agent.state import ChatMessage, TestGenAgentState
from swe_brew.benchmark.container import TestSandbox
from swe_brew.benchmark.generate import GenerationError
from swe_brew.benchmark.runtime import runtime_spec

__all__ = [
    "AgentConfig",
    "AgentLoop",
    "AgentRole",
    "LoopResult",
    "LLMClient",
]


# The three pipeline roles. Used to key the per-role cost accumulator
# in :class:`TestGenAgentState` and to prefix telemetry events by stage.
AgentRole = str    # Literal["setup", "organize", "test_gen"] — kept str for extensibility


# Parser-drift safety valve. If an agent emits no valid action tag for
# this many consecutive turns, abort the trial. Without this cap an
# LLM stuck in a hallucination loop with ``max_steps_per_trial=None``
# would burn tokens indefinitely. The 25-turn threshold tolerates the
# occasional format-recovery a large observation may trigger while
# still cutting a truly lost agent cleanly.
_MAX_CONSECUTIVE_DRIFTS = 25

# LLM HTTP retry policy. Exponential backoff on transient failures
# (5xx, 429, timeouts, transport errors). Non-transient failures (auth,
# 400 bad-request, JSON parse) fail fast. 10 attempts with min(2^n, 60)
# backoff give ~5-6 minutes of retry runway before giving up; the
# per-trial watchdog is the outer bound so we can't retry forever.
_LLM_RETRY_MAX_ATTEMPTS = 10
_LLM_RETRY_BASE_DELAY_S = 2.0
_LLM_RETRY_MAX_DELAY_S = 60.0
_LLM_RETRY_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

# Body markers indicating a transient upstream platform error. the configured model gateway
# occasionally returns HTTP 200 with ``PLATFORM_RUN_ERROR`` in the body
# — a purely status-code-driven retry misses this; match on either the
# English code or the Chinese message.
_LLM_RETRY_BODY_MARKERS = (
    "PLATFORM_RUN_ERROR",
    "平台运行错误",
)

# Single per-trial wall-clock watchdog. One budget per trial, not many
# nested budgets. Two hours is generous enough for a legitimately slow
# multi-module ``mvn install`` plus a long compile-error debug session,
# and only fires on a genuinely hung LLM roundtrip or a reasoning loop
# that survives the drift kill-switch.
_MAX_TRIAL_SECONDS = runtime_spec().trial_timeout_seconds


def _trial_timeout() -> int:
    """Return the current trial wall-clock timeout."""
    return runtime_spec().trial_timeout_seconds

# Heartbeat cadence for long-running tool calls. Between the
# ``llm.call.end`` that decided on a tool and the ``agent.turn`` that
# lands when the tool returns, an ``<mvn_test>`` or ``<mvn_install>``
# can be silent for 30-60 minutes. Without a heartbeat the watcher's
# freshness metric would mistakenly flag this as a stall.
_TOOL_HEARTBEAT_SECONDS = 60.0


# ─── LLM transport ───────────────────────────────────────────────────────────


@dataclass
class LLMClient:
    """OpenAI-compat chat client, message-list-based (multi-turn).

    Emits ``llm.call.begin`` / ``llm.call.end`` telemetry per HTTP call so
    the watcher can compute LLM-side latency independently of tool time.
    Role-aware — the ``role`` field becomes part of the event payload so
    fleet-level dashboards can attribute cost per stage.
    """

    base_url: str
    api_key: str
    model: str
    reasoning_effort: str
    verbosity: str
    fallback_model: str = ""
    role: AgentRole = "test_gen"
    max_output_tokens: int = 16000
    timeout: float = 300.0
    max_input_tokens: int = 900_000
    telemetry: TelemetryLog | None = None
    trial: int = 0
    step: int = 0

    def chat(self, messages: list[ChatMessage]) -> tuple[str, dict]:
        """Try the primary model, then fall back on an exhausted primary.

        The agent loop should not silently use a stale fallback. Keep the
        primary model name stable in state, but route one full retry cycle
        through the fallback model before re-raising.
        """
        try:
            return self._chat_current_model(messages)
        except Exception:
            if not self.fallback_model or self.fallback_model == self.model:
                raise
            original_model = self.model
            self.model = self.fallback_model
            try:
                return self._chat_current_model(messages)
            finally:
                self.model = original_model

    def _chat_current_model(self, messages: list[ChatMessage]) -> tuple[str, dict]:
        """Call the LLM.

        Wraps the HTTP roundtrip in a retry loop with exponential backoff
        for transient failures (429, 5xx, timeouts, connection errors).
        Non-transient failures (auth, 400, JSON parse) fail fast. Every
        attempt emits its own ``llm.call.begin`` / ``llm.call.end``
        telemetry so the watcher's health panel gets a full audit trail —
        a retry looks like a failed call followed by a successful one.
        """
        # Context-length guard. The historical burn mode was repeatedly
        # resubmitting a growing prompt until the provider returned HTTP
        # 400 ``context_length_exceeded``. Estimate token count up front
        # and abort with a readable reason instead of burning a fallback
        # call and another full trial on a prompt that can never fit.
        estimated_tokens = sum(len(m.content) // 4 for m in messages)
        if estimated_tokens > self.max_input_tokens:
            raise GenerationError(
                f"the configured model gateway {self.model}: estimated input "
                f"{estimated_tokens} tokens exceeds cap {self.max_input_tokens}"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "max_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.verbosity:
            payload["verbosity"] = self.verbosity

        last_exc: Exception | None = None
        for attempt in range(1, _LLM_RETRY_MAX_ATTEMPTS + 1):
            if self.telemetry is not None:
                self.telemetry.emit(
                    "llm.call.begin",
                    role=self.role,
                    model=self.model, msg_count=len(messages),
                    trial=self.trial, step=self.step + 1,
                    attempt=attempt,
                )
            t0 = time.perf_counter()
            http_status: int | str = "?"
            error: str | None = None
            transient = False
            try:
                # Per-component httpx timeouts so a stuck connect / write /
                # read all fire deterministically. ``connect`` capped at
                # 30s; read / write /
                # pool use the configured timeout as the actual LLM
                # latency budget per attempt.
                httpx_timeout = httpx.Timeout(
                    self.timeout,
                    connect=min(30.0, self.timeout),
                )
                with httpx.Client(timeout=httpx_timeout) as client:
                    r = client.post(
                        f"{self.base_url.rstrip('/')}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                http_status = r.status_code
                # Check the response body for platform-error markers
                # BEFORE branching on status code — the configured model gateway occasionally
                # returns HTTP 500 with ``PLATFORM_RUN_ERROR`` in the
                # body, which we always want to treat as transient so
                # retries have a chance during a platform-side blip.
                body_preview = r.text[:1000] if r.text else ""
                body_indicates_transient = any(
                    marker in body_preview for marker in _LLM_RETRY_BODY_MARKERS
                )
                if r.status_code >= 400 or body_indicates_transient:
                    error = f"HTTP {r.status_code}: {body_preview[:400]}"
                    transient = (
                        r.status_code in _LLM_RETRY_STATUS_CODES
                        or body_indicates_transient
                    )
                    raise GenerationError(f"the configured model gateway {self.model} → {error}")
                data = r.json()
                if "choices" not in data:
                    error = f"missing 'choices': {data}"
                    raise GenerationError(f"the configured model gateway {self.model} {error}")
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
            except httpx.TimeoutException as e:
                # Covers ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout.
                error = f"{type(e).__name__}: {e!s}"
                transient = True
                last_exc = e
                if self.telemetry is not None:
                    self.telemetry.emit(
                        "llm.call.end",
                        role=self.role,
                        ok=False, model=self.model, error=error,
                        elapsed_s=round(time.perf_counter() - t0, 3),
                        http_status=http_status,
                        trial=self.trial, step=self.step + 1,
                        attempt=attempt, transient=True,
                    )
            except httpx.HTTPError as e:
                # Any httpx transport-level error is transient by
                # definition; retrying is safe because the LLM call is
                # idempotent from our side (payload unchanged between
                # attempts). ``httpx.HTTPError`` covers ConnectError,
                # RemoteProtocolError, ReadError, WriteError, PoolTimeout,
                # NetworkError, and anything future httpx versions add.
                error = f"{type(e).__name__}: {e!s}"
                transient = True
                last_exc = e
                if self.telemetry is not None:
                    self.telemetry.emit(
                        "llm.call.end",
                        role=self.role,
                        ok=False, model=self.model, error=error,
                        elapsed_s=round(time.perf_counter() - t0, 3),
                        http_status=http_status,
                        trial=self.trial, step=self.step + 1,
                        attempt=attempt, transient=True,
                    )
            except Exception as e:
                if error is None:
                    error = f"{type(e).__name__}: {e!s}"
                last_exc = e
                if self.telemetry is not None:
                    self.telemetry.emit(
                        "llm.call.end",
                        role=self.role,
                        ok=False, model=self.model, error=error,
                        elapsed_s=round(time.perf_counter() - t0, 3),
                        http_status=http_status,
                        trial=self.trial, step=self.step + 1,
                        attempt=attempt, transient=transient,
                    )
                if not transient or attempt >= _LLM_RETRY_MAX_ATTEMPTS:
                    raise
                # Fall through into the backoff sleep below.
            else:
                # Success path.
                elapsed = time.perf_counter() - t0
                if self.telemetry is not None:
                    self.telemetry.emit(
                        "llm.call.end",
                        role=self.role,
                        ok=True, model=self.model,
                        elapsed_s=round(elapsed, 3),
                        http_status=http_status,
                        tokens_in=int(usage.get("prompt_tokens") or 0),
                        tokens_out=int(usage.get("completion_tokens") or 0),
                        trial=self.trial, step=self.step + 1,
                        attempt=attempt,
                    )
                return content, usage

            # Transient failure — back off and retry.
            if attempt >= _LLM_RETRY_MAX_ATTEMPTS:
                break
            delay = min(
                _LLM_RETRY_BASE_DELAY_S * (2 ** (attempt - 1)),
                _LLM_RETRY_MAX_DELAY_S,
            )
            if self.telemetry is not None:
                self.telemetry.emit(
                    "llm.call.retry",
                    role=self.role,
                    attempt=attempt, next_attempt=attempt + 1,
                    backoff_s=round(delay, 2),
                    reason=error or "unknown",
                    trial=self.trial, step=self.step + 1,
                )
            time.sleep(delay)

        # All retries exhausted — re-raise the last transient exception.
        if last_exc is not None:
            raise last_exc
        raise GenerationError(
            f"the configured model gateway {self.model}: LLM retries exhausted with no captured exception",
        )


# ─── config ──────────────────────────────────────────────────────────────────


@dataclass
class AgentConfig:
    """Per-agent runtime knobs. Defaults implement the 'no resource caps'
    directive from the 2026-09-16 design discussion.

    ``max_trials``: soft cap. Set high; a *runaway* infinite would still
        be bounded, but we don't want the agent to feel budgeted. Default
        10 for all three roles.
    ``max_steps_per_trial``: **None** = unbounded. The agent submits when
        it's ready; no artificial ceiling.
    ``conversation_window``: **None** = send full history. Rely on the
        model's context window. RepoLaunch's fixed 40-80 was for smaller
        models.
    """

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    fallback_model: str = ""
    reasoning_effort: str = ""
    verbosity: str = ""
    max_output_tokens: int = 16000
    max_input_tokens: int = 900_000
    timeout: float = 300.0
    max_trials: int = 10
    max_steps_per_trial: int | None = None
    conversation_window: int | None = None

    def __post_init__(self) -> None:
        self.base_url = self.base_url or os.environ.get("SWE_BREW_MODEL_BASE_URL", "").rstrip("/")
        self.api_key = self.api_key or os.environ.get("SWE_BREW_MODEL_API_KEY", "")
        self.model = self.model or os.environ.get("SWE_BREW_MODEL", "bailian/deepseek-v4-pro")
        self.fallback_model = self.fallback_model or os.environ.get("SWE_BREW_MODEL_FALLBACK", "")
        self.max_input_tokens = int(os.environ.get(
            "SWE_BREW_MAX_INPUT_TOKENS",
            self.max_input_tokens,
        ))
        self.reasoning_effort = self.reasoning_effort or os.environ.get("SWE_BREW_REASONING_EFFORT", "high")
        self.verbosity = self.verbosity or os.environ.get("SWE_BREW_VERBOSITY", "medium")
        if not self.base_url or not self.api_key:
            raise GenerationError("SWE_BREW_MODEL_BASE_URL and SWE_BREW_MODEL_API_KEY must be set")


# ─── loop result ─────────────────────────────────────────────────────────────


@dataclass
class LoopResult:
    """Return value of ``AgentLoop.run``.

    ``submit_payload`` semantics are role-specific:
      - setup: usually empty; agent submits to signal 'build validated'.
      - organize: JSON blob with test_command + scopes.
      - test_gen: the candidate unified diff.
    """

    role: AgentRole
    submit_payload: str
    submitted: bool
    trials_used: int
    steps_used: int
    state: TestGenAgentState
    tool_calls: list[tuple[str, int, float]] = field(default_factory=list)


# ─── abstract base ───────────────────────────────────────────────────────────


@dataclass
class AgentLoop(ABC):
    """Shared ReAct machinery. Subclasses plug in prompts + actions + tools.

    Subclass responsibilities (implement all four):
      - ``build_prefix(*args)`` — the [system, task] prefix messages for
        this role.
      - ``parse_action(response)`` — return a role-specific action object
        or ``None`` if parser drift.
      - ``execute_action(action)`` — dispatch to the role's tool set,
        return ``ToolResult``.
      - ``extract_submit_payload(action)`` — pull the payload out of a
        submit action.
    """

    sandbox: TestSandbox
    case_id: str
    role: AgentRole
    telemetry: TelemetryLog | None = None
    config: AgentConfig = field(default_factory=AgentConfig)

    def __post_init__(self) -> None:
        if self.role not in {"setup", "organize", "test_gen"}:
            raise ValueError(f"unknown agent role: {self.role!r}")

    # ─── subclass hooks ─────────────────────────────────────────────────

    @abstractmethod
    def build_prefix(self, **kwargs: Any) -> list[ChatMessage]:
        """The system + user prefix for this role. Called once per run."""

    @abstractmethod
    def parse_action(self, response: str) -> Any:
        """Parse the LLM response into an action object; None on drift."""

    @abstractmethod
    def execute_action(self, action: Any) -> ToolResult:       # noqa: F821
        """Dispatch to the role's tool executor."""

    @abstractmethod
    def extract_submit_payload(self, action: Any) -> str:
        """Extract the final payload from a submit action."""

    def validate_submit_payload(self, payload: str, state: Any) -> tuple[bool, str]:
        """Optional role-specific submit preflight.

        Returning ``False`` keeps the current trial alive and feeds the
        reason back to the model as an Observation. The default is a
        no-op, so setup and organize retain their existing submit
        behavior. Test generation overrides this to reject candidates
        whose added tests never execute in the pre-patch container.
        """
        return True, ""

    @abstractmethod
    def parser_drift_help(self) -> str:
        """Reminder string to send when parser drifts (no action tag)."""

    @abstractmethod
    def trial_retry_nudge(self, trial: int) -> str:
        """Message inserted at the start of trial N+1 if the previous
        trial exhausted without submitting."""

    def tool_command(self, action: Any) -> str:
        """Return the command string for a tool action, used for timeout blacklisting."""
        return getattr(action, "args", "") or getattr(action, "kind", "")

    # ─── the loop ───────────────────────────────────────────────────────

    def run(self, **prefix_kwargs: Any) -> LoopResult:
        """Drive the ReAct loop until submit or trial exhaustion."""
        # Ensure the cost dict has our role slot.
        state = TestGenAgentState(case_id=self.case_id)
        state.cost.setdefault(self.role, {"input_tokens": 0, "output_tokens": 0})

        client = LLMClient(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model=self.config.model,
            fallback_model=self.config.fallback_model,
            reasoning_effort=self.config.reasoning_effort,
            verbosity=self.config.verbosity,
            max_output_tokens=self.config.max_output_tokens,
            max_input_tokens=self.config.max_input_tokens,
            timeout=self.config.timeout,
            role=self.role,
            telemetry=self.telemetry,
        )

        prefix = self.build_prefix(**prefix_kwargs)
        state.messages.extend(prefix)
        prefix_len = len(prefix)

        best_payload = ""
        avoid_commands: list[str] = []
        for trial in range(self.config.max_trials):
            state.trials = trial + 1
            self._emit("agent.trial.begin",
                       trial=state.trials, max_trials=self.config.max_trials)

            if trial > 0:
                nudge = self.trial_retry_nudge(trial + 1)
                avoid_block = render_history_hints([], [], avoid_commands=avoid_commands)
                if avoid_block:
                    nudge = f"{nudge}\n\n{avoid_block}"
                state.messages.append(
                    ChatMessage(role="user", content=nudge),
                )

            client.trial = state.trials
            step = 0
            consecutive_drifts = 0
            trial_start = time.monotonic()
            # Defensive guard: a non-positive ``max_steps_per_trial``
            # (0 or negative) would silently no-op every trial; treat as
            # "no cap" instead of "run zero steps".
            effective_max_steps = self.config.max_steps_per_trial
            if effective_max_steps is not None and effective_max_steps <= 0:
                effective_max_steps = None
            while True:
                step += 1
                # Hard cap only if configured (default: no cap). Check BEFORE
                # committing state so state.step reflects the last completed step,
                # not the one we bailed on.
                if (effective_max_steps is not None
                        and step > effective_max_steps):
                    break
                # Wall-clock watchdog. Fires between turns to catch the
                # aggregate case where many legitimate turns collectively
                # exceed the trial budget (the httpx timeout + LLM retry
                # loop covers a single hung roundtrip).
                if time.monotonic() - trial_start > _trial_timeout():
                    self._emit("agent.trial.timeout",
                               reason=f"wall-clock exceeded {_trial_timeout()}s",
                               trial=state.trials, step=state.step,
                               elapsed_s=round(time.monotonic() - trial_start, 1))
                    break
                state.step = step
                client.step = step

                window = self._sliding_window(state.messages, prefix_len)
                content, usage = client.chat(window)
                self._accrue_cost(state, usage)
                state.messages.append(ChatMessage(role="assistant", content=content))

                action = self.parse_action(content)
                if action is None:
                    consecutive_drifts += 1
                    state.messages.append(
                        ChatMessage(role="user",
                                    content=f"Observation:\n{self.parser_drift_help()}"))
                    self._emit_turn(state, "__none__", exit_code=1, elapsed_s=0.0, usage=usage)
                    # Safety valve: N consecutive parser drifts break the
                    # trial before an unbounded hallucination loop can
                    # burn steps × tokens forever.
                    if consecutive_drifts >= _MAX_CONSECUTIVE_DRIFTS:
                        self._emit("agent.trial.abort",
                                   reason=f"{consecutive_drifts} consecutive parser drifts",
                                   trial=state.trials, step=state.step)
                        break
                    continue
                consecutive_drifts = 0            # reset on any valid action

                # Terminal — submit.
                if getattr(action, "kind", None) == "submit":
                    best_payload = self.extract_submit_payload(action)
                    accepted, rejection = self.validate_submit_payload(
                        best_payload,
                        state,
                    )
                    if not accepted:
                        self._emit(
                            "agent.submit.rejected",
                            trial=state.trials,
                            step=state.step,
                            reason=rejection[:400],
                        )
                        state.messages.append(
                            ChatMessage(
                                role="user",
                                content=(
                                    "Observation:\n"
                                    f"Submission rejected: {rejection}\n"
                                    "Do not resubmit until the issue is fixed."
                                ),
                            ),
                        )
                        continue
                    self._emit("agent.submit",
                               trial=state.trials, step=state.step,
                               payload_chars=len(best_payload))
                    return LoopResult(
                        role=self.role,
                        submit_payload=best_payload,
                        submitted=True,
                        trials_used=state.trials,
                        steps_used=state.step,
                        state=state,
                        tool_calls=list(state.commands),
                    )

                # Emit ``agent.tool.begin`` immediately before the tool
                # dispatches so the watcher can distinguish "waiting on
                # a long tool" from "Python hung". A heartbeat thread
                # emits ``agent.tool.heartbeat`` every
                # ``_TOOL_HEARTBEAT_SECONDS`` while the tool is in
                # flight — otherwise a 30-minute ``<mvn_test>`` would
                # produce zero telemetry between the LLM call and the
                # agent turn that lands when the tool returns.
                action_kind = getattr(action, "kind", "?")
                tool_t0 = time.monotonic()
                self._emit("agent.tool.begin",
                           trial=state.trials, step=state.step,
                           action_kind=action_kind)
                heartbeat_stop = self._start_tool_heartbeat(
                    action_kind=action_kind,
                    trial=state.trials, step=state.step,
                    tool_t0=tool_t0,
                )
                try:
                    result = self.execute_action(action)
                except Exception as tool_exc:
                    # If ``execute_action`` raises, we've already emitted
                    # ``agent.tool.begin`` but the closing ``agent.turn``
                    # never fires. The watcher's in-flight marker would
                    # stick forever, biasing every subsequent stall check
                    # to the wrong threshold. Emit ``agent.tool.error``
                    # so the watcher can clear the marker + record the
                    # failure, then re-raise so the outer orchestrator
                    # sees the exception.
                    self._emit("agent.tool.error",
                               trial=state.trials, step=state.step,
                               action_kind=action_kind,
                               error=f"{type(tool_exc).__name__}: {tool_exc!s}"[:400],
                               elapsed_s=round(time.monotonic() - tool_t0, 3))
                    raise
                finally:
                    heartbeat_stop.set()
                command_text = self.tool_command(action).strip()
                if (getattr(result, "timed_out", False) or result.exit_code == 124):
                    if command_text and command_text not in avoid_commands:
                        avoid_commands.append(command_text)
                        self._emit(
                            "agent.tool.timeout",
                            trial=state.trials,
                            step=state.step,
                            action_kind=action_kind,
                            command=command_text[:400],
                        )
                state.commands.append(
                    (action_kind, result.exit_code, result.elapsed_s),
                )
                self._emit_turn(state, action_kind,
                                exit_code=result.exit_code,
                                elapsed_s=round(result.elapsed_s, 3),
                                usage=usage)
                state.messages.append(
                    ChatMessage(role="user",
                                content=f"Observation:\n{result.truncated_content}"),
                )

        # All trials exhausted without submit.
        return LoopResult(
            role=self.role,
            submit_payload=best_payload,
            submitted=False,
            trials_used=state.trials,
            steps_used=state.step,
            state=state,
            tool_calls=list(state.commands),
        )

    # ─── helpers ────────────────────────────────────────────────────────

    def _sliding_window(
        self, messages: list[ChatMessage], prefix_len: int,
    ) -> list[ChatMessage]:
        """Prefix + tail. When ``conversation_window is None`` (default),
        send the FULL history — rely on the model's large context. When
        set to N, send prefix + last N messages.
        """
        cw = self.config.conversation_window
        if cw is None or len(messages) <= prefix_len + cw:
            return list(messages)
        prefix = messages[:prefix_len]
        tail = messages[-cw:]
        return list(prefix) + list(tail)

    def _accrue_cost(self, state: TestGenAgentState, usage: dict) -> None:
        c = state.cost.setdefault(self.role, {"input_tokens": 0, "output_tokens": 0})
        c["input_tokens"] += int(usage.get("prompt_tokens") or 0)
        c["output_tokens"] += int(usage.get("completion_tokens") or 0)

    def _start_tool_heartbeat(
        self, *, action_kind: str, trial: int, step: int, tool_t0: float,
    ) -> threading.Event:
        """Spin a daemon thread that emits ``agent.tool.heartbeat`` every
        ``_TOOL_HEARTBEAT_SECONDS`` seconds until the returned Event is
        set. Idempotent no-op when telemetry is disabled.

        Called by :meth:`AgentLoop.run` immediately before
        ``execute_action`` so the fleet monitor can distinguish "waiting
        on a long tool" from "Python hung" while a single
        ``<mvn_test>`` or ``<mvn_install>`` runs for tens of minutes.
        """
        stop = threading.Event()
        if self.telemetry is None:
            return stop

        def _pump() -> None:
            while not stop.wait(_TOOL_HEARTBEAT_SECONDS):
                elapsed = time.monotonic() - tool_t0
                try:
                    self._emit(
                        "agent.tool.heartbeat",
                        trial=trial, step=step,
                        action_kind=action_kind,
                        elapsed_s=round(elapsed, 1),
                    )
                except Exception:                # noqa: BLE001
                    # A telemetry sink failure must NEVER kill the tool
                    # thread's parent process. Swallow and keep pumping —
                    # the next tick will retry, or the caller's ``stop``
                    # will end the loop cleanly.
                    return

        thread = threading.Thread(
            target=_pump,
            name=f"tool-heartbeat-{self.role}-t{trial}-s{step}",
            daemon=True,
        )
        thread.start()
        return stop

    def _emit(self, event: str, **payload: object) -> None:
        if self.telemetry is not None:
            self.telemetry.emit(event, role=self.role, **payload)

    def _emit_turn(
        self, state: TestGenAgentState, action_kind: str,
        exit_code: int, elapsed_s: float, usage: dict,
    ) -> None:
        c = state.cost[self.role]
        self._emit(
            "agent.turn",
            trial=state.trials, step=state.step,
            action_kind=action_kind, exit_code=exit_code, elapsed_s=elapsed_s,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            cost_input_tokens_cumulative=c["input_tokens"],
            cost_output_tokens_cumulative=c["output_tokens"],
        )

    def render_transcript(self, state: TestGenAgentState) -> str:
        """Flatten the message log for audit trails."""
        return "\n\n---\n\n".join(
            f"[{m.role}]\n{m.content}" for m in state.messages
        )
