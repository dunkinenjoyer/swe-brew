"""Test-generation ReAct agent — thin subclass of :class:`AgentLoop`.

Two-part split:

* :class:`~swe_brew.agent.framework.AgentLoop` owns the shared ReAct
  machinery (message threading, cost accounting, telemetry, submit
  handling, trial retry, wall-clock watchdog, tool heartbeat).
* :class:`TestGenAgent` plugs in test-gen-specific prompts, the read-
  only tool executor (test-gen writes tests, not build state), and
  the :class:`AgentGenerator` adapter used by ``swe-brew-enhance``.

Prompt-slot renderers (``attempt_hint_for``, ``render_cap_clause``,
``render_siblings_block``) live in :mod:`swe_brew.agent.prompt` — next
to the template they consume — so all three roles keep matching flat
``build_prefix`` shapes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from swe_brew.agent.action_parser import TestGenAction, TestGenActionParser
from swe_brew.agent.framework import AgentConfig, AgentLoop, LoopResult
from swe_brew.agent.progress import TelemetryLog
from swe_brew.agent.prompt import (
    REACT_PROMPT,
    SYSTEM_MESSAGE,
    TASK_MESSAGE_TEMPLATE,
    TOOL_DESCRIPTIONS,
    attempt_hint_for,
    render_cap_clause,
    render_organize_notes_block,
    render_siblings_block,
    render_validated_modules_block,
    safe_render,
)
from swe_brew.agent.state import ChatMessage
from swe_brew.agent.tools import ToolResult, execute_action, extract_diff_from_submit
from swe_brew.benchmark.container import TestSandbox
from swe_brew.benchmark.generate import GenerationResult
from swe_brew.benchmark.models import SplitPatches, SweBrewCase
from swe_brew.benchmark.runtime import runtime_spec
from swe_brew.benchmark.surefire import parse_surefire_reports

__all__ = ["AgentGenerator", "TestGenAgent", "LoopResult", "attempt_hint_for"]


_DIFF_NEW_PATH_RE = re.compile(r"^\+\+\+\s+b/(\S+)", re.MULTILINE)
_PL_ARG_RE = re.compile(r"(-pl\s+)(\S+)")
_CLASS_DECL_RE = re.compile(
    r"^\+\s*(?:public\s+|final\s+|abstract\s+)*class\s+(\w+)",
    re.MULTILINE,
)
_METHOD_DECL_RE = re.compile(
    r"^\+\s*"
    r"(?:@\w+(?:\([^)]*\))?\s+)*"
    r"(?:(?:public|protected|private|static|final|synchronized|abstract|default|native)\s+)*"
    r"(?:<[^>]+>\s+)?"
    r"(?:[A-Za-z_][\w<>\[\],.?]*)\s+"
    r"(\w+)\s*\(",
    re.MULTILINE,
)
_NON_METHOD_NAMES = {
    "class",
    "interface",
    "enum",
    "record",
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "new",
    "assert",
    "synchronized",
}

_REFLECTION_MARKER_RE = re.compile(
    r"\bClass\.forName\b|\bgetDeclaredMethod\b|\bgetDeclaredField\b"
    r"|\bEnum\.valueOf\b|\bModifier\."
)


def _code_patch_is_deletion_only(code_patch: str) -> bool:
    """True when the gold code patch removes lines but adds no lines.

    Deletion/removal contracts are the one legitimate reflection case: the
    change contract is the absence of the old API, so ``Class.forName`` with
    ``ClassNotFoundException`` is the correct oracle.
    """
    added = len(re.findall(r"^\+(?!\+\+).*$", code_patch or "", re.MULTILINE))
    removed = len(re.findall(r"^-(?!--).*$", code_patch or "", re.MULTILINE))
    return added == 0 and removed > 0


def _reflection_marker_count(diff_text: str) -> int:
    return len(_REFLECTION_MARKER_RE.findall(diff_text))


def _candidate_touches_existing_test(diff_text: str) -> bool:
    """True when the candidate modifies an existing test file.

    The contract is additive: candidate tests must be new files. Appending to
    an existing test class still mutates pre-existing source, so the simplest
    safe rule is to reject any diff block whose source path is not
    ``/dev/null`` and whose target is under ``src/test``.
    """
    for block in diff_text.split("diff --git "):
        if not block.strip():
            continue
        lines = block.splitlines()
        header = lines[0]
        match = re.match(r"a/(\S+)\s+b/(\S+)", header)
        if not match:
            continue
        _, target = match.groups()
        if target == "/dev/null":
            continue
        if "--- /dev/null" in block:
            continue
        if "/src/test/" in f"/{target.lstrip('/')}":
            return True
    return False


def _candidate_test_spec_from_diff(diff_text: str) -> dict[str, set[str]]:
    """Return ``{simple_class_name: {method_name}}`` for a test diff.

    The parsing is deliberately conservative: it derives candidates from
    ``+++ b/<module>/.../<Name>Test.java`` path stems, added class
    declarations, and added method declarations. If only a path stem is
    found, the class entry remains an empty method set, which is treated
    as "any test in that class" by the execution check.
    """
    if not diff_text:
        return {}

    classes: dict[str, set[str]] = {}
    for path in _DIFF_NEW_PATH_RE.findall(diff_text):
        stem = path.rsplit("/", 1)[-1]
        if stem.endswith(".java"):
            classes.setdefault(stem[:-5], set())

    for match in _CLASS_DECL_RE.finditer(diff_text):
        name = match.group(1)
        if name not in classes:
            # A class declaration without a matching new-file path is not
            # trustworthy for test-id matching; still record it as a
            # class-level candidate when it is otherwise absent.
            classes.setdefault(name, set())

    method_names: set[str] = set()
    for match in _METHOD_DECL_RE.finditer(diff_text):
        line = match.group(0)
        if re.search(r"\b(?:class|interface|enum|record)\s+\w+", line):
            continue
        name = match.group(1)
        if name not in _NON_METHOD_NAMES:
            method_names.add(name)

    if len(classes) == 1 and method_names:
        only = next(iter(classes))
        # A constructor has the same name as the class and is not a test.
        method_names.discard(only)
        classes[only].update(method_names)

    return classes


def _candidate_test_ids_from_diff(diff_text: str) -> set[str]:
    """Return test-id prefixes used by the pre-submit execution check."""
    spec = _candidate_test_spec_from_diff(diff_text)
    ids: set[str] = set()
    for class_name, methods in spec.items():
        if methods:
            ids.update(f"{class_name}#{method}" for method in methods)
        else:
            ids.add(class_name)
    return ids


def _test_id_matches_candidate(test_id: str, candidate_id: str) -> bool:
    """Match a parsed Surefire test id against a candidate spec entry."""
    if "#" in candidate_id:
        class_part, method_part = candidate_id.rsplit("#", 1)
    else:
        class_part, method_part = candidate_id, None

    suite, method = test_id.rsplit("#", 1)
    suite_matches = (
        suite == class_part
        or suite.endswith(f".{class_part}")
        or suite.endswith(f"${class_part}")
    )
    if method_part is None:
        return suite_matches
    return suite_matches and method == method_part


def _augment_preflight_test_command(cmd: str, diff_text: str) -> str:
    """Build a command that can execute candidate tests in-session.

    This mirrors the verify-side scope augmentation but intentionally adds
    ``-am`` when the candidate diff introduces a module outside organize's
    submitted ``-pl`` list. A preflight is a build-and-run check, so it may
    be more demanding than the final verify command.
    """
    cmd = (cmd or runtime_spec().test_command).strip()
    candidate_modules = {
        path.split("/", 1)[0]
        for path in _DIFF_NEW_PATH_RE.findall(diff_text or "")
        if "/" in path
    }
    if not candidate_modules or not cmd:
        return cmd

    pl_match = _PL_ARG_RE.search(cmd)
    if pl_match is None:
        # Reactor-wide command already attempts every module.
        return cmd

    existing = {
        token.strip()
        for token in pl_match.group(2).split(",")
        if token.strip()
    }
    missing = sorted(candidate_modules - existing)
    if not missing:
        return cmd

    union = sorted(existing | candidate_modules)
    new_pl = f"{pl_match.group(1)}{','.join(union)}"
    new_cmd = cmd[:pl_match.start()] + new_pl + cmd[pl_match.end():]
    if not re.search(r"(?:^|\s)-am(?:\s|$)", new_cmd):
        insert_at = pl_match.start() + len(new_pl)
        new_cmd = new_cmd[:insert_at] + " -am" + new_cmd[insert_at:]
    return new_cmd


# ─── test-gen agent (thin subclass of AgentLoop) ─────────────────────────────


@dataclass
class TestGenAgent(AgentLoop):
    """Test-generation ReAct loop.

    Read-only tool set, ``src/test/java/**``-restricted apply_diff.
    Emits a unified diff via ``<submit>``.

    ``run(case=..., split=...)`` → LoopResult. Caller (enhance.py
    orchestrator) is responsible for reading ``submit_payload`` as
    the candidate test-patch string.
    """

    role: str = field(default="test_gen")

    # Prompt-slot configuration; defaults maximise F2P yield.
    #   ``max_tests``: None = uncapped; int = "at most N" clause.
    #   ``max_prd_chars`` / ``max_patch_chars`` / ``max_changed_files_list``:
    #     None = feed the full string / list.
    max_tests: int | None = None
    max_prd_chars: int | None = None
    max_patch_chars: int | None = None
    max_changed_files_list: int | None = None

    # Organize → test-gen plumbing. ``sibling_test_files`` is the
    # validated local test inventory plus the highest-evidence module's
    # tests, rendered into the ``{siblings_block}`` slot so the agent
    # knows what to ``<bash>cat`` for style-match.
    sibling_test_files: list[str] | None = None
    validated_modules: list[str] = field(default_factory=list)
    organize_notes: list[str] = field(default_factory=list)
    organize_test_command: str = ""

    # Pass@k attempt diversity. ``attempt_index=1`` is baseline; higher
    # values pull a rotating diversity nudge from ``ATTEMPT_HINTS``.
    # ``attempt_hint`` is the pre-rendered hint string.
    attempt_index: int = 1
    attempt_hint: str = ""
    split: SplitPatches | None = None

    def build_prefix(
        self,
        case: SweBrewCase | None = None,
        split: SplitPatches | None = None,
        **_ignored: Any,
    ) -> list[ChatMessage]:
        if case is None or split is None:
            raise ValueError("TestGenAgent.build_prefix requires case + split")
        prd = case.prd_text if self.max_prd_chars is None else case.prd_text[: self.max_prd_chars]
        code = split.code_patch if self.max_patch_chars is None else split.code_patch[: self.max_patch_chars]
        if self.max_changed_files_list is None:
            changed_list = "\n".join(f"- {p}" for p in case.changed_files)
        else:
            changed_list = "\n".join(f"- {p}" for p in case.changed_files[: self.max_changed_files_list])
        base_image = runtime_spec().base_image
        validated_modules_block = render_validated_modules_block(
            self.validated_modules,
        )
        organize_notes_block = render_organize_notes_block(self.organize_notes)
        modules_context = "\n\n".join(
            part
            for part in (validated_modules_block, organize_notes_block)
            if part
        )
        return [
            ChatMessage(role="system", content=safe_render(SYSTEM_MESSAGE, base_image=base_image)),
            ChatMessage(
                role="user",
                content=(
                    safe_render(REACT_PROMPT, tools=TOOL_DESCRIPTIONS)
                    + "\n"
                    + safe_render(
                        TASK_MESSAGE_TEMPLATE,
                        code_patch=code,
                        changed_count=len(case.changed_files),
                        changed_files_list=changed_list,
                        prd=prd,
                        cap_clause=render_cap_clause(self.max_tests),
                        siblings_block=render_siblings_block(self.sibling_test_files),
                        validated_modules_block=modules_context,
                        attempt_block=self.attempt_hint or "",
                    )
                ),
            ),
        ]

    def parse_action(self, response: str) -> TestGenAction | None:
        return TestGenActionParser().parse(response)

    def execute_action(self, action: TestGenAction) -> ToolResult:
        # Read-only allowlist + src/test/java-only apply_diff.
        return execute_action(self.sandbox, action)

    def validate_submit_payload(self, payload: str, state: Any) -> tuple[bool, str]:
        """Refuse a candidate until at least one of its tests actually executes.

        The final verify stage remains the precision gate. This check only
        closes the zero-execution failure class where the agent submitted a
        candidate into a module that could not compile or run at all.
        """
        diff = (payload or "").strip()
        if not diff:
            return False, "submitted diff is empty"

        candidate_ids = _candidate_test_ids_from_diff(diff)
        if not candidate_ids:
            return False, "could not identify any candidate test class from the diff"

        if _candidate_touches_existing_test(diff):
            return (
                False,
                "candidate modifies an existing test file. Pre-existing tests "
                "must remain untouched; add your tests as a new file instead.",
            )

        # Behavioral style signal. Non-deletion hunks should prefer direct
        # construction and concrete assertions, but reflection-only tests are
        # still valid F2P candidates. Emit a warning for downstream curation;
        # the executable six-way verify remains the precision gate.
        if (
            self.split is not None
            and not _code_patch_is_deletion_only(self.split.code_patch)
        ):
            reflection_count = _reflection_marker_count(diff)
            direct_instances = len(
                re.findall(r"\bnew\s+[A-Z][A-Za-z0-9_]*\s*\(", diff)
            )
            if reflection_count and direct_instances == 0:
                self._emit(
                    "agent.submit.reflection_warning",
                    reflection_markers=reflection_count,
                    direct_instances=direct_instances,
                )

        command = _augment_preflight_test_command(
            self.organize_test_command,
            diff,
        )
        self._emit(
            "agent.submit.preflight.begin",
            candidate_classes=len(
                {candidate_id.rsplit("#", 1)[0] for candidate_id in candidate_ids}
            ),
            test_command=command,
        )
        try:
            # Stale target/classes and target/test-classes can make an
            # uncompileable candidate appear executed. Verify resets and
            # ``git clean -fdx`` between legs, so preflight must remove
            # build artefacts before running too. This is confined to the
            # bind-mounted /testbed tree.
            self.sandbox.exec(
                "find /testbed -type d -name target -prune "
                "-exec rm -rf {} +",
                timeout=600,
            )
            self.sandbox.clear_surefire_reports()
            self.sandbox.exec(
                command,
                timeout=runtime_spec().command_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            reason = f"preflight test command failed: {type(exc).__name__}: {exc!s}"
            self._emit("agent.submit.preflight.end", ok=False, reason=reason[:400])
            return False, reason

        outcomes = parse_surefire_reports(self.sandbox.snapshot.work_dir)
        executed_ids = {outcome.test_id for outcome in outcomes}
        candidate_executed = any(
            _test_id_matches_candidate(test_id, candidate_id)
            for test_id in executed_ids
            for candidate_id in candidate_ids
        )
        if candidate_executed:
            self._emit("agent.submit.preflight.end", ok=True)
            return True, ""
        self._emit(
            "agent.submit.preflight.end",
            ok=False,
            reason="candidate_tests_not_executed",
        )
        return (
            False,
            "none of the candidate tests appear in Surefire reports; "
            "the candidate module likely did not compile or was outside "
            "the executed test scope. Fix the module/compilation issue or "
            "move the tests to a validated module, then resubmit.",
        )

    def extract_submit_payload(self, action: TestGenAction) -> str:
        return extract_diff_from_submit(action.args)

    def parser_drift_help(self) -> str:
        return (
            "No valid action tag found in your response. Reply with exactly "
            "one of: <apply_diff>…</apply_diff>, <mvn_test_compile></mvn_test_compile>, "
            "<mvn_test></mvn_test>, <read_surefire>…</read_surefire>, "
            "<git_grep>…</git_grep>, <bash>…</bash>, <search>…</search>, "
            "or <submit>…</submit>."
        )

    def trial_retry_nudge(self, trial: int) -> str:
        return (
            f"Trial {trial}/{self.config.max_trials}. Your previous submission "
            "did not verify (either no diff, apply failed, compile failed, "
            "or no tests were fail-to-pass). Please try again — start with "
            "<git_grep> to confirm the symbols you plan to reference actually "
            "exist, then <apply_diff>, then <mvn_test_compile>."
        )


# ─── AgentGenerator facade ──────────────────────────────────────────────────


@dataclass
class AgentGenerator:
    """Compatibility adapter for :mod:`benchmark.enhance` and existing tests.

    Thin wrapper around :class:`TestGenAgent`. Public API:
    ``generate(case, split) → GenerationResult`` and
    ``run(case, split) → LoopResult``.

    Defaults are tuned for test-gen against large closed-source
    Java codebases: 6 trials × 40 steps per trial, no cap on emitted tests. The
    subprocess tool timeout (``_TOOL_TIMEOUT_SECONDS = 3600`` in
    :mod:`tools`) covers a slow ``mvn test`` phase; the per-trial
    wall-clock watchdog in :mod:`framework` is the outer bound.

    Setup + organize agents keep their AgentConfig default
    ``max_trials=10`` — a slow Maven mirror may need repeated
    retries with tweaked ``settings.xml`` — though most cases submit
    on the first trial.
    """

    sandbox: TestSandbox
    max_trials: int = 6
    max_steps_per_trial: int | None = 50
    max_tests: int | None = None
    max_prd_chars: int | None = None
    max_patch_chars: int | None = None
    max_changed_files_list: int | None = None
    sibling_test_files: list[str] | None = None
    validated_modules: list[str] = field(default_factory=list)
    organize_notes: list[str] = field(default_factory=list)
    organize_test_command: str = ""
    attempt_index: int = 1
    telemetry: TelemetryLog | None = None
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    fallback_model: str = ""
    reasoning_effort: str = ""
    verbosity: str = ""

    def _make_config(self) -> AgentConfig:
        return AgentConfig(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            fallback_model=self.fallback_model,
            reasoning_effort=self.reasoning_effort,
            verbosity=self.verbosity,
            max_trials=self.max_trials,
            max_steps_per_trial=self.max_steps_per_trial,
        )

    def generate(self, case: SweBrewCase, split: SplitPatches) -> GenerationResult:
        loop_result = self.run(case, split)
        config = self._make_config()
        return GenerationResult(
            candidate_patch=loop_result.submit_payload,
            is_stub=False,
            model=config.model,
            raw_response=TestGenAgent(
                sandbox=self.sandbox, case_id=case.case_id,
            ).render_transcript(loop_result.state),
        )

    def run(self, case: SweBrewCase, split: SplitPatches) -> LoopResult:
        agent = TestGenAgent(
            sandbox=self.sandbox,
            case_id=case.case_id,
            telemetry=self.telemetry,
            config=self._make_config(),
            max_tests=self.max_tests,
            max_prd_chars=self.max_prd_chars,
            max_patch_chars=self.max_patch_chars,
            max_changed_files_list=self.max_changed_files_list,
            sibling_test_files=self.sibling_test_files,
            validated_modules=self.validated_modules,
            organize_notes=self.organize_notes,
            organize_test_command=self.organize_test_command,
            attempt_index=self.attempt_index,
            attempt_hint=attempt_hint_for(self.attempt_index),
            split=split,
        )
        return agent.run(case=case, split=split)
