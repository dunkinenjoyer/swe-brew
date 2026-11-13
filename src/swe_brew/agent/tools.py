"""Tool execution for the test-gen agent.

The five tools mirror the shape of ``launch/agent/setup/setup.py``'s
``observation_for_setup_action``: parse the action, dispatch to the
sandbox, return a ``ToolResult`` with the observation string that gets
threaded back into the LLM's next turn.

Narrower than RepoLaunch's "run any bash command" tool by design — we
know the domain (Maven + JUnit + git-grep on a Java tree), so we can
provide exactly the tools that matter and reject drift.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass

import httpx

from swe_brew.agent.action_parser import TestGenAction
from swe_brew.benchmark.container import EXEC_TIMEOUT_EXIT_CODE, CommandResult, TestSandbox
from swe_brew.benchmark.runtime import runtime_spec

__all__ = [
    "ToolResult",
    "execute_action",
    "execute_action_write_allowed",
    "extract_diff_from_submit",
]


@dataclass
class ToolResult:
    """Observation for one action — what we hand back to the LLM.

    ``truncated_content`` is the observation text (bounded to keep the
    context window sane). ``exit_code`` mirrors CommandResult's, or 0/1
    for tools that don't shell out. ``is_stop`` is True only for
    ``<submit>`` — signals the loop driver to break.
    """

    truncated_content: str
    exit_code: int = 0
    is_stop: bool = False
    elapsed_s: float = 0.0


# Maximum characters of Maven / grep output handed back per turn.
# 200 KB ≈ 50 000 tokens is generous enough to preserve the useful
# middle of a multi-module compile error trail — the agent's ability
# to correlate failures across cycles depends on seeing the full output.
_MAX_OBS_CHARS = 200000

# Single per-command tool timeout. One hour is enough for the slowest
# legitimate command — ``mvn install`` on a big multi-module project
# when the dependency mirror is cold — but bounded enough that a hung docker
# exec doesn't wedge the daemon. The real budget is the per-trial
# wall-clock watchdog in :mod:`framework`.
_TOOL_TIMEOUT_SECONDS = runtime_spec().command_timeout_seconds
_BASH_PERTEST_TIMEOUT_SECONDS = 180


def _tool_timeout() -> int:
    """Return the current runtime command timeout.

    Runtime callers use this instead of the module constant so a later
    ``load_dotenv`` or fleet environment override is honoured.
    """
    return runtime_spec().command_timeout_seconds


def _truncate(text: str, cap: int = _MAX_OBS_CHARS) -> str:
    """Head + tail slice for very long outputs — mid-cut hides the useful
    tail (which is what javac / Surefire put the actual failure in)."""
    if len(text) <= cap:
        return text
    head_n = cap // 3
    tail_n = cap - head_n
    return (
        text[:head_n]
        + f"\n\n… [{len(text) - cap} chars elided] …\n\n"
        + text[-tail_n:]
    )


# ─── individual tool implementations ─────────────────────────────────────────


def _apply_diff(sandbox: TestSandbox, args: str) -> ToolResult:
    """Apply a unified diff to /testbed. Rejects diffs that touch non-test
    paths — we're writing tests, not modifying the solution."""
    diff = _strip_fenced_diff(args)
    if not diff.strip():
        return ToolResult(
            truncated_content="No diff payload found. Wrap the diff in ```diff … ``` inside <apply_diff>.",
            exit_code=1,
        )

    marker = runtime_spec().test_path_marker
    # Cheap safety check: refuse any hunk that writes outside the test source root.
    bad_paths = []
    for m in re.finditer(r"^diff --git a/(\S+) b/(\S+)", diff, re.MULTILINE):
        a, b = m.group(1), m.group(2)
        for p in (a, b):
            if p != "/dev/null" and marker not in f"/{p.lstrip('/')}":
                bad_paths.append(p)
    if bad_paths:
        return ToolResult(
            truncated_content=(
                f"Rejected: diff touches non-test paths — you may only add / modify files under "
                f"{marker}. Offending: {sorted(set(bad_paths))[:5]}"
            ),
            exit_code=1,
        )

    t0 = time.perf_counter()
    r = sandbox.apply_patch(diff)
    elapsed = time.perf_counter() - t0
    obs = f"exit={r.exit_code}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}".strip()
    return ToolResult(
        truncated_content=_truncate(obs),
        exit_code=r.exit_code,
        elapsed_s=elapsed,
    )


def _mvn_test_compile(sandbox: TestSandbox, _args: str) -> ToolResult:
    """Run ``mvn -B -q test-compile`` to catch javac errors fast."""
    cmd = runtime_spec().compile_command
    t0 = time.perf_counter()
    timeout = (
        min(_tool_timeout(), _BASH_PERTEST_TIMEOUT_SECONDS)
        if "-Dtest=" in cmd
        else _tool_timeout()
    )
    r = sandbox.exec(cmd, timeout=timeout)
    elapsed = time.perf_counter() - t0
    if r.timed_out:
        return ToolResult(
            truncated_content=(
                f"$ {cmd}\nexit=124\nTIMEOUT after {timeout}s. "
                "Do not retry this exact command unchanged. Narrow the "
                "module list and remove reactor flags unless Maven proves "
                "they are required."
            ),
            exit_code=EXEC_TIMEOUT_EXIT_CODE,
            elapsed_s=elapsed,
        )
    compile_failed = r.exit_code != 0 or _has_maven_errors(r)
    summary = _summarize_javac(r, force_failure=compile_failed)
    return ToolResult(
        truncated_content=_truncate(summary),
        exit_code=1 if compile_failed else 0,
        elapsed_s=elapsed,
    )


def _mvn_test(sandbox: TestSandbox, _args: str) -> ToolResult:
    """Run the full mvn test invocation. Summarize Surefire output into
    per-class pass/fail counts to keep the observation compact."""
    from swe_brew.benchmark.surefire import parse_surefire_reports
    t0 = time.perf_counter()
    sandbox.clear_surefire_reports()
    r = sandbox.exec(runtime_spec().test_command, timeout=_tool_timeout())
    elapsed = time.perf_counter() - t0
    if r.timed_out:
        return ToolResult(
            truncated_content=(
                f"$ {runtime_spec().test_command}\nexit=124\nTIMEOUT after "
                f"{_tool_timeout()}s. Do not retry unchanged. Narrow "
                "the scope and investigate which test is hanging."
            ),
            exit_code=EXEC_TIMEOUT_EXIT_CODE,
            elapsed_s=elapsed,
        )

    outcomes = parse_surefire_reports(sandbox.snapshot.work_dir)
    counts: dict[str, dict[str, int]] = {}
    for o in outcomes:
        cls = o.test_id.rsplit("#", 1)[0]
        d = counts.setdefault(cls, {"pass": 0, "fail": 0, "error": 0, "skip": 0, "missing": 0})
        d[o.status.value] = d.get(o.status.value, 0) + 1

    if not counts:
        obs = (
            f"mvn exit={r.exit_code}. Surefire wrote no reports.\n"
            f"stdout tail:\n{r.stdout[-2000:]}"
        )
        compile_failed = r.exit_code != 0 or _has_maven_errors(r)
        return ToolResult(
            truncated_content=_truncate(obs),
            exit_code=1 if compile_failed else 0,
            elapsed_s=elapsed,
        )
    else:
        lines = [f"mvn exit={r.exit_code}. Surefire found {len(counts)} test classes:"]
        for cls in sorted(counts):
            c = counts[cls]
            lines.append(
                f"  {cls}: pass={c['pass']} fail={c['fail']} "
                f"error={c['error']} skip={c['skip']}",
            )
        obs = "\n".join(lines)
    return ToolResult(truncated_content=_truncate(obs), exit_code=0, elapsed_s=elapsed)


def _read_surefire(sandbox: TestSandbox, args: str) -> ToolResult:
    """Dump one Surefire XML by relative path."""
    path = args.strip()
    if not path:
        return ToolResult(truncated_content="Empty path.", exit_code=1)
    if ".." in path.split("/") or path.startswith("/"):
        return ToolResult(truncated_content="Path must be repo-relative.", exit_code=1)
    target = sandbox.snapshot.work_dir / path
    if not target.exists():
        return ToolResult(truncated_content=f"File not found: {path}", exit_code=1)
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return ToolResult(truncated_content=f"read failed: {e}", exit_code=1)
    return ToolResult(truncated_content=_truncate(text), exit_code=0)


# ─── bash exploration (allowlisted) ──────────────────────────────────────────

# Read-only exploration verbs — narrow set that lets the agent probe the tree
# without straying into shell hazards. Loosely modelled on RepoLaunch's
# `<command>bash command</command>` tool (launch/agent/setup/setup.py:36-51)
# but with an allowlist because we don't want the agent installing packages
# or mutating state.
_BASH_ALLOWLIST: frozenset[str] = frozenset({
    "ls", "cat", "head", "tail", "wc", "find", "tree",
    "grep", "rg",
    # Text-processing helpers for chained read-only probes. All strict
    # stream transformers; none can create files without ``>``
    # redirection (banned below). ``xargs`` is the pipe-glue that makes
    # ``find | xargs grep`` work.
    "sort", "uniq", "cut", "tr", "xargs",
    "java", "javac", "mvn",
    "git",
    "echo", "pwd", "which", "type", "file", "stat", "date",
    "true", "false",
    "test", "["      # shell test builtins (harmless)
})

# Substrings that must not appear anywhere in the command string. Pipes
# (``|``), composition (``&&`` / ``||`` / ``;``), subshells (``$(...)``,
# backticks), and input redirection (``<``) are all ALLOWED — the
# canonical exploration idiom ``find … | xargs grep … | head -5``
# depends on them. Write redirection (``>``) and state-mutating verbs
# stay banned as belt-and-suspenders behind the first-token allowlist.
_BASH_FORBIDDEN_SUBSTR: tuple[str, ...] = (
    ">",                                     # output redirection (write)
    " rm ", " rm\t", " sudo ", " sudo\t",
    "curl ", "wget ", "apt ", "apt-get ", "pip ", "npm ", "yarn ",
    "chmod ", "chown ", "kill ", "pkill ",
    " mv ", " mv\t", " cp ", " cp\t",
    "&\n", "& ",                             # backgrounding
)

# Dangerous mvn subverbs that mutate the local Maven repo or install artifacts.
_MVN_FORBIDDEN_SUBVERBS: tuple[str, ...] = (
    "install", "deploy", "clean", "wrapper:", "plugin:"
)


def _bash(sandbox: TestSandbox, args: str) -> ToolResult:
    """Read-only bash exploration with composition allowed.

    Rules:
      * First token must be in :data:`_BASH_ALLOWLIST`.
      * Every token that follows a composition operator (``;``, ``&&``,
        ``||``, ``|``) must also be allowlisted, so ``find | xargs grep``
        is fine but ``ls; rm -rf /`` is refused.
      * :data:`_BASH_FORBIDDEN_SUBSTR` blocks write redirection (``>``)
        and state-mutating verbs (``rm``, ``mv``, ``cp``, ``chmod``,
        ``curl``, ``wget``, package managers, backgrounding).
      * For ``mvn``, subverb blacklist bans install/deploy/clean.
    """
    cmd = args.strip()
    if not cmd:
        return ToolResult(truncated_content="Empty command.", exit_code=1)
    if len(cmd) > 4000:
        return ToolResult(truncated_content="Command too long (max 4000 chars).", exit_code=1)

    tokens = cmd.split()
    first = tokens[0]
    if first not in _BASH_ALLOWLIST:
        return ToolResult(
            truncated_content=(
                f"Refused: '{first}' not on allowlist. Allowed verbs: "
                f"{sorted(_BASH_ALLOWLIST)}"
            ),
            exit_code=1,
        )
    # With composition allowed, every verb-position token (after a
    # separator) must also be allowlisted. Cheap parse — split on the
    # composition operators and re-check each segment's leading token.
    # Not shell-perfect; the substring ban below covers what this misses.
    import re
    segments = re.split(r"(?:\|\||&&|\||;)", cmd)
    for seg in segments[1:]:                    # segments[0] already checked
        seg_tokens = seg.strip().split()
        if not seg_tokens:
            continue
        verb = seg_tokens[0]
        if verb not in _BASH_ALLOWLIST:
            return ToolResult(
                truncated_content=(
                    f"Refused: chained verb '{verb}' not on allowlist. "
                    f"Every verb-position token in a composed command must "
                    f"be in the allowlist. Allowed: {sorted(_BASH_ALLOWLIST)}"
                ),
                exit_code=1,
            )
    for bad in _BASH_FORBIDDEN_SUBSTR:
        if bad in cmd:
            return ToolResult(
                truncated_content=(
                    f"Refused: contains forbidden operator/verb '{bad.strip()}'. "
                    "This tool is READ-ONLY exploration. Pipes / composition / "
                    "subshells are allowed; only WRITE operations (``>``, "
                    "``rm``, ``mv``, ``chmod``, ``curl``, package managers) are "
                    "banned. Use <apply_diff> for state changes."
                ),
                exit_code=1,
            )
    if first == "mvn":
        for bad in _MVN_FORBIDDEN_SUBVERBS:
            if bad in cmd:
                return ToolResult(
                    truncated_content=(
                        f"Refused: 'mvn {bad}' would mutate local state. "
                        "Allowed mvn subverbs: help:*, dependency:*, "
                        "test-compile, versions:*, validate. "
                        "Use <mvn_test> for the full test phase."
                    ),
                    exit_code=1,
                )

    t0 = time.perf_counter()
    # Bounded by :data:`_TOOL_TIMEOUT_SECONDS`; a slow ``find`` over a
    # big multi-module tree or a cold ``mvn help:effective-pom`` can
    # occasionally take minutes.
    timeout = (
        min(_tool_timeout(), _BASH_PERTEST_TIMEOUT_SECONDS)
        if "-Dtest=" in cmd
        else _tool_timeout()
    )
    r = sandbox.exec(cmd, timeout=timeout)
    elapsed = time.perf_counter() - t0
    obs = f"$ {cmd}\nexit={r.exit_code}\n{r.stdout}".rstrip()
    if r.stderr:
        obs += f"\n[stderr]\n{r.stderr}"
    return ToolResult(
        truncated_content=_truncate(obs),
        exit_code=r.exit_code,
        elapsed_s=elapsed,
    )


# ─── Tavily web search (public knowledge only) ───────────────────────────────


def _search(sandbox: TestSandbox, args: str) -> ToolResult:  # noqa: ARG001
    """Tavily web search.

    REPOSITORY-SPECIFIC DISCIPLINE (enforced in prompt, echoed here):
    - Web results have zero knowledge of this repository's classes,
      packages, or configuration. This tool exists ONLY to look up
      public knowledge — JUnit 4 semantics, Mockito API, unified diff
      format, Java stdlib. Do NOT trust web results about symbols
      that need to match our specific patch.

    Fails cleanly if TAVILY_API_KEY is not set — the agent should
    continue using <git_grep> and <bash> instead.
    """
    q = args.strip()
    if not q:
        return ToolResult(truncated_content="Empty query.", exit_code=1)
    if len(q) > 300:
        return ToolResult(truncated_content="Query too long (max 300 chars).", exit_code=1)

    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return ToolResult(
            truncated_content=(
                "Search unavailable: TAVILY_API_KEY not set in environment. "
                "Continue with <git_grep> and <bash> for in-tree exploration."
            ),
            exit_code=1,
        )

    payload = {
        "api_key": api_key,
        "query": q,
        "search_depth": "basic",           # 'basic' is faster + cheaper than 'advanced'
        "max_results": 5,
        "include_answer": True,            # Tavily-summarized top answer
        "include_domains": [],
        "exclude_domains": [],
    }
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.post("https://api.tavily.com/search", json=payload)
    except httpx.HTTPError as e:
        return ToolResult(
            truncated_content=f"Search HTTP error: {e!s}. Continue without web results.",
            exit_code=1,
        )
    elapsed = time.perf_counter() - t0
    if r.status_code >= 400:
        return ToolResult(
            truncated_content=f"Tavily returned HTTP {r.status_code}: {r.text[:300]}",
            exit_code=1,
            elapsed_s=elapsed,
        )
    try:
        data = r.json()
    except json.JSONDecodeError:
        return ToolResult(truncated_content=f"Tavily returned non-JSON: {r.text[:300]}", exit_code=1)

    lines = ["WEB SEARCH RESULT — reminder: this repository's source is not indexed; use only for public knowledge."]
    if data.get("answer"):
        lines.append(f"\nSummary: {data['answer']}")
    for i, res in enumerate(data.get("results", [])[:5], start=1):
        title = res.get("title", "")[:150]
        url = res.get("url", "")
        content = res.get("content", "")[:400]
        lines.append(f"\n[{i}] {title}\n    {url}\n    {content}")
    return ToolResult(
        truncated_content=_truncate("\n".join(lines)),
        exit_code=0,
        elapsed_s=elapsed,
    )


def _git_grep(sandbox: TestSandbox, args: str) -> ToolResult:
    """git-grep the checked-out worktree for a symbol / literal.

    Queries up to 2000 chars accommodate fully-qualified generic type
    signatures. ``--max-count=200`` per file is generous enough for the
    hundreds of call sites a common symbol has in an Java
    tree. Timeout uses :data:`_TOOL_TIMEOUT_SECONDS`; a common-literal
    grep over a huge tree occasionally takes minutes.
    """
    q = args.strip()
    if not q:
        return ToolResult(truncated_content="Empty query.", exit_code=1)
    if len(q) > 2000:
        return ToolResult(truncated_content="Query too long (max 2000 chars).", exit_code=1)
    try:
        r = subprocess.run(
            [
                "git", "-C", str(sandbox.snapshot.work_dir),
                "grep", "-n", "--fixed-strings",
                "--max-count=200",
                "--",
                q,
            ],
            capture_output=True, text=True, timeout=_tool_timeout(),
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            truncated_content=(
                f"git-grep timed out after {_tool_timeout()}s. "
                "The literal is likely too common — narrow it (e.g. add "
                "package qualifier or method signature) and try again."
            ),
            exit_code=1,
        )
    out = r.stdout
    if not out.strip():
        obs = f"No matches for {q!r} in the pre-patch tree. This symbol likely does not exist — do not reference it."
    else:
        obs = f"Matches for {q!r}:\n{out}"
    return ToolResult(truncated_content=_truncate(obs), exit_code=r.returncode)


# ─── dispatch ────────────────────────────────────────────────────────────────


def execute_action(sandbox: TestSandbox, action: TestGenAction) -> ToolResult:
    """Route a parsed action to its tool. Terminal ``submit`` is signalled
    with ``is_stop=True`` — the loop driver harvests
    ``action.args`` as the final diff."""
    if action.kind == "submit":
        diff = extract_diff_from_submit(action.args)
        return ToolResult(
            truncated_content=f"submitted diff ({len(diff)} chars)",
            exit_code=0,
            is_stop=True,
        )
    if action.kind == "apply_diff":
        return _apply_diff(sandbox, action.args)
    if action.kind == "mvn_test_compile":
        return _mvn_test_compile(sandbox, action.args)
    if action.kind == "mvn_test":
        return _mvn_test(sandbox, action.args)
    if action.kind == "read_surefire":
        return _read_surefire(sandbox, action.args)
    if action.kind == "git_grep":
        return _git_grep(sandbox, action.args)
    if action.kind == "bash":
        return _bash(sandbox, action.args)
    if action.kind == "search":
        return _search(sandbox, action.args)
    if action.kind == "mvn_install":
        return ToolResult(
            truncated_content=(
                "Refused: <mvn_install> is a state-mutating action, not "
                "available to the test-gen agent. Use <mvn_test_compile> "
                "or <mvn_test> instead."
            ),
            exit_code=1,
        )
    return ToolResult(
        truncated_content=f"Unknown action kind: {action.kind!r}",
        exit_code=1,
    )


# ─── write-allowed dispatch (setup + organize agents) ───────────────────────


def execute_action_write_allowed(sandbox: TestSandbox, action: TestGenAction) -> ToolResult:
    """Dispatch for setup + organize agents. Unlike ``execute_action``:

    - ``<bash>`` runs with NO allowlist and NO shell-operator restrictions
      (matches RepoLaunch's ``<command>bash command</command>``).
    - ``<apply_diff>`` accepts diffs touching ANY path, not just
      ``src/test/java/**``.
    - ``<mvn_install>`` is available.
    """
    if action.kind == "submit":
        return ToolResult(
            truncated_content=f"submitted (payload {len(action.args)} chars)",
            exit_code=0,
            is_stop=True,
        )
    if action.kind == "apply_diff":
        return _apply_diff_write_allowed(sandbox, action.args)
    if action.kind == "mvn_test_compile":
        return _mvn_test_compile(sandbox, action.args)
    if action.kind == "mvn_test":
        return _mvn_test(sandbox, action.args)
    if action.kind == "mvn_install":
        return _mvn_install(sandbox, action.args)
    if action.kind == "read_surefire":
        return _read_surefire(sandbox, action.args)
    if action.kind == "git_grep":
        return _git_grep(sandbox, action.args)
    if action.kind == "bash":
        return _bash_write_allowed(sandbox, action.args)
    if action.kind == "search":
        return _search(sandbox, action.args)
    return ToolResult(
        truncated_content=f"Unknown action kind: {action.kind!r}",
        exit_code=1,
    )


def _apply_diff_write_allowed(sandbox: TestSandbox, args: str) -> ToolResult:
    """apply_diff for setup/organize — accepts diffs to any path."""
    diff = _strip_fenced_diff(args)
    if not diff.strip():
        return ToolResult(
            truncated_content="No diff payload found. Wrap the diff in ```diff … ``` inside <apply_diff>.",
            exit_code=1,
        )
    t0 = time.perf_counter()
    r = sandbox.apply_patch(diff)
    elapsed = time.perf_counter() - t0
    obs = f"exit={r.exit_code}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}".strip()
    return ToolResult(
        truncated_content=_truncate(obs),
        exit_code=r.exit_code,
        elapsed_s=elapsed,
    )


def _bash_write_allowed(sandbox: TestSandbox, args: str) -> ToolResult:
    """bash for setup/organize — no allowlist, no shell-operator restrictions.

    Matches RepoLaunch's <command> tool. The agent is trusted to modify
    state; that's the whole point of the setup + organize roles.
    """
    cmd = args.strip()
    if not cmd:
        return ToolResult(truncated_content="Empty command.", exit_code=1)
    # Still cap length to avoid pathological input.
    if len(cmd) > 4000:
        return ToolResult(truncated_content="Command too long (max 4000 chars).", exit_code=1)
    t0 = time.perf_counter()
    # ``mvn install`` on a big project with a cold dependency cache can
    # legitimately take 20-40 min; 1h is a sane cap.
    timeout = (
        min(_tool_timeout(), _BASH_PERTEST_TIMEOUT_SECONDS)
        if "-Dtest=" in cmd
        else _tool_timeout()
    )
    r = sandbox.exec(cmd, timeout=timeout)
    elapsed = time.perf_counter() - t0
    if r.timed_out:
        return ToolResult(
            truncated_content=(
                f"$ {cmd}\nexit=124\nTIMEOUT after {timeout}s. "
                "The command was killed. Do not retry it unchanged; narrow "
                "the Maven module list, remove unnecessary reactor flags, and "
                "identify the hanging test before resubmitting."
            ),
            exit_code=EXEC_TIMEOUT_EXIT_CODE,
            elapsed_s=elapsed,
        )
    obs = f"$ {cmd}\nexit={r.exit_code}\n{r.stdout}".rstrip()
    if r.stderr:
        obs += f"\n[stderr]\n{r.stderr}"
    return ToolResult(
        truncated_content=_truncate(obs),
        exit_code=r.exit_code,
        elapsed_s=elapsed,
    )


def _mvn_install(sandbox: TestSandbox, _args: str) -> ToolResult:
    """Explicit mvn install helper for setup agent — bundles the flags
    that RepoLaunch's setup phase would run."""
    cmd = runtime_spec().install_command
    t0 = time.perf_counter()
    r = sandbox.exec(cmd, timeout=_tool_timeout())
    elapsed = time.perf_counter() - t0
    if r.timed_out:
        return ToolResult(
            truncated_content=(
                f"$ {cmd}\nexit=124\nTIMEOUT after {_tool_timeout()}s. "
                "Do not retry unchanged. Use a narrower reactor scope and "
                "diagnose which module or dependency is hanging."
            ),
            exit_code=EXEC_TIMEOUT_EXIT_CODE,
            elapsed_s=elapsed,
        )
    install_failed = r.exit_code != 0 or _has_maven_errors(r)
    obs = (
        _summarize_javac(r, force_failure=install_failed)
        if install_failed
        else f"mvn install OK (exit={r.exit_code}, {elapsed:.1f}s)"
    )
    return ToolResult(
        truncated_content=_truncate(obs),
        exit_code=1 if install_failed else 0,
        elapsed_s=elapsed,
    )


# ─── helpers ─────────────────────────────────────────────────────────────────


def extract_diff_from_submit(payload: str) -> str:
    """Pull the unified diff out of a ``<submit>`` payload.

    Same three-tier preference as ``benchmark.generate._extract_diff``:
    fenced ``diff`` block > any ``diff --git`` onward > raw text.
    """
    m = re.search(r"```diff\s*\n(.*?)```", payload, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"(diff --git .+)", payload, re.DOTALL)
    if m:
        return m.group(1)
    return payload.strip()


def _strip_fenced_diff(payload: str) -> str:
    """Same as extract_diff_from_submit but named for the apply_diff caller.
    The LLM sometimes wraps the diff in a fence inside ``<apply_diff>`` and
    sometimes not — accept both."""
    return extract_diff_from_submit(payload)


def _has_maven_errors(r: CommandResult) -> bool:
    """True when Maven emitted javac/compiler ``[ERROR]`` lines.

    ``-Dmaven.compiler.failOnError=false`` intentionally lets Maven exit 0
    despite compiler errors. The tool contract must still report that as a
    failure to the ReAct loop, or the agent will submit a non-compiling diff.
    """
    return "[ERROR]" in r.stdout or "[ERROR]" in r.stderr


def _summarize_javac(r: CommandResult, *, force_failure: bool = False) -> str:
    """Extract the useful bits from Maven test-compile output.

    Maven's ``-q`` still emits ERROR lines from javac. We keep those plus
    the first ~30 lines around them; drop the download noise.
    """
    if not force_failure and r.ok:
        return f"test-compile OK (exit={r.exit_code})"
    lines = r.stdout.splitlines()
    err_indices = [i for i, line in enumerate(lines) if "[ERROR]" in line]
    if not err_indices:
        return f"test-compile FAILED (exit={r.exit_code}) — no [ERROR] lines found. tail:\n{r.stdout[-2000:]}"
    # Give a 3-line window around each ERROR line to keep enough context.
    keep = set()
    for i in err_indices:
        for j in range(max(0, i - 1), min(len(lines), i + 4)):
            keep.add(j)
    picked = "\n".join(lines[j] for j in sorted(keep))
    return f"test-compile FAILED (exit={r.exit_code}). Compiler errors:\n{picked}"
