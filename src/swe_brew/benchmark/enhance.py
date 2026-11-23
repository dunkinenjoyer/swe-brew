"""CLI orchestrator for the SWE-BREW benchmark enhancer — 6-stage pipeline.

    swe-brew-enhance <case-dir> [options]

Pipeline stages (per docs/DESIGN-ENHANCED-2026-09-16.md §1):

    snapshot → sandbox → setup_agent → organize_agent → generate → verify

Artefacts written to ``<case-dir>/enhanced/`` (per docs/DESIGN §5):

    result.json          canonical RepoLaunch-shape manifest (aggregate)
    setup.json           SetupOutput
    organize.json        OrganizeOutput
    verify_report.json   full 6-way matrix + candidate acceptance gate + dual gold scoping
    test_results.json    lean scorecard-facing projection
    telemetry.jsonl      per-event stream
    stages/{setup,organize,generate}/trial-N/{turns.jsonl, diff, outcome.json}
    logs/…

Fault tolerance (grounded on ``launch/agent/{setup,organize}/save.py``):
Each stage is wrapped so its exception is recorded in ``result.json``
but never kills downstream stages. Even a fully-failed case emits a
valid ``result.json`` with ``completed: false, exception: "..."`` — the
'never lose the trajectory' invariant.

Flags of interest:
    --stub-llm             Use gold test_patch as candidate (skip real LLM).
    --use-agent-loop       Enable the ReAct test-gen agent.
    --use-setup-agent      Enable the setup ReAct agent (default: True when
                           --use-agent-loop).
    --use-organize-agent   Enable the organize ReAct agent (default: True
                           when --use-agent-loop).
    --workspace DIR        Where snapshot worktrees + logs live.
    --dry-run              Plan-only; stop before container start.
    --keep-sandbox         Leave the container running after we finish.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from swe_brew.agent.archive import (
    archive_final,
    archive_stage_final,
    archive_trial,
    capture_sandbox_diff,
    safe_write_json,
)
from swe_brew.agent.framework import AgentConfig
from swe_brew.agent.loop import AgentGenerator
from swe_brew.agent.organize_agent import (
    OrganizeAgent,
    OrganizeOutput,
    derive_test_scope_validated,
    parse_organize_manifest,
)
from swe_brew.agent.progress import TelemetryLog, make_progress, stage
from swe_brew.agent.prompt import render_history_hints
from swe_brew.agent.setup_agent import SetupAgent, SetupOutput
from swe_brew.benchmark import base_commit as _bc
from swe_brew.benchmark.case_adapter import load_case_from_dir
from swe_brew.benchmark.container import start_sandbox
from swe_brew.benchmark.generate import (
    GenerationError,
    Generator,
    LLMGenerator,
    StubGenerator,
)
from swe_brew.benchmark.git_diff import derive_patches
from swe_brew.benchmark.models import TestResults, TestStatus, enhanced_dir
from swe_brew.benchmark.module_model import ValidatedModuleSet
from swe_brew.benchmark.patch_split import is_test_path, split_gold_patch
from swe_brew.benchmark.result_json import build_result_json
from swe_brew.benchmark.runtime import runtime_spec
from swe_brew.benchmark.snapshot import cleanup_snapshot, create_snapshot, local_clone_path
from swe_brew.benchmark.surefire import parse_surefire_reports, surefire_report_paths
from swe_brew.benchmark.test_discovery import parse_test_declarations
from swe_brew.benchmark.verify import (
    new_fail_to_fail_ids,
    new_pass_to_fail_ids,
    verify_case,
)

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_reused_stages(
    source: Path,
) -> tuple[SetupOutput | None, OrganizeOutput | None, str]:
    """Load a previous attempt's setup/organize outputs and pipeline patch.

    The build-fix patch is optional. It was only persisted for attempts
    created after ``pipeline.patch`` persistence landed, so callers must
    tolerate an empty string and degrade to the documented behavior
    (setup/organize metadata is still reused; build fixes are lost).
    """
    setup_output: SetupOutput | None = None
    organize_output: OrganizeOutput | None = None
    pipeline_patch = ""

    setup_path = source / "setup.json"
    if setup_path.exists():
        data = json.loads(setup_path.read_text(encoding="utf-8"))
        setup_output = SetupOutput(
            success=bool(data.get("success", False)),
            duration_s=float(data.get("duration_s", 0.0)),
            submitted=bool(data.get("submitted", False)),
            commands_run=list(data.get("commands_run", [])),
            modifications=list(data.get("modifications", [])),
            submit_notes=str(data.get("submit_notes", "")),
            trials_used=int(data.get("trials_used", 0)),
            steps_used=int(data.get("steps_used", 0)),
            cost=dict(data.get("cost", {})),
            notes=list(data.get("notes", [])),
        )

    organize_path = source / "organize.json"
    if organize_path.exists():
        data = json.loads(organize_path.read_text(encoding="utf-8"))
        organize_output = OrganizeOutput(
            success=bool(data.get("success", False)),
            duration_s=float(data.get("duration_s", 0.0)),
            submitted=bool(data.get("submitted", False)),
            test_command=str(data.get("test_command", runtime_spec().test_command)),
            modules=list(data.get("modules", [])),
            validated_modules=list(data.get("validated_modules", [])),
            test_scope_full=list(data.get("test_scope_full", [])),
            test_scope_local=list(data.get("test_scope_local", [])),
            test_scope_developer=list(data.get("test_scope_developer", [])),
            pertest_template=str(
                data.get("pertest_template", runtime_spec().pertest_template)
            ),
            trials_used=int(data.get("trials_used", 0)),
            steps_used=int(data.get("steps_used", 0)),
            cost=dict(data.get("cost", {})),
            fallback_used=bool(data.get("fallback_used", False)),
            notes=list(data.get("notes", [])),
        )

    pipeline_path = source / "pipeline.patch"
    if pipeline_path.exists():
        pipeline_patch = pipeline_path.read_text(encoding="utf-8")

    return setup_output, organize_output, pipeline_patch


# ─── the 6 stages, each fault-tolerant ───────────────────────────────────────


@app.command()
def enhance(
    case_dir: Path = typer.Argument(..., help="Directory containing a case.json"),
    stub_llm: bool = typer.Option(False, "--stub-llm", help="Use gold test_patch as candidate (skip real LLM call)"),
    workspace: Path | None = typer.Option(None, "--workspace", help="Where snapshot worktrees + logs live"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan-only; stop before container start"),
    keep_sandbox: bool = typer.Option(False, "--keep-sandbox", help="Leave the container running after we finish"),
    test_command_override: str = typer.Option("", "--test-command", help="Override the test_command discovered by organize (empty = use discovered/fallback)"),
    use_agent_loop: bool = typer.Option(False, "--use-agent-loop", help="Drive test-gen through the ReAct loop"),
    use_setup_agent: bool = typer.Option(True, "--use-setup-agent/--no-setup-agent", help="Enable the setup ReAct agent (default: on)"),
    use_organize_agent: bool = typer.Option(True, "--use-organize-agent/--no-organize-agent", help="Enable the organize ReAct agent (default: on)"),
    agent_max_trials: int = typer.Option(10, "--agent-max-trials", help="Soft cap on trials per agent (default 10; unbounded steps within each trial)"),
    agent_max_steps: int = typer.Option(50, "--agent-max-steps", help="Hard per-trial step cap for setup/organize agents (default 50)"),
    test_gen_max_trials: int = typer.Option(6, "--test-gen-max-trials", help="Soft cap on test-generation trials (default 6)"),
    test_gen_max_steps: int = typer.Option(100, "--test-gen-max-steps", help="Hard per-trial step cap for test generation (default 100)"),
    out_dir: Path | None = typer.Option(None, "--out-dir", help="Override the output directory (default: <case-dir>/enhanced). Used by pass@k orchestrator to redirect each attempt to its own subdir."),
    attempt_index: int = typer.Option(1, "--attempt-index", help="1-based attempt index for pass@k diversity nudges. Attempt 1 is baseline; higher attempts get different discipline emphases (exception-path, boundary, happy-path, state-transition)."),
    reuse_stages: bool = typer.Option(False, "--reuse-stages", help="Reuse setup/organize artifacts instead of running those agents. Requires --reuse-source."),
    reuse_source: Path | None = typer.Option(None, "--reuse-source", help="Directory containing a previous setup.json, organize.json, and optional pipeline.patch."),
    collect_coverage: bool = typer.Option(False, "--collect-coverage", help="Run two extra JaCoCo legs (pred_pre/pred_post) and emit executable-line coverage delta."),
) -> None:
    """Enhance a single case with the 6-stage RepoLaunch-parity pipeline."""
    load_dotenv(Path.cwd() / ".env", override=False)

    workspace = workspace or Path(os.environ.get("SWE_BREW_WORKSPACE", "_work")).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    started_at = dt.datetime.now(tz=dt.UTC)
    t_start = time.perf_counter()

    # ── input ──────────────────────────────────────────────────────────────
    case = load_case_from_dir(case_dir)
    scope_split = split_gold_patch(case.gold_patch)
    clone = local_clone_path(case.repo_short, getattr(case, 'local_repo', None))
    console.rule("[bold]resolve")
    dual = _bc.resolve_dual(case, clone, test_files=scope_split.test_files)
    console.print(
        f"base_commit = {dual.base_commit[:12]}   head_commit = {dual.head_commit[:12]}   ({dual.method})",
    )
    split = derive_patches(
        clone, dual.base_commit, dual.head_commit,
        test_scope=scope_split.test_files,
        code_scope=None,
    )

    # ── plan ───────────────────────────────────────────────────────────────
    _render_plan(case, dual, split, stub_llm, use_agent_loop, use_setup_agent,
                 use_organize_agent, workspace, dry_run)

    if dry_run:
        console.print("[yellow]--dry-run set; stopping before container start.[/yellow]")
        return

    # ── stage bookkeeping ──────────────────────────────────────────────────
    # Pass@k: honour --out-dir if the orchestrator gave us one, so each
    # attempt's artefacts land in its own subdir (e.g.
    # ``<case-dir>/enhanced-attempt-001/``). Default: ``<case-dir>/enhanced/``.
    if out_dir is not None:
        out = out_dir
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = enhanced_dir(case_dir)
    telemetry_path = out / "telemetry.jsonl"

    # Which stages are enabled? All 6 always run in principle, but
    # setup/organize can be skipped for backward-compat / stub mode.
    if reuse_stages:
        if reuse_source is None:
            exception_msg = "reuse-stages requires --reuse-source"
            console.print(f"[red]{exception_msg}[/red]")
            _write_failed_manifest(enhanced_dir(case_dir), case, exception_msg, started_at, t_start)
            return
        setup_output, organize_output, pipeline_patch = _load_reused_stages(reuse_source)
        if setup_output is not None:
            archive_stage_final(out, "setup.json", setup_output)
        if organize_output is not None:
            archive_stage_final(out, "organize.json", organize_output)
        setup_on = False
        organize_on = False
    else:
        pipeline_patch = ""
        setup_on    = use_agent_loop and use_setup_agent and not stub_llm
        organize_on = use_agent_loop and use_organize_agent and not stub_llm
    n_stages = 4 + (1 if setup_on else 0) + (1 if organize_on else 0)

    exception_msg: str | None = None
    if not reuse_stages:
        setup_output = None
        organize_output = None
    verify_report = None
    sandbox = None
    snapshot = None

    with TelemetryLog(telemetry_path, case.case_id) as tel, make_progress(console) as progress:
        task = progress.add_task(
            f"[bold]{case.case_id[:60]}",
            total=n_stages,
            success=0,
            fail=0,
        )

        # 1. snapshot
        try:
            with stage(progress, task, tel, "snapshot"):
                snapshot = create_snapshot(
                    case,
                    workspace,
                    base_commit=dual.base_commit,
                    suffix=f"{attempt_index or 1}-{int(time.time())}",
                )
                tel.emit("snapshot.done",
                         base_commit=snapshot.base_commit,
                         work_dir=str(snapshot.work_dir),
                         resolution_method=snapshot.resolution_method)
                console.print(f"[dim]snapshot @ {snapshot.base_commit[:12]}  work_dir = {snapshot.work_dir}[/dim]")
        except Exception as e:
            exception_msg = f"snapshot failed: {type(e).__name__}: {e!s}"
            console.print(f"[red]{exception_msg}[/red]")
            _write_failed_manifest(out, case, exception_msg, started_at, t_start)
            return

        # 2. sandbox
        try:
            with stage(progress, task, tel, "sandbox"):
                sandbox = start_sandbox(snapshot)
                tel.emit("sandbox.done",
                         image_tag=sandbox.image_tag,
                         container_short_id=sandbox.container.short_id)
                console.print(f"[dim]container = {sandbox.container.short_id}   image_tag = {sandbox.image_tag}[/dim]")
        except Exception as e:
            exception_msg = f"sandbox failed: {type(e).__name__}: {e!s}"
            console.print(f"[red]{exception_msg}[/red]")
            _write_failed_manifest(out, case, exception_msg, started_at, t_start,
                                   snapshot=snapshot)
            return

        try:
            # 3. setup agent (optional)
            if setup_on:
                setup_output = _run_setup_stage(
                    progress, task, tel, sandbox, snapshot, case, out,
                    max_trials=agent_max_trials,
                    max_steps_per_trial=agent_max_steps,
                )

            # 4. organize agent (optional)
            if organize_on:
                organize_output = _run_organize_stage(
                    progress, task, tel, sandbox, snapshot, case, out,
                    setup_output=setup_output,
                    max_trials=agent_max_trials,
                    max_steps_per_trial=agent_max_steps,
                )

            # Persist any setup/organize build fixes before generate and
            # verify. Verify resets each leg to base_commit, so this patch
            # is the only way to retain POM/settings changes.
            if not reuse_stages:
                pipeline_patch = capture_sandbox_diff(
                    snapshot.work_dir,
                    snapshot.base_commit,
                )
            (out / "pipeline.patch").write_text(
                pipeline_patch,
                encoding="utf-8",
            )

            # 5. generate — 1:1 downstream of organize. ``max_trials`` /
            # ``max_steps`` come from :class:`AgentGenerator` defaults.
            gen_result = _run_generate_stage(
                progress, task, tel, sandbox, case, split, out,
                stub_llm=stub_llm,
                use_agent_loop=use_agent_loop,
                organize_output=organize_output,
                attempt_index=attempt_index,
                agent_max_trials=test_gen_max_trials,
                agent_max_steps=test_gen_max_steps,
            )

            # 6. verify — consumes organize.test_command if available.
            # :func:`_resolve_test_command` widens ``-pl`` only for modules
            # the candidate diff actually touches; it never injects ``-am``.
            test_command = _resolve_test_command(
                override=test_command_override,
                organize=organize_output,
                candidate_diff=gen_result.candidate_patch,
            )
            verify_report = _run_verify_stage(
                progress, task, tel, sandbox, case, split,
                gen_result=gen_result,
                test_command=test_command,
                out=out,
                setup_output=setup_output,
                organize_output=organize_output,
                raw_gold_patch_parse=scope_split.classification,
                code_patch_source=split.classification,
                pipeline_patch=pipeline_patch,
                collect_coverage=collect_coverage,
            )

        except Exception as e:
            exception_msg = f"pipeline exception in one of the agent stages: {type(e).__name__}: {e!s}"
            console.print(f"[red]{exception_msg}[/red]")
        finally:
            if sandbox and not keep_sandbox:
                sandbox.stop()
            if snapshot is not None:
                cleanup_snapshot(snapshot.source_clone, snapshot.work_dir)

    # ── write canonical result.json (RepoLaunch-shape) ─────────────────────
    elapsed = time.perf_counter() - t_start
    docker_layers = _extract_docker_layers(verify_report, sandbox, snapshot)
    manifest = build_result_json(
        case_id=case.case_id,
        completed=(exception_msg is None and verify_report is not None),
        organize_completed=(organize_output is not None and organize_output.success),
        exception=exception_msg,
        duration_seconds=elapsed,
        docker_image=(sandbox.image_tag if sandbox else ""),
        docker_image_layers=docker_layers,
        setup_output=setup_output,
        organize_output=organize_output,
        verify_report=verify_report,
        started_at=started_at,
    )
    archive_final(out, manifest)

    _render_final_summary(out, verify_report, exception_msg)


# ─── stage runners ───────────────────────────────────────────────────────────


def _run_setup_stage(progress, task, tel, sandbox, snapshot, case, out, max_trials,
                     max_steps_per_trial=50):
    """Setup agent — validate build. Fault-tolerant: on failure, records
    the exception in setup.json and returns a degraded SetupOutput; does
    not raise."""
    with stage(progress, task, tel, "setup"):
        agent = SetupAgent(
            sandbox=sandbox,
            case_id=case.case_id,
            telemetry=tel,
            config=AgentConfig(
                max_trials=max_trials,
                max_steps_per_trial=max_steps_per_trial,
            ),
        )
        history_hints = render_history_hints(
            case.setup_cmds,
            case.test_cmds,
            avoid_commands=case.avoid_commands,
        )
        t0 = time.perf_counter()
        try:
            result = agent.run(
                repo_short=case.repo_short,
                base_commit=snapshot.base_commit,
                changed_files=case.changed_files,
                history_hints=history_hints,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            output = SetupOutput(
                success=False, duration_s=elapsed, submitted=False,
                notes=[f"setup exception: {type(e).__name__}: {e!s}"],
            )
            archive_stage_final(out, "setup.json", output)
            return output

        elapsed = time.perf_counter() - t0
        # Capture what setup actually modified in the sandbox.
        modifications_diff = capture_sandbox_diff(snapshot.work_dir, snapshot.base_commit)

        # Trial archive for the setup stage's final trial.
        archive_trial(
            out, stage="setup", trial=result.trials_used,
            loop_result=result,
            diff_content=modifications_diff,
            extra_outcome={"submit_notes": result.submit_payload},
        )

        output = SetupOutput(
            success=result.submitted,
            duration_s=elapsed,
            submitted=result.submitted,
            commands_run=[k for k, _, _ in result.tool_calls],
            modifications=[{"path": "sandbox diff", "diff_chars": len(modifications_diff)}]
                           if modifications_diff.strip() else [],
            submit_notes=result.submit_payload,
            trials_used=result.trials_used,
            steps_used=result.steps_used,
            cost=result.state.cost.get("setup", {}),
            notes=[],
        )
        if output.success:
            try:
                sandbox.clear_surefire_reports()
                sandbox.exec(
                    runtime_spec().test_command,
                    timeout=runtime_spec().command_timeout_seconds,
                )
                verify_outcomes = parse_surefire_reports(snapshot.work_dir)
                passed = sum(
                    1 for o in verify_outcomes
                    if o.status is TestStatus.PASSED
                )
                failed = sum(
                    1 for o in verify_outcomes
                    if o.status in (TestStatus.FAILED, TestStatus.ERROR)
                )
                if not verify_outcomes:
                    output.success = False
                    output.notes.append(
                        "setup verify produced no Surefire reports; fail closed",
                    )
                elif passed == 0:
                    output.success = False
                    output.notes.append(
                        "setup verify produced no passing tests; fail closed",
                    )
                else:
                    output.notes.append(
                        f"setup verify passed={passed} failed={failed} "
                        f"total={len(verify_outcomes)}",
                    )
            except Exception as e:  # noqa: BLE001
                output.success = False
                output.notes.append(
                    f"setup verify exception: {type(e).__name__}: {e!s}",
                )
        archive_stage_final(out, "setup.json", output)
        return output


def _run_organize_stage(progress, task, tel, sandbox, snapshot, case, out,
                        setup_output, max_trials, max_steps_per_trial=50):
    """Organize agent — discover test_command + inventory tests.
    Fault-tolerant: falls back to hardcoded default if the agent fails."""
    with stage(progress, task, tel, "organize"):
        setup_summary = "(no setup output)"
        if setup_output is not None:
            # Feed the full submit_notes; the whole point of the
            # setup → organize handoff is context transfer.
            setup_summary = (
                f"setup success={setup_output.success}, "
                f"submit_notes={setup_output.submit_notes}"
            )
        agent = OrganizeAgent(
            sandbox=sandbox,
            case_id=case.case_id,
            telemetry=tel,
            config=AgentConfig(
                max_trials=max_trials,
                max_steps_per_trial=max_steps_per_trial,
            ),
        )
        history_hints = render_history_hints(
            case.setup_cmds,
            case.test_cmds,
            avoid_commands=case.avoid_commands,
        )
        t0 = time.perf_counter()
        try:
            result = agent.run(
                repo_short=case.repo_short,
                base_commit=snapshot.base_commit,
                changed_files=case.changed_files,
                setup_summary=setup_summary,
                history_hints=history_hints,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            output = OrganizeOutput(
                success=False, duration_s=elapsed, submitted=False,
                fallback_used=True,
                notes=[f"organize exception: {type(e).__name__}: {e!s}"],
            )
            archive_stage_final(out, "organize.json", output)
            return output

        elapsed = time.perf_counter() - t0
        modifications_diff = capture_sandbox_diff(snapshot.work_dir, snapshot.base_commit)
        archive_trial(
            out, stage="organize", trial=result.trials_used,
            loop_result=result,
            diff_content=modifications_diff,
            extra_outcome={"submit_payload_preview": result.submit_payload[:500]},
        )

        manifest = parse_organize_manifest(result.submit_payload)
        if "failure" in manifest or not result.submitted:
            reason = manifest.get("failure", "no submit") if isinstance(manifest, dict) else "no submit"
            output = OrganizeOutput(
                success=False, duration_s=elapsed, submitted=result.submitted,
                fallback_used=True,
                trials_used=result.trials_used,
                steps_used=result.steps_used,
                cost=result.state.cost.get("organize", {}),
                notes=[f"organize did not produce a valid manifest: {reason}"],
            )
        else:
            full_files = manifest.get("test_scope_full_files") or manifest.get("test_scope_full") or []
            raw_test_command = (
                manifest.get("test_command", "").strip()
                or runtime_spec().test_command
            )
            modules = manifest.get("modules", [])
            test_bearing_files: list[str] = []
            for rel_path in full_files:
                if not rel_path or "/" not in rel_path or rel_path.startswith("/"):
                    continue
                source_path = snapshot.work_dir / rel_path
                if not source_path.exists():
                    continue
                try:
                    declarations = parse_test_declarations(
                        source_path.read_text(encoding="utf-8"),
                    )
                except Exception:  # noqa: BLE001
                    declarations = []
                if declarations:
                    test_bearing_files.append(rel_path)
            test_bearing_modules = {
                rel_path.split("/", 1)[0]
                for rel_path in test_bearing_files
            }
            test_command = _expand_organize_test_command(
                raw_test_command,
                test_bearing_modules,
            )
            sandbox.clear_surefire_reports()
            sandbox.exec(
                test_command,
                timeout=runtime_spec().command_timeout_seconds,
            )
            report_modules = {
                path.relative_to(snapshot.work_dir).parts[0]
                for path in surefire_report_paths(snapshot.work_dir)
                if len(path.relative_to(snapshot.work_dir).parts) > 1
            }
            if not report_modules:
                output = OrganizeOutput(
                    success=False,
                    duration_s=elapsed,
                    submitted=result.submitted,
                    fallback_used=True,
                    trials_used=result.trials_used,
                    steps_used=result.steps_used,
                    cost=result.state.cost.get("organize", {}),
                    notes=["organize command produced no Surefire reports; fail closed"],
                )
                archive_stage_final(out, "organize.json", output)
                return output
            validated = ValidatedModuleSet.from_executed_reports(
                modules,
                report_modules,
            )
            developer_scope = list(case.changed_files or [])
            spec = runtime_spec()
            developer_test_files = [f for f in developer_scope
                                    if spec.test_path_marker in f"/{f.lstrip('/')}"
                                    and any(f.endswith(sfx) for sfx in spec.test_file_suffixes)]
            local_files = derive_test_scope_validated(
                full_files,
                list(validated.validated_modules),
            )
            # ``OrganizeOutput`` has three required positional fields
            # (``success``, ``duration_s``, ``submitted``). Always
            # supply them explicitly — no-arg construction crashes.
            output = OrganizeOutput(
                success=True,
                duration_s=elapsed,
                submitted=True,
                test_command=test_command,
                modules=modules,
                validated_modules=list(validated.validated_modules),
                test_scope_full=full_files,
                test_scope_local=local_files,
                test_scope_developer=developer_test_files,
                pertest_template=manifest.get(
                    "pertest_template",
                    runtime_spec().pertest_template,
                ),
                trials_used=result.trials_used,
                steps_used=result.steps_used,
                cost=result.state.cost.get("organize", {}),
                fallback_used=False,
                notes=manifest.get("notes", []),
            )

        archive_stage_final(out, "organize.json", output)
        return output


def _build_generate_sibling_tests(
    organize_output: OrganizeOutput | None,
) -> list[str] | None:
    """Return organize's validated local test inventory.

    This is evidence the organize agent actually ran, not a guessed host
    module. Fall back to the full reported inventory only when no local
    test files were validated.
    """
    if organize_output is None:
        return None
    return (
        list(organize_output.test_scope_local)
        or list(organize_output.test_scope_full)
        or None
    )


def _run_generate_stage(progress, task, tel, sandbox, case, split, out,
                        stub_llm, use_agent_loop,
                        organize_output=None, attempt_index=1,
                        agent_max_trials=2,
                        agent_max_steps=50):
    """Test-gen stage — either stub, single-shot LLM, or ReAct agent loop.

    Test-gen is 1:1 downstream of organize. The organize agent's
    validated ``test_scope_local`` is threaded into the agent so it can
    ``<bash>cat`` a sibling test before writing its first line — same
    JUnit version, same Mockito idioms, same assertion library.
    ``attempt_index`` seeds the pass@k diversity nudge.
    """
    with stage(progress, task, tel, "generate"):
        # Use organize's validated local inventory; if empty, fall back
        # to the full inventory. When organize didn't run at all, hand
        # None so the agent's fallback discovery discipline kicks in.
        sibling_tests: list[str] | None = None
        if organize_output is not None:
            available_test_modules = sorted(organize_output.validated_modules)
            sibling_tests = _build_generate_sibling_tests(
                organize_output,
            )

        gen: Generator
        if stub_llm:
            gen = StubGenerator()
        elif use_agent_loop:
            gen = AgentGenerator(
                sandbox=sandbox,
                telemetry=tel,
                sibling_test_files=sibling_tests,
                validated_modules=available_test_modules or [],
                organize_notes=list(organize_output.notes) if organize_output else [],
                organize_test_command=(
                    organize_output.test_command
                    if organize_output and organize_output.test_command
                    else ""
                ),
                attempt_index=attempt_index,
                max_trials=agent_max_trials,
                max_steps_per_trial=agent_max_steps,
            )
        else:
            gen = LLMGenerator()
        gen_result = gen.generate(case, split)
        if organize_output is not None and not gen_result.is_stub:
            _ensure_candidate_modules_are_test_paths(gen_result.candidate_patch)
        console.print(
            f"[dim]candidate_patch chars = {len(gen_result.candidate_patch)}  "
            f"is_stub={gen_result.is_stub}  model={gen_result.model or '-'}[/dim]",
        )
        # Trial archive for the final generate trial (we don't have per-
        # trial data through the Generator protocol; the agent-loop path
        # emits per-turn telemetry which is captured in telemetry.jsonl).
        try:
            trial_dir = out / "stages" / "generate" / "trial-001"
            trial_dir.mkdir(parents=True, exist_ok=True)
            (trial_dir / "candidate.diff").write_text(gen_result.candidate_patch, encoding="utf-8")
            if gen_result.raw_response:
                (trial_dir / "transcript.txt").write_text(
                    gen_result.raw_response,
                    encoding="utf-8",
                )
            safe_write_json(trial_dir / "outcome.json", {
                "stage":           "generate",
                "trial":           1,
                "is_stub":         gen_result.is_stub,
                "model":           gen_result.model,
                "prompt_tokens":   gen_result.prompt_tokens,
                "completion_tokens": gen_result.completion_tokens,
                "candidate_chars": len(gen_result.candidate_patch),
            })
        except Exception:                        # noqa: BLE001
            pass
        return gen_result


def _run_verify_stage(progress, task, tel, sandbox, case, split, gen_result,
                      test_command, out, setup_output, organize_output,
                      raw_gold_patch_parse="unidiff-parsed",
                      code_patch_source="git-native",
                      pipeline_patch="",
                      collect_coverage=False):
    """Verify — 6-way matrix + candidate acceptance gate + dual gold scoping.
    Fault-tolerant: if verify_case raises, we still write a minimal
    verify_report.json marker."""
    with stage(progress, task, tel, "verify"):
        try:
            # Wire organize's 3-way scope into verify.
            report = verify_case(
                case,
                split,
                gen_result.candidate_patch,
                sandbox,
                test_command=test_command,
                is_stub=gen_result.is_stub,
                llm_model=gen_result.model,
                test_scope_developer=(organize_output.test_scope_developer
                                      if organize_output else None),
                test_scope_discovered=(organize_output.test_scope_full
                                       if organize_output else None),
                test_scope_local=(organize_output.test_scope_local
                                  if organize_output else None),
                raw_gold_patch_parse=raw_gold_patch_parse,
                code_patch_source=code_patch_source,
                pipeline_patch=pipeline_patch,
                collect_coverage=collect_coverage,
            )
            bad_tests = new_fail_to_fail_ids(report.runs) | new_pass_to_fail_ids(
                report.runs,
            )
            if bad_tests:
                pruned = _prune_candidate_bad_tests(
                    gen_result.candidate_patch,
                    bad_tests,
                )
                if pruned != gen_result.candidate_patch:
                    tel.emit(
                        "verify.prune_candidate",
                        removed_test_ids=sorted(bad_tests),
                        candidate_before=len(gen_result.candidate_patch),
                        candidate_after=len(pruned),
                    )
                    # Keep the generated artifact consistent with the
                    # precision-stage verdict. Pass@k merging reads the
                    # verify report, but auditors must not see a stale
                    # candidate.diff containing tests the gate removed.
                    (out / "stages" / "generate" / "trial-001" / "candidate.diff").write_text(
                        pruned,
                        encoding="utf-8",
                    )
                    gen_result.candidate_patch = pruned
                    report = verify_case(
                        case,
                        split,
                        pruned,
                        sandbox,
                        test_command=test_command,
                        is_stub=gen_result.is_stub,
                        llm_model=gen_result.model,
                        test_scope_developer=(organize_output.test_scope_developer
                                              if organize_output else None),
                        test_scope_discovered=(organize_output.test_scope_full
                                               if organize_output else None),
                        test_scope_local=(organize_output.test_scope_local
                                          if organize_output else None),
                        raw_gold_patch_parse=raw_gold_patch_parse,
                        code_patch_source=code_patch_source,
                        pipeline_patch=pipeline_patch,
                        collect_coverage=collect_coverage,
                    )
            for cfg, r in report.runs.items():
                tel.emit("verify.run.end",
                         configuration=cfg,
                         tests_count=len(r.tests),
                         exit_code=r.exit_code,
                         elapsed_s=r.ran_seconds)
            _emit_gate_summary(tel, report)

            # Persist
            (out / "verify_report.json").write_text(
                report.model_dump_json(indent=2), encoding="utf-8",
            )
            results = TestResults(
                case_id=case.case_id,
                image_tag=sandbox.image_tag,
                test_command=test_command,
                FAIL_TO_PASS=report.fail_to_pass,
                PASS_TO_PASS=report.pass_to_pass,
                ASSERTION_F2P=report.assertion_f2p,
                COMPILE_F2P=report.compile_f2p,
                validity_gate=report.validity_gate,
                meta={
                    "base_commit": case.base_commit or "",
                    "resolution_method": "",
                    "llm_stub": str(report.llm_stub),
                    "llm_model": report.llm_model or "",
                    "test_command_source": ("organize" if organize_output and organize_output.success
                                            else "override" if test_command != runtime_spec().test_command
                                            else "default"),
                },
            )
            (out / "test_results.json").write_text(
                results.model_dump_json(indent=2), encoding="utf-8",
            )
            return report
        except Exception as e:
            marker = {
                "exception": f"{type(e).__name__}: {e!s}",
                "validity_gate": False,
                "test_command_used": test_command,
                "note": "verify_case crashed — see logs and telemetry for detail",
            }
            safe_write_json(out / "verify_report.json", marker)
            tel.emit("verify.exception",
                     error=marker["exception"], test_command=test_command)
            return None


# ─── helpers ─────────────────────────────────────────────────────────────────


def _resolve_test_command(
    override: str,
    organize: OrganizeOutput | None,
    *,
    candidate_diff: str = "",
) -> str:
    """Pick the test_command with clear precedence:

    1. ``--test-command`` flag if provided (respected verbatim);
    2. organize agent's discovered command when it succeeded;
    3. :func:`runtime_spec` fallback command.

    Whichever base wins (except the explicit override) is post-processed
    by :func:`_augment_test_command_scope`: the ``-pl`` scope is unioned
    with the candidate diff's module only. The candidate patch is kept in
    the module where test-gen wrote it; verify is responsible for running
    that module and grading the result.
    """
    if override.strip():
        return override.strip()

    if organize and organize.success and organize.test_command.strip():
        base = organize.test_command.strip()
    else:
        base = runtime_spec().test_command

    return _augment_test_command_scope(
        base,
        candidate_diff=candidate_diff,
    )


_PL_ARG_RE = re.compile(r"(-pl\s+)(\S+)")
_DIFF_MODULE_RE = re.compile(r"^\+\+\+\s+b/([^/]+)/", re.MULTILINE)


def _modules_from_diff(diff_text: str) -> set[str]:
    """Extract the top-level module dir from every ``+++ b/<module>/…``
    line in a unified diff. Returns the set of touched module names."""
    if not diff_text:
        return set()
    return set(_DIFF_MODULE_RE.findall(diff_text))


def _ensure_candidate_modules_are_test_paths(candidate_diff: str) -> None:
    """Keep the candidate search space broad but structurally Maven-test-only.

    Recall is intentionally high here: any reported Maven test module may be
    a candidate. The downstream six-way verify is responsible for precision.
    We only reject non-test paths or non-Maven module names.
    """
    paths: list[str] = []
    for path_match in re.finditer(
        r"^\+\+\+\s+b/(\S+)",
        candidate_diff or "",
        re.MULTILINE,
    ):
        paths.append(path_match.group(1))
    for path_match in re.finditer(
        r"^diff --git a/(\S+) b/(\S+)",
        candidate_diff or "",
        re.MULTILINE,
    ):
        left, right = path_match.groups()
        paths.append(right if right != "/dev/null" else left)

    bad_paths: list[str] = []
    bad_modules: list[str] = []
    for path in paths:
        if path == "/dev/null" or not is_test_path(path):
            bad_paths.append(path)
        else:
            module = path.split("/", 1)[0]
            if module.startswith(".") or not re.fullmatch(r"[A-Za-z0-9_.-]+", module):
                bad_modules.append(module)
    if bad_paths or bad_modules:
        raise GenerationError(
            "candidate patch has non-Maven-test paths or module names: "
            + ", ".join(sorted(set(bad_paths + bad_modules))),
        )


def _augment_test_command_scope(
    cmd: str, *, candidate_diff: str,
) -> str:
    """Widen the ``-pl`` scope of a Maven test_command to cover the
    module the candidate diff touches.

    * If ``cmd`` has a ``-pl`` argument, union its module list with
      modules parsed from ``candidate_diff`` (unified diff), then rewrite
      the argument.
    * If ``cmd`` has no ``-pl`` at all, it's already reactor-wide —
      leave it alone.

    The result is a functional superset of the input: every test the
    original would have run still runs, plus the agent's tests that
    were previously out of scope.
    """
    pl_match = _PL_ARG_RE.search(cmd)
    if pl_match is None:
        # No -pl → reactor-wide; already covers everything. Just return.
        return cmd

    existing_modules = {m.strip() for m in pl_match.group(2).split(",") if m.strip()}
    diff_modules = _modules_from_diff(candidate_diff)
    union = sorted(existing_modules | diff_modules)

    # Rewrite -pl arg.
    new_pl_arg = f"{pl_match.group(1)}{','.join(union)}"
    new_cmd = cmd[:pl_match.start()] + new_pl_arg + cmd[pl_match.end():]

    return new_cmd


def _expand_organize_test_command(
    cmd: str,
    test_bearing_modules: set[str],
) -> str:
    """Expand the organize command to cover all reported test modules.

    This is organize-side evidence assembly, not verify augmentation. The
    manifest already reported ``modules`` and ``test_scope_full_files``;
    this helper turns that evidence into the exact command verify will
    execute. ``-am`` is added only when the manifest's own ``-pl`` scope
    omitted test-bearing modules and therefore could not have exercised
    them without building reactor dependencies.
    """
    if not test_bearing_modules:
        return cmd

    pl_match = _PL_ARG_RE.search(cmd)
    if pl_match is None:
        # No -pl means root reactor; already covers every module.
        return cmd

    existing = {m.strip() for m in pl_match.group(2).split(",") if m.strip()}
    missing = sorted(test_bearing_modules - existing)
    union = sorted(existing | set(missing))
    new_pl_arg = f"{pl_match.group(1)}{','.join(union)}"
    if missing:
        new_cmd = cmd[:pl_match.start()] + new_pl_arg + cmd[pl_match.end():]
    else:
        new_cmd = cmd

    # Multi-module ``-pl`` builds commonly need reactor dependencies.
    # Add ``-am`` only after the manifest's own evidence already selected
    # more than one module; verify augmentation remains responsible for
    # widening candidate modules only.
    if len(union) > 1 and not re.search(r"(?:^|\s)-am(?:\s|$)", new_cmd):
        insert_at = pl_match.start() + len(new_pl_arg)
        new_cmd = new_cmd[:insert_at] + " -am" + new_cmd[insert_at:]
    return new_cmd


def _extract_docker_layers(verify_report, sandbox, snapshot) -> dict:
    """Prefer the docker_layers embedded in verify_report (that's the
    authoritative snapshot at verify time). Fall back to what we can
    query directly on the sandbox."""
    if verify_report and hasattr(verify_report, "docker_layers") and verify_report.docker_layers:
        return dict(verify_report.docker_layers)
    return {
        "image_tag":         sandbox.image_tag if sandbox else "",
        "container_short_id": sandbox.container.short_id if sandbox else "",
        "base_commit":       snapshot.base_commit if snapshot else "",
    }


def _emit_gate_summary(tel: TelemetryLog, report) -> None:
    intersection_n = len(report.pred_gold_f2p_intersection)
    gold_n = len(report.gold_fail_to_pass)
    recall_pct = (100.0 * intersection_n / gold_n) if gold_n else None
    tel.emit(
        "verify.gate",
        valid=report.validity_gate,
        reasons=list(report.validity_notes),
        pred_f2p_n=len(report.fail_to_pass),
        pred_p2p_n=len(report.pass_to_pass),
        gold_f2p_n=gold_n,
        gold_p2p_n=len(report.gold_pass_to_pass),
        intersection_n=intersection_n,
        recall_pct=recall_pct,
    )


def _prune_candidate_bad_tests(
    candidate_diff: str,
    bad_test_ids: set[str],
) -> str:
    """Remove candidate-added FAIL→FAIL or PASS→FAIL methods from a diff."""
    bad_methods: dict[str, set[str]] = {}
    for test_id in bad_test_ids:
        if "#" not in test_id:
            continue
        class_name, method_name = test_id.rsplit("#", 1)
        simple_class = class_name.rsplit(".", 1)[-1]
        bad_methods.setdefault(simple_class, set()).add(method_name)

    if not candidate_diff or not bad_methods:
        return candidate_diff

    def prune_block(block: str) -> str | None:
        class_match = re.search(
            r"^\+\s*(?:public\s+|final\s+|abstract\s+)*class\s+(\w+)",
            block,
            flags=re.MULTILINE,
        )
        path_match = re.search(r"^\+\+\+\s+b/(\S+)", block, flags=re.MULTILINE)
        path_stem = Path(path_match.group(1)).stem if path_match else ""
        simple_class = class_match.group(1) if class_match else path_stem
        methods = bad_methods.get(simple_class, set())

        lines = block.splitlines(keepends=True)
        out: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if methods and line.startswith("+") and "(" in line:
                match = re.search(r"public\s+\S+\s+(\w+)\s*\(", line)
                if match and match.group(1) in methods:
                    while out and out[-1].startswith("+") and (
                        "@Test" in out[-1]
                        or "@org.junit.Test" in out[-1]
                        or "@Ignore" in out[-1]
                    ):
                        out.pop()
                    depth = 0
                    found_open = False
                    while i < len(lines):
                        current = lines[i]
                        if current.startswith("+"):
                            if "{" in current:
                                depth += current.count("{")
                                found_open = True
                            if "}" in current:
                                depth -= current.count("}")
                            if found_open and depth <= 0:
                                i += 1
                                break
                        i += 1
                    continue
            out.append(line)
            i += 1

        has_added_content = any(
            line.startswith("+") and not line.startswith("+++")
            for line in out
        )
        if not has_added_content:
            return None
        return "".join(out)

    blocks = re.split(r"(?=^diff --git )", candidate_diff, flags=re.MULTILINE)
    pruned = [prune_block(block) for block in blocks]
    return "".join(b for b in pruned if b is not None)


def _render_plan(case, dual, split, stub_llm, use_agent_loop, use_setup, use_organize,
                 workspace, dry_run) -> None:
    t = Table(title=f"SWE-BREW · enhance plan for {case.case_id}")
    t.add_column("field")
    t.add_column("value", overflow="fold")
    t.add_row("repo_short",       case.repo_short)
    t.add_row("mr_id",            case.mr_id)
    t.add_row("prd chars",        str(len(case.prd_text)))
    t.add_row("base / head",      f"{dual.base_commit[:12]}  →  {dual.head_commit[:12]}   ({dual.method})")
    t.add_row("code_patch (git)", f"{len(split.code_patch):>8d} chars  {len(split.code_files):>3d} files")
    t.add_row("test_patch (git)", f"{len(split.test_patch):>8d} chars  {len(split.test_files):>3d} files")
    if stub_llm:
        cand_src = "STUB (gold test_patch as candidate)"
    elif use_agent_loop:
        stages_desc = [
            "setup" if use_setup else "SKIP-setup",
            "organize" if use_organize else "SKIP-organize",
            "test-gen",
        ]
        cand_src = f"AGENT-LOOP (stages: {' → '.join(stages_desc)}, model={os.environ.get('SWE_BREW_MODEL','?')})"
    else:
        cand_src = f"LLM single-shot (model={os.environ.get('SWE_BREW_MODEL','?')})"
    t.add_row("candidate source", cand_src)
    t.add_row("workspace",        str(workspace))
    t.add_row("dry_run",          str(dry_run))
    console.print(t)


def _render_final_summary(out: Path, report, exception_msg: str | None) -> None:
    s = Table(title="run summary")
    s.add_column("field")
    s.add_column("value")
    if exception_msg:
        s.add_row("status", f"[red]failed[/red] · {exception_msg}")
    elif report is None:
        s.add_row("status", "[yellow]completed without verify report[/yellow]")
    else:
        s.add_row("validity_gate",       "✅" if report.validity_gate else "❌ " + " | ".join(report.validity_notes)[:200])
        s.add_row("pred F2P",            str(len(report.fail_to_pass)))
        s.add_row("pred P2P",            str(len(report.pass_to_pass)))
        s.add_row("gold F2P (dev)",      str(len(report.gold_fail_to_pass)))
        s.add_row("pred ∩ gold F2P",     str(len(report.pred_gold_f2p_intersection)))
        if report.gold_fail_to_pass:
            recall_pct = 100.0 * len(report.pred_gold_f2p_intersection) / len(report.gold_fail_to_pass)
            s.add_row("recall vs gold",  f"{recall_pct:.1f}%  ({len(report.pred_gold_f2p_intersection)}/{len(report.gold_fail_to_pass)})")
        for cfg, r in report.runs.items():
            s.add_row(cfg, f"tests={len(r.tests)}  exit={r.exit_code}  {r.ran_seconds:.1f}s")
    console.print(s)
    console.print(f"[green]wrote {out}/result.json + verify_report.json + telemetry.jsonl (+ per-stage archives)[/green]")


def _write_failed_manifest(
    out: Path, case, exception_msg: str, started_at: dt.datetime,
    t_start: float, *, snapshot=None,
) -> None:
    """Called from early-stage failures (snapshot / sandbox) to still
    emit a valid result.json — RepoLaunch's 'save always runs' invariant.
    """
    elapsed = time.perf_counter() - t_start
    manifest = build_result_json(
        case_id=case.case_id,
        completed=False,
        exception=exception_msg,
        duration_seconds=elapsed,
        docker_image_layers={"base_commit": snapshot.base_commit if snapshot else ""},
        started_at=started_at,
    )
    archive_final(out, manifest)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
