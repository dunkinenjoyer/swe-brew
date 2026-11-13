"""Docker image build + exec wrapper for a snapshot.

Why not use RepoLaunch's ``SetupRuntime`` directly
--------------------------------------------------
``SetupRuntime`` (``launch/launch/core/runtime.py``) is a great API but
its ``_start_container`` hard-codes ``https://github.com/{repo}`` for the
in-container clone (``launch/launch/core/platforms/linux.py:344-352``)
and has no path for private the configured Git host. Rather than subclass and
monkey-patch, we build our own thin wrapper around ``docker-py``: it's
maybe 80 lines and gives us total control over the base_commit checkout
and the Maven cache.

Container layout
----------------
- Base image: ``eclipse-temurin:8-jdk-focal`` (Java 8 by default;
  can be overridden per-case). RepoLaunch's ``JavaHandler``
  uses the same family.
- ``/testbed`` — the snapshot's working copy is mounted (bind-mount)
  read-write so tests + Surefire XMLs appear immediately in the host
  workspace.
- ``/root/.m2`` — mounted from a shared Maven cache so we don't
  re-download deps every run.

For POC we do NOT push the image to any registry; images stay local to
Docker Desktop and are tagged ``swe_brew/<repo>:<mr_id>``.
"""

from __future__ import annotations

import os
import shlex
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import docker
from docker.models.containers import Container

from swe_brew.benchmark.runtime import runtime_spec
from swe_brew.benchmark.snapshot import Snapshot

__all__ = [
    "ContainerError",
    "TestSandbox",
    "CommandResult",
    "get_client",
    "start_sandbox",
    "EXEC_TIMEOUT_EXIT_CODE",
]


# POSIX-conventional exit code for "command timed out" — returned when
# ``TestSandbox.exec`` fires its wall-clock kill on a hung docker exec.
# Matches ``coreutils timeout``'s default and lets callers distinguish
# timeout kills from real command failures.
EXEC_TIMEOUT_EXIT_CODE = 124


class ContainerError(RuntimeError):
    pass


@dataclass
class CommandResult:
    """Return value of ``TestSandbox.exec``."""

    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def timed_out(self) -> bool:
        """True if ``exec`` fired its wall-clock kill. Distinct from a
        legitimate command failure at ``exit_code=124`` — check ``stderr``
        for the ``[timed out after Ns]`` sentinel."""
        return self.exit_code == EXEC_TIMEOUT_EXIT_CODE and "timed out" in self.stderr


# ─── docker client ───────────────────────────────────────────────────────────


def get_client() -> docker.DockerClient:
    """Return a docker-py client. Raises with a friendly message if the daemon is unreachable."""
    try:
        client = docker.from_env()
        client.ping()
        return client
    except Exception as e:  # noqa: BLE001
        raise ContainerError(
            f"cannot reach Docker daemon: {e!s}. "
            f"Start Docker Desktop (or set DOCKER_HOST) and retry.",
        ) from e


# ─── the sandbox ─────────────────────────────────────────────────────────────


@dataclass
class TestSandbox:
    """A live container for a snapshot. Prefer ``start_sandbox()`` factory."""

    container: Container
    snapshot: Snapshot
    image_tag: str
    workdir: str = "/testbed"

    # ─── exec ─────────────────────────────────────────────────────────────

    def exec(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        """Run ``bash -lc`` inside the container with an enforceable timeout.

        Design (ported from RepoLaunch's ``LinuxRuntime.send_command``):

        * A background thread drains the exec socket into a byte buffer.
        * The main thread joins the drain with the wall-clock ``timeout``.
          If the drain doesn't finish, we've hit the ceiling.
        * On timeout, kill the command's process group via a canary-PID
          file: the exec wrapper writes ``$$`` (bash's own PID, which
          equals its PGID because we launch under ``setsid --wait``) to
          ``/tmp/.swe-brew-exec-<uuid>`` BEFORE running the user command;
          a second docker exec reads that PGID and issues
          ``kill -TERM -PGID`` then ``kill -KILL -PGID``.

        ``timeout=None`` waits forever (used by verify pipeline legs).
        ``timeout=N`` enforces an N-second ceiling; the returned
        :class:`CommandResult` has ``exit_code=EXEC_TIMEOUT_EXIT_CODE``
        (124, coreutils convention) plus a ``[timed out ...]`` stderr
        sentinel so callers can distinguish real command failures from
        wall-clock kills.

        Why ``setsid --wait``. Without ``--wait``, setsid returns
        immediately after fork+setsid(); docker sees the exec's main
        pid gone and closes the multiplexed stdout socket ~3-6 s later,
        cutting the drain off while the target command is still running.
        Symptom: every mvn verify leg logs ``ran_seconds ≈ 3.6 s``,
        ``exit=0``, empty ``stdout_tail`` regardless of what mvn is
        actually doing. ``--wait`` blocks the setsid parent until the
        child exits, keeping the exec stream open for the full command
        duration. The PGID from the canary still resolves the same
        session, so :meth:`_kill_process_group_from_canary` is unaffected.
        """
        api = self.container.client.api

        # Canary path — per-exec unique. Bash writes its PID here BEFORE
        # exec'ing the user command; if we need to kill, we read this
        # file to get the PGID. Cleanup happens in the kill path
        # (``rm -f`` after the KILL) or by container teardown.
        canary = f"/tmp/.swe-brew-exec-{uuid.uuid4().hex}"
        inner = f"echo $$ > {canary}; exec bash -lc {shlex.quote(cmd)}"
        wrapped = "setsid --wait bash -c " + shlex.quote(inner)

        exec_id = api.exec_create(
            self.container.id,
            ["bash", "-lc", wrapped],
            workdir=self.workdir,
            tty=False,
        )["Id"]

        # Start the stream and drain in a background thread so the main
        # thread can enforce a wall-clock join.
        stream = api.exec_start(exec_id, detach=False, stream=True, demux=False)
        output_buf = bytearray()
        stop_flag = threading.Event()

        def _drain() -> None:
            try:
                for chunk in stream:
                    if stop_flag.is_set():
                        break
                    if chunk:
                        output_buf.extend(chunk)
            except Exception:                    # noqa: BLE001
                # Socket disconnect / broken pipe when we kill the exec
                # from another thread. Expected on the timeout path;
                # swallow so the join can complete.
                return

        drain_thread = threading.Thread(
            target=_drain,
            name=f"exec-drain-{exec_id[:8]}",
            daemon=True,
        )
        drain_thread.start()
        drain_thread.join(timeout=timeout)

        timed_out = False
        if timeout is not None and drain_thread.is_alive():
            timed_out = True
            stop_flag.set()
            # Kill the process group via a second docker exec that
            # reads the canary PGID. Best-effort; never let this
            # failure kill the caller.
            self._kill_process_group_from_canary(canary, cmd)
            # Give the drain a bounded window to finish reading residual
            # output from the killed process — 5s matches RepoLaunch's
            # ``kill_timeout`` in linux.py:189.
            drain_thread.join(timeout=5)

        info = api.exec_inspect(exec_id)
        exit_code = info.get("ExitCode")
        if timed_out:
            exit_code = EXEC_TIMEOUT_EXIT_CODE
        elif exit_code is None:
            # ExitCode is None when the exec is still running (e.g. we
            # killed the drain but the container process didn't actually
            # exit). Report as timeout to be safe — a legitimate
            # completion would have set ExitCode.
            exit_code = EXEC_TIMEOUT_EXIT_CODE if timed_out else 0

        output = bytes(output_buf).decode(errors="replace")
        stderr = ""
        if timed_out:
            stderr = (
                f"[timed out after {timeout}s; process group killed via "
                f"{canary}]"
            )
        return CommandResult(
            exit_code=int(exit_code),
            stdout=output,
            stderr=stderr,
        )

    def _kill_process_group_from_canary(self, canary: str, cmd: str) -> None:
        """Send SIGTERM then SIGKILL to the PGID recorded in ``canary``.

        The canary file holds a single integer — the PGID of the
        wrapper bash that ran the user command via ``setsid --wait``.
        ``kill -TERM -PGID`` signals every process in the group; a
        follow-up ``kill -KILL -PGID`` after a short grace period force-
        kills anything that ignored SIGTERM.

        Best-effort. If the container is unresponsive or the canary was
        never written, we swallow the failure — the goal is to unblock
        subsequent execs, not to be perfectly clean.
        """
        api = self.container.client.api
        # SIGTERM first (allows Java shutdown hooks / mvn to write
        # residual output), 0.5s grace, then SIGKILL, then clean up
        # the canary itself. ``|| true`` so any missing-canary case
        # doesn't fail the outer bash. Also try a targeted pkill of
        # the user command literal in case the process group escaped
        # (e.g. a daemon that re-parented itself).
        kill_script = (
            f"pgid=$(cat {shlex.quote(canary)} 2>/dev/null || true); "
            f"if [ -n \"$pgid\" ]; then "
            f"  kill -TERM -\"$pgid\" 2>/dev/null || true; "
            f"  sleep 0.5; "
            f"  kill -KILL -\"$pgid\" 2>/dev/null || true; "
            f"fi; "
            # Belt-and-suspenders: also try to reap any lingering mvn
            # forks by command signature. Harmless on the happy path.
            f"pkill -KILL -f {shlex.quote(cmd[:100])} 2>/dev/null || true; "
            f"rm -f {shlex.quote(canary)} 2>/dev/null || true"
        )
        try:
            kill_id = api.exec_create(
                self.container.id,
                ["bash", "-c", kill_script],
                workdir=self.workdir,
                tty=False,
            )["Id"]
            # Bounded drain — kill script should finish in <5s. If it
            # doesn't, something is very wrong with the daemon; give
            # up rather than block the caller further.
            kill_stream = api.exec_start(kill_id, detach=False, stream=True, demux=False)
            kill_start = time.monotonic()
            try:
                for _chunk in kill_stream:
                    if time.monotonic() - kill_start > 10:
                        break
            except Exception:                    # noqa: BLE001
                pass
        except Exception:                        # noqa: BLE001
            # Kill exec creation itself failed. Container may be in a
            # weird state. The caller's wall-clock kill already set
            # ``timed_out=True``, so they'll surface a TIMEOUT result;
            # subsequent execs may still be usable if the daemon
            # recovers, or the case will be torn down cleanly by the
            # outer orchestrator.
            pass

    def apply_patch(self, patch_text: str) -> CommandResult:
        """Apply a unified-diff patch to the snapshot **on the host**.

        The snapshot's ``work_dir`` is bind-mounted at ``/testbed`` in the
        container, so a change made here is instantly visible to the
        running Maven build inside. We deliberately do NOT shell into the
        container for patch application, because:

        * stock ``eclipse-temurin`` images ship no ``git`` binary
          (previously observed as ``bash: git: command not found``);
        * ``echo <base64>``-into-shell hit ARG_MAX for the 289KB
          ``code_patch`` in this corpus (``exec /usr/bin/bash: argument
          list too long``).

        Applying on the host uses the system's own ``git`` and streams
        the patch on stdin — no argv limit, no dependency on the
        container image.
        """
        import subprocess
        if not patch_text.strip():
            return CommandResult(exit_code=0, stdout="", stderr="")
        proc = subprocess.run(
            ["git", "-C", str(self.snapshot.work_dir),
             "apply", "--index", "--recount", "--whitespace=nowarn", "-"],
            input=patch_text, capture_output=True, text=True, timeout=120,
        )
        return CommandResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

    def reset_working_tree(self) -> CommandResult:
        """Hard-reset the host worktree to the snapshot commit.

        Same rationale as ``apply_patch``: git runs on the host, changes
        show through the ``/testbed`` bind-mount, no container shell
        required. We also clean ``target/`` explicitly because Maven
        artefacts (esp. surefire reports) shouldn't leak between legs.
        """
        import subprocess
        r = subprocess.run(
            ["git", "-C", str(self.snapshot.work_dir),
             "reset", "--hard", self.snapshot.base_commit],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0:
            clean = subprocess.run(
                ["git", "-C", str(self.snapshot.work_dir), "clean", "-fdx"],
                capture_output=True, text=True, timeout=60,
            )
            if clean.returncode != 0:
                subprocess.run(
                    ["sudo", "-n", "git", "-C", str(self.snapshot.work_dir),
                     "clean", "-fdx"],
                    capture_output=True, text=True, timeout=60,
                )
        return CommandResult(exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr)

    def clear_surefire_reports(self) -> CommandResult:
        """Delete Surefire/Failsafe XMLs from inside the container.

        Maven runs as root in the container and creates reports owned by
        root on the host bind mount. Host-side cleanup can silently fail
        when the pipeline runs as an unprivileged user, so stale reports
        from a prior verify leg leak into the next parse and can suppress
        a genuinely passing ``pred_post`` leg.
        """
        find_cmd = (
            "find /testbed -type f -name 'TEST-*.xml' "
            "\\( -path '*/surefire-reports/*' "
            "-o -path '*/failsafe-reports/*' \\) "
            "-delete"
        )
        return self.exec(find_cmd, timeout=60)

    def stop(self) -> None:
        """Stop and remove the container."""
        try:
            self.container.stop(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.container.remove(force=True)
        except Exception:  # noqa: BLE001
            pass


# ─── factory ─────────────────────────────────────────────────────────────────


def _ensure_maven_dist() -> Path:
    """Return a host-side Maven distribution for bind-mounting into sandboxes.

    The stock ``eclipse-temurin`` image ships a JDK but no build tool, so
    ``mvn`` exits 127 inside the container. Maven is architecture-neutral
    (scripts + jars), so one host download serves every container platform.
    """
    # Operators with a pre-provisioned Maven (e.g. air-gapped hosts) point
    # SWE_BREW_MAVEN_DIST at the distribution root; it wins over the download.
    override = os.environ.get("SWE_BREW_MAVEN_DIST", "").strip()
    if override:
        root = Path(override)
        if not (root / "bin" / "mvn").exists():
            raise ContainerError(f"SWE_BREW_MAVEN_DIST has no bin/mvn: {override}")
        return root
    version = "3.9.16"
    root = Path.home() / ".swe-brew-tools" / f"apache-maven-{version}"
    complete = (root / "bin" / "mvn").exists() and (root / "lib" / f"maven-core-{version}.jar").exists()
    if not complete:
        import tarfile
        import tempfile
        import urllib.request

        url = (
            f"https://archive.apache.org/dist/maven/maven-3/{version}/binaries/"
            f"apache-maven-{version}-bin.tar.gz"
        )
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "maven.tgz"
            urllib.request.urlretrieve(url, archive)
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(root.parent)
    if not (root / "bin" / "mvn").exists():
        raise ContainerError(f"maven distribution incomplete at {root}; delete it and retry")
    return root


def _base_image_pull_candidates(base_image: str, mirror: str | None) -> list[str]:
    """Return ordered pull refs: direct first, configured mirror fallback second."""
    candidates = [base_image]
    if mirror:
        mirror = mirror.rstrip("/")
        candidates.append(f"{mirror}/{base_image}")
    return list(dict.fromkeys(candidates))


def _ensure_base_image(client: docker.DockerClient, base_image: str) -> None:
    """Pull ``base_image`` through direct Docker Hub then a mirror fallback.

    Docker Desktop registry-mirror settings can be operator-specific, so the
    pipeline cannot assume they are configured. This helper honours
    ``SWE_BREW_REGISTRY_MIRROR`` and falls back to a configured mirror for
    restricted-network hosts.
    """
    try:
        client.images.get(base_image)
        return
    except docker.errors.ImageNotFound:
        pass

    mirror = os.environ.get("SWE_BREW_REGISTRY_MIRROR", "docker.m.daocloud.io").strip()
    errors: list[str] = []
    for candidate in _base_image_pull_candidates(base_image, mirror):
        try:
            pulled = client.images.pull(candidate, platform="linux/amd64")
            if candidate != base_image:
                repo, tag = base_image.split(":", 1)
                pulled.tag(repository=repo, tag=tag)
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")

    raise ContainerError(
        "failed to pull base image "
        f"{base_image!r} (tried {len(errors)} candidate(s)): {' | '.join(errors)}",
    )


def start_sandbox(
    snapshot: Snapshot,
    *,
    image_prefix: str | None = None,
    # JDK 21 (not 8): some repositories publish transitive deps as Java 11
    # classfiles even for projects whose poms target 1.8. A JDK-8
    # JVM refuses to load them; a JDK-21 JVM reads all classfiles v52-65
    # and still honours <maven.compiler.target>1.8</>. Rosetta on ARM64
    # Mac runs the linux/amd64 image at ~40% native speed — acceptable
    # for POC, not for the main 540-attempt run.
    base_image: str | None = None,
    mem_limit: str | None = None,
    cpus: str | None = None,
    m2_cache: Path | None = None,
) -> TestSandbox:
    """Pull the base image, start a container with the snapshot bind-mounted.

    Non-goals for POC: building a repo-specific image with pre-downloaded
    deps. That's an optimization for the main experiment run. For the POC
    we bind-mount the snapshot dir into a stock JDK image; the Maven
    distribution itself is bind-mounted from the host (the stock image
    ships no build tool), and deps resolve at first run (cached via
    the ``m2_cache`` mount).

    Returns a ``TestSandbox`` — remember to ``.stop()`` when done.
    """
    if snapshot.work_dir is None or not snapshot.work_dir.exists():
        raise ContainerError(f"snapshot work_dir missing: {snapshot.work_dir!r}")

    if base_image is None:
        base_image = runtime_spec().base_image

    prefix = image_prefix or os.environ.get("SWE_BREW_IMAGE_PREFIX", "swe_brew")
    image_tag = f"{prefix}/{snapshot.repo_short}:{snapshot.base_commit[:12]}"

    mem = mem_limit or os.environ.get("SWE_BREW_DOCKER_MEM", "6g")
    cpu = cpus or os.environ.get("SWE_BREW_DOCKER_CPUS", "4")
    java_tool_options = runtime_spec().java_tool_options

    client = get_client()

    _ensure_base_image(client, base_image)

    # Shared maven cache (host-side dir).
    if m2_cache is None:
        m2_cache = Path.home() / ".swe-brew-m2-cache"
    m2_cache.mkdir(parents=True, exist_ok=True)

    maven = _ensure_maven_dist()
    volumes = {
        str(snapshot.work_dir.resolve()): {"bind": "/testbed", "mode": "rw"},
        str(m2_cache.resolve()):         {"bind": "/root/.m2", "mode": "rw"},
        str(maven.resolve()):            {"bind": "/opt/maven", "mode": "ro"},
    }

    container = client.containers.run(
        image=base_image,
        # Suffix with pid+timestamp so a retry after a crashed run never
        # collides with an orphaned container of the same logical task.
        name=(
            f"swe-brew-{os.environ.get('SWE_BREW_CONTAINER_SUFFIX', '')}"
            f"{snapshot.case_id[:28]}-{os.getpid()}-{int(time.time())}"
        ),
        command=["tail", "-f", "/dev/null"],  # keep alive; we exec into it
        detach=True,
        auto_remove=False,
        working_dir="/testbed",
        volumes=volumes,
        mem_limit=mem,
        nano_cpus=int(float(cpu) * 1e9),
        network_mode=os.environ.get("SWE_BREW_DOCKER_NETWORK", "bridge"),
        environment={
            "MAVEN_OPTS": "-Dmaven.artifact.threads=8 -Dorg.slf4j.simpleLogger.showDateTime=true",
            "JDK_JAVA_OPTIONS": java_tool_options,
            "LANG": "C.UTF-8",
            "PATH": "/opt/maven/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        },
        platform="linux/amd64",  # target x86_64; use Rosetta on ARM64 Mac
    )

    return TestSandbox(container=container, snapshot=snapshot, image_tag=image_tag)
