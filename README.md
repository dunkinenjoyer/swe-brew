# swe-brew

Reference-assisted benchmark test enhancement with deterministic six-cell
verification. Given an authorized repository and a historical change (exact
base commit, request text, reference source/test patches), swe-brew generates
candidate change-discriminating tests and verifies them against a six-cell
execution matrix before admitting the task.

## Prerequisites

You need to pre-install: **Git**, **Python >= 3.12**, and **Docker**
(running and reachable by the current user). For Java/Maven corpora, a host
JDK (17+ recommended) is used to prepare snapshots; tests run inside a
Docker container.

## Install

From source:

```bash
git clone <this-repo>
cd swe-brew
pip install -e .
```

Or install the wheel directly:

```bash
pip install swe_brew-0.1.0-py3-none-any.whl
```

To check the installation:

```bash
python -c "import swe_brew.benchmark.enhance as e; print(e.__file__)"
```

## Configure

Model access is environment-based; credentials stay out of every file.
swe-brew talks to any OpenAI-compatible endpoint:

```bash
export SWE_BREW_MODEL_BASE_URL=...   # any OpenAI-compatible endpoint
export SWE_BREW_MODEL_API_KEY=...
export SWE_BREW_MODEL=...            # e.g. gpt-4o, claude-sonnet-...
```

Optional web search for the generation agent (public knowledge only):

```bash
export TAVILY_API_KEY=...
```

Optional host overrides (sane defaults are used when unset):

```bash
export SWE_BREW_LOCAL_CLONE_ROOTS=/path/to/clones  # comma-separated; pre-cloned repos win over fetching
export SWE_BREW_WORKSPACE=/path/to/work            # snapshot worktrees + logs
export SWE_BREW_MAVEN_DIST=/path/to/apache-maven   # pre-provisioned Maven for air-gapped hosts
export SWE_BREW_REGISTRY_MIRROR=...                # docker registry mirror for restricted networks
```

## Run one task

Point the enhancer at a task directory that contains a ``case.json``
(the task id, the base commit, the reference code/test patches, and the
problem statement):

```bash
swe-brew-enhance /path/to/task --use-agent-loop
# equivalently: python -m swe_brew.benchmark.enhance /path/to/task --use-agent-loop
```

One invocation runs snapshot, setup, organize, bounded multi-turn
generation, deterministic six-cell verification, and the admission decision.
Results are written to ``<task>/enhanced/`` — a SWE-bench-Live-format
``result.json`` (``FAIL_TO_PASS`` / ``PASS_TO_PASS``), a
``verify_report.json``, and a ``telemetry.jsonl``.

Per-attempt pass@k diversity and a union merger aggregate the strongest
candidates across attempts.

## Monitor a batch

```bash
swe-brew-watch /path/to/run-root
# equivalently: python -m swe_brew.agent.watch /path/to/run-root
```

A live rich dashboard renders per-task stage, F2P/P2P counts, and admission
decisions as they land, driven only by journaled telemetry events.

## License

MIT — see [LICENSE](LICENSE). Third-party attributions (SWT-Bench,
RepoLaunch) are in [NOTICE](NOTICE).
