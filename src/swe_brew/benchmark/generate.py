"""Generate a candidate ``test_patch`` from ``(PRD, gold_code_patch)``.

Two strategies:

- ``StubGenerator`` — returns the gold ``test_patch`` verbatim. Used
  for POC dry-runs where we want to prove the verify plumbing works
  without spending an LLM call.

- ``LLMGenerator`` — talks to the the configured model gateway OpenAI-compatible proxy
  (via ``httpx`` — we deliberately do NOT depend on ``openai`` or
  ``litellm`` at runtime for the POC, since the proxy is stable OpenAI-v1
  shape and we want a small dep footprint).

Both return a raw unified-diff string. The caller is responsible for
applying it inside the container.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from swe_brew.agent.prompt import safe_render
from swe_brew.benchmark.models import SplitPatches, SweBrewCase

__all__ = [
    "Generator",
    "StubGenerator",
    "LLMGenerator",
    "GenerationResult",
    "GenerationError",
    "load_prompt_template",
]


class GenerationError(RuntimeError):
    pass


@dataclass
class GenerationResult:
    """Return value of ``Generator.generate``."""

    candidate_patch: str        # unified diff
    is_stub: bool               # True iff produced by the stub
    model: str = ""             # model name if LLM
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw_response: str = ""      # for auditing


class Generator(Protocol):
    def generate(self, case: SweBrewCase, split: SplitPatches) -> GenerationResult: ...


# ─── stub ────────────────────────────────────────────────────────────────────


class StubGenerator:
    """Return the gold test_patch verbatim; used for plumbing tests.

    Rationale: for cases that ship real gold tests, feeding those tests
    as the "candidate" is the ideal fixed point for validating the
    6-way runner + candidate acceptance gate — they should pass validity by
    construction.
    """

    def generate(self, case: SweBrewCase, split: SplitPatches) -> GenerationResult:
        return GenerationResult(candidate_patch=split.test_patch, is_stub=True)


# ─── LLM ─────────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = Path(__file__).parent / "prompts" / "java_test_gen.md"


def load_prompt_template() -> str:
    return _PROMPT_TEMPLATE.read_text(encoding="utf-8")


def _extract_diff(text: str) -> str:
    """Pull a unified-diff out of an LLM response.

    Preference order:
      1. A fenced ```diff ... ``` block.
      2. Everything from the first ``diff --git`` line onward.
      3. The raw text (last resort — will probably fail ``git apply``).
    """
    m = re.search(r"```diff\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"(diff --git .+)", text, re.DOTALL)
    if m:
        return m.group(1)
    return text.strip()


@dataclass
class LLMGenerator:
    """OpenAI-compat client for the the configured model gateway proxy.

    Reads config from env by default:
      - SWE_BREW_MODEL_BASE_URL          (e.g. https://models.example.com/api/openai/v1)
      - SWE_BREW_MODEL_API_KEY           (sub-AK)
      - SWE_BREW_MODEL             (e.g. bailian/deepseek-v4-pro)
      - SWE_BREW_MODEL_FALLBACK    (e.g. gpt-5.6-terra)
      - SWE_BREW_REASONING_EFFORT  low | medium | high  (reasoning models only)
      - SWE_BREW_VERBOSITY         low | medium | high  (reasoning models only)

    The reasoning knobs are silently dropped for models that don't
    understand them (the proxy accepts+ignores; verified on gpt-4.1).
    """

    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    fallback_model: str | None = None
    reasoning_effort: str | None = None
    verbosity: str | None = None
    max_tests: int = 6
    max_prd_chars: int = 16000    # deepseek-v4-pro has generous context
    max_patch_chars: int = 60000
    max_output_tokens: int = 16000  # candidate diff can be a few KB; reasoning tokens add more
    timeout: float = 300.0        # reasoning models are slower

    def __post_init__(self) -> None:
        self.base_url = self.base_url or os.environ.get("SWE_BREW_MODEL_BASE_URL", "").rstrip("/")
        self.api_key = self.api_key or os.environ.get("SWE_BREW_MODEL_API_KEY", "")
        self.model = self.model or os.environ.get("SWE_BREW_MODEL", "bailian/deepseek-v4-pro")
        self.fallback_model = self.fallback_model or os.environ.get("SWE_BREW_MODEL_FALLBACK", "gpt-5.5-0424-global")
        self.reasoning_effort = self.reasoning_effort or os.environ.get("SWE_BREW_REASONING_EFFORT", "high")
        self.verbosity = self.verbosity or os.environ.get("SWE_BREW_VERBOSITY", "medium")
        if not self.base_url or not self.api_key:
            raise GenerationError("SWE_BREW_MODEL_BASE_URL and SWE_BREW_MODEL_API_KEY must be set (see .env / .env.example)")

    def _call(self, model: str, prompt: str) -> tuple[str, dict]:
        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You emit unified diffs only. Never emit prose. Never emit multiple diffs."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_output_tokens,
        }
        # Reasoning knobs — the proxy accepts+ignores for non-reasoning
        # models, so this is safe to always include.
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.verbosity:
            payload["verbosity"] = self.verbosity

        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
            )
        if r.status_code >= 400:
            raise GenerationError(f"the configured model gateway {model} → HTTP {r.status_code}: {r.text[:400]}")
        try:
            data = r.json()
        except Exception as e:
            raise GenerationError(f"the configured model gateway {model} returned non-JSON: {r.text[:400]}") from e
        if "choices" not in data:
            raise GenerationError(f"the configured model gateway {model} response missing 'choices': {data}")
        content = data["choices"][0]["message"]["content"]
        return content, data.get("usage", {})

    def generate(self, case: SweBrewCase, split: SplitPatches) -> GenerationResult:
        # Compose the prompt from template.
        prd = case.prd_text[: self.max_prd_chars]
        code = split.code_patch[: self.max_patch_chars]
        changed_list = "\n".join(f"- {p}" for p in case.changed_files[:80])
        prompt = safe_render(
            load_prompt_template(),
            prd=prd,
            prd_len=len(prd),
            code_patch=code,
            changed_count=len(case.changed_files),
            changed_files_list=changed_list,
            repo_short=case.repo_short,
            max_tests=self.max_tests,
        )

        # Try primary → fallback.
        errors: list[str] = []
        for candidate_model in (self.model, self.fallback_model):
            if not candidate_model:
                continue
            try:
                content, usage = self._call(candidate_model, prompt)
            except GenerationError as e:
                errors.append(f"{candidate_model}: {e}")
                continue
            diff = _extract_diff(content)
            return GenerationResult(
                candidate_patch=diff,
                is_stub=False,
                model=candidate_model,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                raw_response=content,
            )
        raise GenerationError("all LLM candidates failed: " + " | ".join(errors))
