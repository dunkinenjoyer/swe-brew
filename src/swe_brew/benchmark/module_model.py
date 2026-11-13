"""Validated Maven module model for organize -> generate -> verify.

The previous implementation inferred test-host modules from changed-file
prefixes and repo-name conventions. That produced a changed-prefix module in
the verify reactor even when organize never validated it with the ``test``
goal. This module makes the contract explicit:

* ``all_modules`` comes from the actual reactor inventory emitted by
  organize.
* ``validated_modules`` are exactly the reactor modules covered by the
  organize command that was executed successfully.
* Test generation and verify-command augmentation may only use modules
  in that validated set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PL_ARG_RE = re.compile(r"(?:^|\s)-pl(?:\s+|=)(\S+)")


@dataclass(frozen=True)
class ValidatedModuleSet:
    """Module inventory plus the subset proven by organize's test run."""

    all_modules: tuple[str, ...]
    validated_modules: tuple[str, ...]
    method: str

    @classmethod
    def from_organize(cls, modules: list[str], test_command: str) -> ValidatedModuleSet:
        """Build the model from organize's submitted inventory + command."""
        normalized = tuple(sorted({m.strip() for m in modules if m and m.strip()}))
        pl = _parse_pl_modules(test_command)
        if pl:
            # ``-pl`` is only evidence when every requested module exists
            # in the reactor inventory organize reported. Otherwise fail
            # closed instead of trusting a hallucinated module name.
            selected = tuple(sorted(pl & set(normalized)))
            if selected:
                method = "pl"
            elif normalized:
                method = "pl-no-match"
            else:
                method = "pl-unverified"
        elif normalized:
            # No -pl means the root reactor was run, so every listed module
            # was in the executed reactor.
            selected = normalized
            method = "root-reactor"
        else:
            selected = ()
            method = "unknown"
        return cls(
            all_modules=normalized,
            validated_modules=selected,
            method=method,
        )

    @classmethod
    def from_executed_reports(
        cls,
        modules: list[str],
        report_modules: set[str],
    ) -> ValidatedModuleSet:
        """Build the model from modules that actually emitted Surefire reports.

        This is stronger than parsing ``-pl``: a module can be syntactically
        present in a root-reactor command but compile-fail and emit no report.
        """
        normalized = tuple(sorted({m.strip() for m in modules if m and m.strip()}))
        selected = tuple(sorted(report_modules & set(normalized)))
        method = "executed-reports" if selected else "no-reports"
        return cls(
            all_modules=normalized,
            validated_modules=selected,
            method=method,
        )

    def contains(self, module: str) -> bool:
        return module in self.validated_modules


def _parse_pl_modules(test_command: str) -> set[str]:
    modules: set[str] = set()
    for match in _PL_ARG_RE.finditer(test_command or ""):
        modules.update(m.strip() for m in match.group(1).split(",") if m.strip())
    return modules
