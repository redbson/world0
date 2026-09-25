"""Layer boundaries — the conceptual core must not depend on the agent layer.

AGENTS.md (Rule 6, "Preserve boundary clarity") separates the cognitive
core (concepts, relations, context, activation, projection) from memory,
workflow and tool execution.  ``world0.agents`` is that adjacent system:
sessions, failure recovery, CLIs and web UI.  This test walks the import
statements of every core module so the boundary is checked, not merely
documented.

The rules encode the *current* dependency graph so a regression is
caught the moment a core module reaches upward:

* nothing outside ``world0.agents`` imports ``world0.agents``;
* the pure core packages import nothing from the LLM-facing or
  presentation packages;
* ``world0.world`` (the facade) may use extraction/prompts (text ingest
  needs an extractor) but never the agent layer.
"""

from __future__ import annotations

import ast
from pathlib import Path

import world0

SRC = Path(world0.__file__).parent

# Packages that make up the conceptual core.  They may import each other
# and ``world0.schemas`` / ``world0.core``; nothing from FORBIDDEN_FOR_CORE.
CORE_PACKAGES = {
    "schemas",
    "core",
    "concepts",
    "relations",
    "dynamics",
    "projection",
    "perspectives",
    "communities",
    "store",
    "metrics",
}

# Upper layers: LLM access, extraction, agent runtime, presentation.
FORBIDDEN_FOR_CORE = {
    "agents",
    "llm",
    "extraction",
    "prompts",
    "models",
    "visualization",
    "world",
    "spaces",
}

AGENT_LAYER = "agents"


def _top_level_imports(path: Path) -> set[str]:
    """Second-level package names imported from ``world0`` by one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] == "world0" and len(parts) > 1:
                found.add(parts[1])
            elif parts[0] == "world0":
                # ``from world0 import X`` — X may be a subpackage
                found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "world0" and len(parts) > 1:
                    found.add(parts[1])
    return found


def _modules_of(package: str) -> list[Path]:
    return sorted((SRC / package).rglob("*.py"))


def _package_names() -> set[str]:
    return {p.name for p in SRC.iterdir() if p.is_dir() and not p.name.startswith("_")}


class TestCoreDoesNotReachUp:
    def test_all_packages_are_classified(self):
        """Every top-level package is either core, forbidden-for-core or
        one of the known LLM-facing helpers, so a new package cannot slip
        past this test unclassified."""
        known = CORE_PACKAGES | FORBIDDEN_FOR_CORE | {"sources"}
        assert _package_names() <= known, _package_names() - known

    def test_core_packages_import_only_core(self):
        violations: list[str] = []
        for package in sorted(CORE_PACKAGES):
            for module in _modules_of(package):
                bad = _top_level_imports(module) & FORBIDDEN_FOR_CORE
                if bad:
                    rel = module.relative_to(SRC)
                    violations.append(f"{rel} imports world0.{sorted(bad)}")
        assert not violations, "\n".join(violations)

    def test_nothing_outside_agents_imports_agents(self):
        violations: list[str] = []
        for module in SRC.rglob("*.py"):
            rel = module.relative_to(SRC)
            if rel.parts[0] == AGENT_LAYER:
                continue
            if AGENT_LAYER in _top_level_imports(module):
                violations.append(str(rel))
        assert not violations, violations

    def test_world_facade_does_not_import_agents_or_presentation(self):
        violations: list[str] = []
        for module in _modules_of("world"):
            bad = _top_level_imports(module) & {AGENT_LAYER, "llm", "spaces"}
            if bad:
                violations.append(f"{module.relative_to(SRC)} imports {sorted(bad)}")
        assert not violations, "\n".join(violations)
