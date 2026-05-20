"""Read and write prompt override configuration files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from world0.prompts.model import PromptSpec, PromptValidationIssue
from world0.prompts.registry import PromptRegistry


PROMPTS_CONFIG_NAME = "prompts.json"


def prompt_config_path(store_path: str | Path) -> Path:
    return Path(store_path).expanduser() / PROMPTS_CONFIG_NAME


def load_prompt_registry(
    *paths: str | Path,
    ignore_missing: bool = True,
) -> PromptRegistry:
    """Load the default registry plus overrides from one or more JSON files."""
    registry = PromptRegistry()
    for path in paths:
        p = Path(path).expanduser()
        if not p.exists():
            if ignore_missing:
                continue
            raise FileNotFoundError(p)
        data = json.loads(p.read_text(encoding="utf-8"))
        for prompt_id, item in data.get("prompts", {}).items():
            if prompt_id not in registry.ids():
                continue
            base = registry.default(prompt_id)
            merged: dict[str, Any] = base.to_dict(include_id=False)
            merged.update(item)
            if "template" in item and "variables" not in item:
                merged["variables"] = []
            registry.set_override(PromptSpec.from_dict(prompt_id, merged))
    return registry


def save_prompt_overrides(path: str | Path, registry: PromptRegistry) -> None:
    """Persist only user overrides to a prompt config file."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "prompts": {
            prompt_id: spec.to_dict(include_id=False)
            for prompt_id, spec in sorted(registry.overrides().items())
        },
    }
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_prompt_config(registry: PromptRegistry) -> dict[str, Any]:
    """Return a full editable prompt config document."""
    return {
        "version": 1,
        "prompts": {
            spec.id: spec.to_dict(include_id=False)
            for spec in registry.all()
        },
    }


def validate_prompt_config(path: str | Path) -> list[PromptValidationIssue]:
    registry = load_prompt_registry(path, ignore_missing=False)
    return registry.validate()
