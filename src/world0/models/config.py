"""Per-operation model configuration for World 0."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODELS_CONFIG_NAME = "models.json"

DEFAULT_OPERATIONS: tuple[str, ...] = (
    "extraction",
    "query_extract",
    "learn_summary",
    "answer",
    "search_brief",
    "research_source",
    "research_report",
    "session_compaction",
    "agent_loop",
)


@dataclass(frozen=True)
class OperationModelSpec:
    """Model override for one World 0 operation."""

    operation: str
    provider: str = ""
    model: str = ""
    enabled: bool = True
    notes: str = ""

    def provider_model(self) -> str:
        if self.provider and self.model:
            return f"{self.provider}/{self.model}"
        return self.model

    def to_dict(self, *, include_operation: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "enabled": self.enabled,
            "provider": self.provider,
            "model": self.model,
            "notes": self.notes,
        }
        if include_operation:
            data["operation"] = self.operation
        return data

    @classmethod
    def from_dict(cls, operation: str, data: dict[str, Any]) -> "OperationModelSpec":
        return cls(
            operation=operation,
            provider=str(data.get("provider", "")).strip(),
            model=str(data.get("model", "")).strip(),
            enabled=bool(data.get("enabled", True)),
            notes=str(data.get("notes", "")).strip(),
        )


class OperationModelConfig:
    """Registry of per-operation model overrides."""

    def __init__(self, operations: tuple[str, ...] = DEFAULT_OPERATIONS) -> None:
        self._operations = tuple(operations)
        self._overrides: dict[str, OperationModelSpec] = {}

    def operations(self) -> list[str]:
        return list(self._operations)

    def set(self, spec: OperationModelSpec) -> None:
        if spec.operation not in self._operations:
            raise KeyError(f"Unknown model operation: {spec.operation}")
        self._overrides[spec.operation] = spec

    def clear(self, operation: str) -> None:
        if operation not in self._operations:
            raise KeyError(f"Unknown model operation: {operation}")
        self._overrides.pop(operation, None)

    def get(self, operation: str) -> OperationModelSpec | None:
        if operation not in self._operations:
            raise KeyError(f"Unknown model operation: {operation}")
        spec = self._overrides.get(operation)
        if spec and spec.enabled and spec.model:
            return spec
        return None

    def raw(self, operation: str) -> OperationModelSpec | None:
        if operation not in self._operations:
            raise KeyError(f"Unknown model operation: {operation}")
        return self._overrides.get(operation)

    def overrides(self) -> dict[str, OperationModelSpec]:
        return dict(self._overrides)

    def validate(self) -> list[str]:
        issues: list[str] = []
        for operation, spec in sorted(self._overrides.items()):
            if operation not in self._operations:
                issues.append(f"{operation}: unknown operation")
            if spec.enabled and not spec.model:
                issues.append(f"{operation}: enabled override requires model")
            if spec.provider and spec.provider not in {
                "openai",
                "anthropic",
                "azure-openai",
            }:
                issues.append(f"{operation}: unknown provider {spec.provider!r}")
        return issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "operations": {
                operation: spec.to_dict()
                for operation, spec in sorted(self._overrides.items())
            },
        }


def model_config_path(store_path: str | Path) -> Path:
    return Path(store_path).expanduser() / MODELS_CONFIG_NAME


def load_operation_model_config(*paths: str | Path) -> OperationModelConfig:
    config = OperationModelConfig()
    for path in paths:
        p = Path(path).expanduser()
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for operation, item in data.get("operations", {}).items():
            if operation not in config.operations():
                continue
            config.set(OperationModelSpec.from_dict(operation, item))
    return config


def save_operation_model_config(
    path: str | Path,
    config: OperationModelConfig,
) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
