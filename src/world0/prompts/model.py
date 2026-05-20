"""Prompt configuration models for World 0 runtime prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


_VARIABLE_RE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")


@dataclass(frozen=True)
class PromptSpec:
    """A named prompt template with lightweight render metadata."""

    id: str
    template: str
    description: str = ""
    variables: tuple[str, ...] = field(default_factory=tuple)
    output: str = "text"
    enabled: bool = True
    schema_hint: dict[str, Any] | None = None

    def render(self, **values: Any) -> str:
        """Render ``{{variable}}`` placeholders with provided values."""
        missing = [name for name in self.variable_names() if name not in values]
        if missing:
            joined = ", ".join(missing)
            raise KeyError(f"Missing prompt variable(s) for {self.id}: {joined}")

        def replace(match: re.Match[str]) -> str:
            return str(values[match.group(1)])

        return _VARIABLE_RE.sub(replace, self.template)

    def variable_names(self) -> tuple[str, ...]:
        """Return declared plus discovered template variables."""
        names = list(self.variables)
        for match in _VARIABLE_RE.finditer(self.template):
            name = match.group(1)
            if name not in names:
                names.append(name)
        return tuple(names)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "enabled": self.enabled,
            "template": self.template,
            "description": self.description,
            "variables": list(self.variable_names()),
            "output": self.output,
        }
        if include_id:
            data["id"] = self.id
        if self.schema_hint is not None:
            data["schema_hint"] = self.schema_hint
        return data

    @classmethod
    def from_dict(cls, prompt_id: str, data: dict[str, Any]) -> "PromptSpec":
        return cls(
            id=prompt_id,
            template=str(data.get("template", "")),
            description=str(data.get("description", "")),
            variables=tuple(str(v) for v in data.get("variables", [])),
            output=str(data.get("output", "text")),
            enabled=bool(data.get("enabled", True)),
            schema_hint=data.get("schema_hint"),
        )


@dataclass(frozen=True)
class PromptValidationIssue:
    """A validation issue found in prompt configuration."""

    prompt_id: str
    message: str
    severity: str = "error"

    def render(self) -> str:
        return f"[{self.severity}] {self.prompt_id}: {self.message}"
