"""Prompt registry with layered runtime overrides."""

from __future__ import annotations

from collections.abc import Iterable

from world0.prompts.defaults import default_prompt_specs
from world0.prompts.model import PromptSpec, PromptValidationIssue


class PromptRegistry:
    """Registry for built-in prompts plus user overrides."""

    def __init__(
        self,
        defaults: Iterable[PromptSpec] | None = None,
        overrides: Iterable[PromptSpec] | None = None,
    ) -> None:
        self._defaults = {spec.id: spec for spec in (defaults or default_prompt_specs())}
        self._overrides: dict[str, PromptSpec] = {}
        for spec in overrides or []:
            self.set_override(spec)

    def set_override(self, spec: PromptSpec) -> None:
        if spec.id not in self._defaults:
            raise KeyError(f"Unknown prompt id: {spec.id}")
        self._overrides[spec.id] = spec

    def clear_override(self, prompt_id: str) -> None:
        self._overrides.pop(prompt_id, None)

    def get(self, prompt_id: str) -> PromptSpec:
        try:
            return self._overrides.get(prompt_id) or self._defaults[prompt_id]
        except KeyError as exc:
            raise KeyError(f"Unknown prompt id: {prompt_id}") from exc

    def default(self, prompt_id: str) -> PromptSpec:
        try:
            return self._defaults[prompt_id]
        except KeyError as exc:
            raise KeyError(f"Unknown prompt id: {prompt_id}") from exc

    def render(self, prompt_id: str, **values: object) -> str:
        return self.get(prompt_id).render(**values)

    def all(self) -> list[PromptSpec]:
        return [self.get(prompt_id) for prompt_id in sorted(self._defaults)]

    def ids(self) -> list[str]:
        return sorted(self._defaults)

    def overrides(self) -> dict[str, PromptSpec]:
        return dict(self._overrides)

    def is_overridden(self, prompt_id: str) -> bool:
        return prompt_id in self._overrides

    def validate(self) -> list[PromptValidationIssue]:
        issues: list[PromptValidationIssue] = []
        for prompt_id in self.ids():
            spec = self.get(prompt_id)
            if not spec.template.strip():
                issues.append(PromptValidationIssue(prompt_id, "template is empty"))
            if spec.output not in {"text", "json"}:
                issues.append(
                    PromptValidationIssue(
                        prompt_id,
                        f"output must be 'text' or 'json', got {spec.output!r}",
                    )
                )
            for name in spec.variable_names():
                marker = "{{" + name + "}}"
                loose_marker = "{{ " + name + " }}"
                if marker not in spec.template and loose_marker not in spec.template:
                    issues.append(
                        PromptValidationIssue(
                            prompt_id,
                            f"declared variable {name!r} is not used in template",
                            severity="warning",
                        )
                    )
        return issues
