"""Tests for configurable runtime prompts."""

import json

from world0.agents.skill import SkillRegistry, register_builtin_skills
from world0.extraction.extractor import ConceptExtractor
from world0.llm.base import LLMProvider
from world0.prompts import (
    PromptRegistry,
    PromptSpec,
    load_prompt_registry,
    prompt_config_path,
    save_prompt_overrides,
)


class RecordingLLM(LLMProvider):
    def __init__(self) -> None:
        self.system = ""
        self.user = ""

    def complete_json(self, system: str, user: str) -> str:
        self.system = system
        self.user = user
        return json.dumps({"concepts": [{"name": "python"}], "relations": []})


def test_prompt_spec_renders_template_variables():
    spec = PromptSpec(
        "demo.prompt",
        "Hello {{name}}, use {{ tool }}.",
        variables=("name", "tool"),
    )

    assert spec.render(name="World 0", tool="projection") == (
        "Hello World 0, use projection."
    )


def test_prompt_config_loads_template_override(tmp_path):
    path = prompt_config_path(tmp_path)
    path.write_text(
        json.dumps({
            "version": 1,
            "prompts": {
                "agent.answer.system": {
                    "template": "Custom answer prompt.",
                }
            },
        }),
        encoding="utf-8",
    )

    registry = load_prompt_registry(path)

    assert registry.render("agent.answer.system") == "Custom answer prompt."
    assert registry.is_overridden("agent.answer.system")


def test_save_prompt_overrides_writes_only_overrides(tmp_path):
    registry = PromptRegistry()
    registry.set_override(PromptSpec(
        "agent.answer.system",
        "Override only this prompt.",
    ))

    path = prompt_config_path(tmp_path)
    save_prompt_overrides(path, registry)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert set(data["prompts"]) == {"agent.answer.system"}
    assert data["prompts"]["agent.answer.system"]["template"] == (
        "Override only this prompt."
    )


def test_concept_extractor_uses_configured_prompt():
    llm = RecordingLLM()
    registry = PromptRegistry(overrides=[
        PromptSpec("extraction.concepts_relations.system", "Extract for test.")
    ])

    extractor = ConceptExtractor(llm, prompt_registry=registry)
    obs = extractor.extract("Python supports scripting.")

    assert llm.system == "Extract for test."
    assert obs.concepts == ["python"]


def test_builtin_skills_use_configured_prompt():
    registry = PromptRegistry(overrides=[
        PromptSpec("skill.digest_article.user", "Digest custom: {{text}}")
    ])
    skills = SkillRegistry()
    register_builtin_skills(skills, registry)

    prompt = skills.get("digest_article").render_prompt(text="sample")

    assert prompt == "Digest custom: sample"
