"""Tests for the Skill system."""

import json
import tempfile
from pathlib import Path

import pytest

from world0.agents.skill import (
    Skill,
    SkillParam,
    SkillRegistry,
    builtin_skills,
    register_builtin_skills,
)


# ── SkillParam ──────────────────────────────────────────────────────


class TestSkillParam:
    def test_defaults(self):
        p = SkillParam(name="text", description="Some text")
        assert p.required is True
        assert p.default == ""

    def test_optional(self):
        p = SkillParam(name="x", description="opt", required=False, default="abc")
        assert p.required is False
        assert p.default == "abc"


# ── Skill ───────────────────────────────────────────────────────────


class TestSkill:
    def _make_skill(self):
        return Skill(
            name="test_skill",
            description="A test skill",
            prompt_template="Hello {{name}}, topic is {{topic}}.",
            parameters=[
                SkillParam("name", "User name"),
                SkillParam("topic", "Topic to discuss"),
            ],
            tags=["test", "demo"],
        )

    def test_render_prompt(self):
        skill = self._make_skill()
        prompt = skill.render_prompt(name="Alice", topic="Python")
        assert prompt == "Hello Alice, topic is Python."

    def test_render_prompt_missing_uses_default(self):
        skill = Skill(
            name="s",
            description="d",
            prompt_template="Value: {{x}}",
            parameters=[SkillParam("x", "desc", required=False, default="fallback")],
        )
        prompt = skill.render_prompt()
        assert prompt == "Value: fallback"

    def test_validate_params_ok(self):
        skill = self._make_skill()
        errors = skill.validate_params(name="Alice", topic="Python")
        assert errors == []

    def test_validate_params_missing(self):
        skill = self._make_skill()
        errors = skill.validate_params(name="Alice")
        assert len(errors) == 1
        assert "topic" in errors[0]

    def test_validate_params_all_missing(self):
        skill = self._make_skill()
        errors = skill.validate_params()
        assert len(errors) == 2

    def test_to_dict(self):
        skill = self._make_skill()
        d = skill.to_dict()
        assert d["name"] == "test_skill"
        assert len(d["parameters"]) == 2
        assert d["tags"] == ["test", "demo"]

    def test_system_override(self):
        skill = Skill(
            name="s",
            description="d",
            prompt_template="do stuff",
            system_override="You are a special agent.",
        )
        assert skill.system_override == "You are a special agent."


# ── SkillRegistry ───────────────────────────────────────────────────


class TestSkillRegistry:
    def test_register_and_get(self):
        reg = SkillRegistry()
        skill = Skill(name="a", description="desc", prompt_template="hi")
        reg.register(skill)
        assert reg.get("a") is skill
        assert reg.get("nonexistent") is None

    def test_all_and_names(self):
        reg = SkillRegistry()
        reg.register(Skill(name="a", description="d", prompt_template="p"))
        reg.register(Skill(name="b", description="d", prompt_template="p"))
        assert set(reg.names()) == {"a", "b"}
        assert len(reg.all()) == 2

    def test_len_and_contains(self):
        reg = SkillRegistry()
        assert len(reg) == 0
        assert "a" not in reg
        reg.register(Skill(name="a", description="d", prompt_template="p"))
        assert len(reg) == 1
        assert "a" in reg

    def test_search_by_name(self):
        reg = SkillRegistry()
        reg.register(Skill(name="digest_article", description="d", prompt_template="p"))
        reg.register(Skill(name="analyze", description="d", prompt_template="p"))
        results = reg.search("digest")
        assert len(results) == 1
        assert results[0].name == "digest_article"

    def test_search_by_description(self):
        reg = SkillRegistry()
        reg.register(Skill(name="a", description="deep analysis tool", prompt_template="p"))
        results = reg.search("analysis")
        assert len(results) == 1

    def test_search_by_tag(self):
        reg = SkillRegistry()
        reg.register(Skill(name="a", description="d", prompt_template="p", tags=["review"]))
        reg.register(Skill(name="b", description="d", prompt_template="p", tags=["learn"]))
        results = reg.search("review")
        assert len(results) == 1
        assert results[0].name == "a"

    def test_search_case_insensitive(self):
        reg = SkillRegistry()
        reg.register(Skill(name="MySkill", description="d", prompt_template="p"))
        assert len(reg.search("myskill")) == 1
        assert len(reg.search("MYSKILL")) == 1

    def test_load_from_file(self):
        data = {
            "skills": [
                {
                    "name": "custom_skill",
                    "description": "A custom skill",
                    "prompt_template": "Do {{action}}",
                    "parameters": [
                        {"name": "action", "description": "What to do", "required": True}
                    ],
                    "tags": ["custom"],
                },
                {
                    "name": "simple",
                    "description": "Simple one",
                    "prompt_template": "Just do it",
                },
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            reg = SkillRegistry()
            count = reg.load_from_file(f.name)

        assert count == 2
        assert "custom_skill" in reg
        assert "simple" in reg
        skill = reg.get("custom_skill")
        assert len(skill.parameters) == 1
        assert skill.tags == ["custom"]

    def test_load_from_nonexistent_file(self):
        reg = SkillRegistry()
        assert reg.load_from_file("/nonexistent/path.json") == 0

    def test_load_from_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json!!!")
            f.flush()
            reg = SkillRegistry()
            assert reg.load_from_file(f.name) == 0


# ── Built-in Skills ────────────────────────────────────────────────


class TestBuiltinSkills:
    def test_builtin_count(self):
        skills = builtin_skills()
        assert len(skills) == 7

    def test_builtin_names(self):
        names = {s.name for s in builtin_skills()}
        expected = {
            "research_topic",
            "digest_article", "analyze_topic", "build_knowledge_map",
            "review_and_connect", "summarize_world", "learn_and_quiz",
        }
        assert names == expected

    def test_register_builtin(self):
        reg = SkillRegistry()
        register_builtin_skills(reg)
        assert len(reg) == 7
        assert "digest_article" in reg

    def test_research_topic_renders(self):
        reg = SkillRegistry()
        register_builtin_skills(reg)
        skill = reg.get("research_topic")
        prompt = skill.render_prompt(
            topic="agentic coding",
            focus="benchmarks",
            sources_limit="3",
            save_findings="true",
        )
        assert "agentic coding" in prompt
        assert "benchmarks" in prompt

    def test_digest_article_renders(self):
        reg = SkillRegistry()
        register_builtin_skills(reg)
        skill = reg.get("digest_article")
        prompt = skill.render_prompt(text="Hello world article content")
        assert "Hello world article content" in prompt

    def test_analyze_topic_renders(self):
        reg = SkillRegistry()
        register_builtin_skills(reg)
        skill = reg.get("analyze_topic")
        prompt = skill.render_prompt(topic="machine learning")
        assert "machine learning" in prompt

    def test_build_knowledge_map_renders(self):
        reg = SkillRegistry()
        register_builtin_skills(reg)
        skill = reg.get("build_knowledge_map")
        prompt = skill.render_prompt(concepts="Python, AI, Web")
        assert "Python, AI, Web" in prompt

    def test_parameterless_skills(self):
        """review_and_connect and summarize_world have no params."""
        reg = SkillRegistry()
        register_builtin_skills(reg)
        for name in ["review_and_connect", "summarize_world"]:
            skill = reg.get(name)
            assert skill.parameters == []
            errors = skill.validate_params()
            assert errors == []
            prompt = skill.render_prompt()
            assert len(prompt) > 0
