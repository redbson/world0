"""Tests for the PKM Agent."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from world0.agents.pkm import PKMAgent
from world0.llm.base import LLMProvider
from world0.schemas.relation import RelationType
from world0.schemas.types import Observation


class FakeLLM(LLMProvider):
    """Fake LLM for testing — returns predictable JSON responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses) if responses else []
        self._call_count = 0

    def complete_json(self, system: str, user: str) -> str:
        self._call_count += 1
        if self._responses:
            return self._responses.pop(0)
        # Default: return a concept extraction response
        return json.dumps({
            "concepts": [
                {"name": "python", "description": "programming language"},
                {"name": "machine learning", "description": "AI subfield"},
            ],
            "relations": [
                {"source": "python", "target": "machine learning", "type": "supports"},
            ],
        })


class FakeResearchLLM(LLMProvider):
    """LLM stub that supports the research workflow prompts."""

    def complete_json(self, system: str, user: str) -> str:
        if '"seeds"' in system:
            return json.dumps({"seeds": ["agent research", "benchmarks"]})
        if '"key_points"' in system and '"open_questions"' in system:
            return json.dumps({
                "summary": "This source compares agent research systems and evaluation criteria.",
                "key_points": [
                    "Independent agents need source retrieval.",
                    "Evaluation should track citations and gaps.",
                ],
                "concepts": ["agent research", "citations", "evaluation"],
                "open_questions": ["How should retrieval quality be scored?"],
            })
        if '"findings"' in system and '"next_steps"' in system:
            return json.dumps({
                "summary": "Source notes converge on retrieval, synthesis, and explicit uncertainty tracking.",
                "findings": [
                    "Research agents need web retrieval and source grounding.",
                    "Structured gaps and next steps improve follow-on work.",
                ],
                "gaps": ["Longitudinal evaluation data is still sparse."],
                "next_steps": ["Benchmark retrieval quality across providers."],
            })
        if "cognitive projection" in user.lower():
            return "The current projection emphasizes source grounding, citations, and evaluation."
        return json.dumps({
            "concepts": [
                {"name": "agent research", "description": "autonomous research workflow"},
                {"name": "citations", "description": "source grounding"},
            ],
            "relations": [
                {"source": "agent research", "target": "citations", "type": "supports"},
            ],
        })


@pytest.fixture
def tmp_store(tmp_path: Path) -> Path:
    return tmp_path / "test_pkm"


@pytest.fixture
def agent(tmp_store: Path) -> PKMAgent:
    """Agent without LLM — for structured input tests."""
    return PKMAgent(store_path=tmp_store, llm=None)


@pytest.fixture
def agent_with_llm(tmp_store: Path) -> PKMAgent:
    """Agent with fake LLM — for text input tests."""
    llm = FakeLLM()
    return PKMAgent(store_path=tmp_store, llm=llm)


class TestLearnStructured:
    def test_learn_new_concepts(self, agent: PKMAgent) -> None:
        obs = Observation(
            concepts=["python", "machine learning"],
            relations=[("python", "machine learning", "supports")],
            task="test",
            source="test",
        )
        result = agent.learn_structured(obs)
        assert "python" in result.lower()
        assert "machine learning" in result.lower()

    def test_learn_reinforces_existing(self, agent: PKMAgent) -> None:
        obs = Observation(concepts=["python"], task="test")
        agent.learn_structured(obs)
        result = agent.learn_structured(obs)
        assert "reinforced" in result.lower()

    def test_learn_empty(self, agent: PKMAgent) -> None:
        obs = Observation()
        result = agent.learn_structured(obs)
        assert "no changes" in result.lower()


class TestLearnText:
    def test_learn_requires_llm(self, agent: PKMAgent) -> None:
        with pytest.raises(RuntimeError, match="LLM provider"):
            agent.learn("some text")

    def test_learn_with_llm(self, agent_with_llm: PKMAgent) -> None:
        result = agent_with_llm.learn("Python is great for ML")
        assert "python" in result.lower()

    def test_learn_empty_text(self, agent_with_llm: PKMAgent) -> None:
        result = agent_with_llm.learn("")
        assert "empty" in result.lower()


class TestAsk:
    def test_ask_empty(self, agent: PKMAgent) -> None:
        result = agent.ask("")
        assert "provide" in result.lower()

    def test_ask_no_concepts_found(self, agent: PKMAgent) -> None:
        result = agent.ask("quantum computing")
        assert "no concepts" in result.lower() or "couldn't" in result.lower()

    def test_ask_with_concepts(self, agent: PKMAgent) -> None:
        obs = Observation(
            concepts=["python", "machine learning", "deep learning"],
            relations=[
                ("python", "machine learning", "supports"),
                ("machine learning", "deep learning", "contains"),
            ],
            task="setup",
        )
        agent.learn_structured(obs)
        result = agent.ask("python machine learning")
        # Should return a projection since we have no LLM
        assert "python" in result.lower()
        assert "projection basis" in result.lower()


class TestResearch:
    def test_research_topic(self, tmp_store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from world0.agents import research as research_utils

        llm = FakeResearchLLM()
        agent = PKMAgent(store_path=tmp_store, llm=llm)

        monkeypatch.setattr(
            research_utils,
            "search_web",
            lambda query, limit=5: [
                research_utils.SearchResult(
                    title="Agent Research Survey",
                    url="https://example.com/survey",
                    snippet="A survey on research agents and citations.",
                ),
            ],
        )
        monkeypatch.setattr(
            research_utils,
            "fetch_web_document",
            lambda url, max_chars=12000: research_utils.FetchedDocument(
                title="Agent Research Survey",
                url=url,
                text="Research agents require search, synthesis, and citations.",
            ),
        )

        result = agent.research_topic(
            "independent research agents",
            focus="citations",
            max_sources=1,
            save_findings=True,
        )

        assert "research brief" in result.lower()
        assert "sources reviewed" in result.lower()
        assert "world 0 update" in result.lower()
        assert "projection into world 0" in result.lower()


class TestExplore:
    def test_explore_nonexistent(self, agent: PKMAgent) -> None:
        result = agent.explore("nonexistent")
        assert "not found" in result.lower()

    def test_explore_existing(self, agent: PKMAgent) -> None:
        obs = Observation(
            concepts=["python", "flask"],
            relations=[("python", "flask", "supports")],
            descriptions={"python": "A versatile programming language"},
            task="test",
        )
        agent.learn_structured(obs)
        result = agent.explore("python")
        assert "python" in result.lower()
        assert "maturity" in result.lower()
        assert "confidence" in result.lower()
        assert "versatile" in result.lower()

    def test_explore_shows_relations(self, agent: PKMAgent) -> None:
        obs = Observation(
            concepts=["python", "flask", "django"],
            relations=[
                ("python", "flask", "supports"),
                ("python", "django", "supports"),
            ],
            task="test",
        )
        agent.learn_structured(obs)
        result = agent.explore("python")
        assert "flask" in result.lower()
        assert "django" in result.lower()

    def test_concept_card(self, agent: PKMAgent) -> None:
        obs = Observation(
            concepts=["python", "fastapi"],
            relations=[("fastapi", "python", "depends_on")],
            descriptions={"python": "A versatile programming language"},
            task="api design",
            source="session_01",
        )
        agent.learn_structured(obs)
        card = agent.concept_card("python")
        assert card is not None
        assert card["name"] == "python"
        assert card["description"]
        assert card["relation_count"] >= 1
        assert "api design" in card["tasks"]


class TestConnect:
    def test_connect_new_concepts(self, agent: PKMAgent) -> None:
        result = agent.connect("python", "web development", "supports")
        assert "connected" in result.lower()

    def test_connect_invalid_type(self, agent: PKMAgent) -> None:
        result = agent.connect("a", "b", "invalid_type")
        assert "invalid" in result.lower()

    def test_connect_default_type(self, agent: PKMAgent) -> None:
        result = agent.connect("a", "b")
        assert "related_to" in result.lower()


class TestSearch:
    def test_search_empty_world(self, agent: PKMAgent) -> None:
        result = agent.search("python")
        assert "no concepts" in result.lower()

    def test_search_by_name(self, agent: PKMAgent) -> None:
        obs = Observation(concepts=["python", "javascript", "typescript"])
        agent.learn_structured(obs)
        result = agent.search("script")
        assert "javascript" in result.lower()
        assert "typescript" in result.lower()
        assert "python" not in result.lower()

    def test_search_by_description(self, agent: PKMAgent) -> None:
        obs = Observation(
            concepts=["python"],
            descriptions={"python": "A versatile language"},
        )
        agent.learn_structured(obs)
        result = agent.search("versatile")
        assert "python" in result.lower()


class TestReflect:
    def test_reflect_empty_world(self, agent: PKMAgent) -> None:
        result = agent.reflect()
        assert "reflection complete" in result.lower()

    def test_reflect_with_concepts(self, agent: PKMAgent) -> None:
        obs = Observation(concepts=["python", "rust", "go"])
        agent.learn_structured(obs)
        result = agent.reflect()
        assert "reflection complete" in result.lower()


class TestStatus:
    def test_status_empty(self, agent: PKMAgent) -> None:
        result = agent.status()
        assert "concepts: 0" in result.lower() or "concepts:** 0" in result.lower()

    def test_status_with_concepts(self, agent: PKMAgent) -> None:
        obs = Observation(
            concepts=["python", "rust"],
            relations=[("python", "rust", "contrasts")],
        )
        agent.learn_structured(obs)
        result = agent.status()
        assert "2" in result  # 2 concepts


class TestHandleInput:
    def test_natural_language_is_ask(self, agent: PKMAgent) -> None:
        obs = Observation(concepts=["python"])
        agent.learn_structured(obs)
        result = agent.handle_input("tell me about python")
        assert result is not None

    def test_quit_returns_none(self, agent: PKMAgent) -> None:
        assert agent.handle_input("/quit") is None
        assert agent.handle_input("/exit") is None

    def test_help_command(self, agent: PKMAgent) -> None:
        result = agent.handle_input("/help")
        assert result is not None
        assert "learn" in result.lower()

    def test_status_command(self, agent: PKMAgent) -> None:
        result = agent.handle_input("/status")
        assert result is not None
        assert "concepts" in result.lower()

    def test_unknown_command(self, agent: PKMAgent) -> None:
        result = agent.handle_input("/unknown")
        assert result is not None
        assert "unknown" in result.lower()


class TestSeedExtraction:
    def test_fallback_keyword_extraction(self, agent: PKMAgent) -> None:
        seeds = agent._extract_seeds("How does machine learning work with python?")
        assert len(seeds) > 0
        assert "machine" in seeds or "learning" in seeds or "python" in seeds

    def test_llm_seed_extraction(self, tmp_store: Path) -> None:
        llm = FakeLLM(responses=[
            json.dumps({"seeds": ["machine learning", "python"]}),
        ])
        agent = PKMAgent(store_path=tmp_store, llm=llm)
        seeds = agent._extract_seeds("How does machine learning work?")
        assert "machine learning" in seeds
        assert "python" in seeds
