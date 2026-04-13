"""Tests for the PKM Agent."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from world0.agents.pkm import PKMAgent
from world0.agents.provider import ChatResponse
from world0.agents.session import TurnSummary
from world0.agents.state import AgentLifecycleStatus
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


class FakeChatProvider:
    provider_name = "openai"
    model = "openai/gpt-5.4"

    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)

    def chat(self, messages, *, system=None, tools=None):  # noqa: ANN001
        if not self._responses:
            raise AssertionError("No more fake chat responses configured")
        return self._responses.pop(0)


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


class TestDialogueSedimentation:
    def test_sediment_dialogue_turn_ingests_concepts(self, tmp_store: Path) -> None:
        llm = FakeLLM(responses=[
            json.dumps({
                "concepts": [
                    {"name": "model serving", "description": "runtime model endpoint"},
                    {"name": "pytorch", "description": "training framework"},
                    {"name": "training pipeline", "description": "upstream training path"},
                ],
                "relations": [
                    {"source": "model serving", "target": "pytorch", "type": "depends_on"},
                    {"source": "pytorch", "target": "training pipeline", "type": "supports"},
                ],
            }),
        ])
        agent = PKMAgent(store_path=tmp_store, llm=llm)

        sediment = agent.sediment_dialogue_turn(
            "How does model serving connect back upstream?",
            "Model serving depends on PyTorch and the training pipeline.",
            mode="chat",
        )

        assert sediment["status"] == "ingested"
        assert agent.latest_dialogue_sediment() is not None
        assert agent.latest_dialogue_sediment()["mode"] == "chat"
        assert agent.world.concepts.resolve("model serving") is not None
        assert agent.world.concepts.resolve("pytorch") is not None

    def test_sediment_dialogue_turn_skips_when_disabled(self, tmp_store: Path) -> None:
        agent = PKMAgent(store_path=tmp_store, llm=FakeLLM())
        agent.configure_runtime(
            provider="none",
            auto_sediment_dialogue=False,
        )

        sediment = agent.sediment_dialogue_turn(
            "What matters for deployment?",
            "Deployment depends on monitoring.",
            mode="chat",
        )

        assert sediment["status"] == "skipped"
        assert "disabled" in sediment["reason"].lower()

    def test_configure_runtime_rejects_invalid_sediment_interval(
        self,
        agent_with_llm: PKMAgent,
    ) -> None:
        with pytest.raises(ValueError, match="between 1 and 20"):
            agent_with_llm.configure_runtime(dialogue_sediment_interval=21)

    def test_agent_chat_auto_sediments_successful_turn(self, tmp_store: Path) -> None:
        llm = FakeLLM(responses=[
            json.dumps({
                "concepts": [
                    {"name": "model serving", "description": "runtime endpoint"},
                    {"name": "fastapi", "description": "web framework"},
                    {"name": "deployment", "description": "release path"},
                ],
                "relations": [
                    {"source": "model serving", "target": "fastapi", "type": "depends_on"},
                    {"source": "model serving", "target": "deployment", "type": "depends_on"},
                ],
            }),
        ])
        agent = PKMAgent(store_path=tmp_store, llm=llm)
        agent._chat_provider = FakeChatProvider([
            ChatResponse(
                content="Model serving depends on FastAPI and deployment.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ])

        result = agent.agent_chat("What runtime path matters for model serving?")

        assert "fastapi" in result.lower()
        sediment = agent.latest_dialogue_sediment()
        assert sediment is not None
        assert sediment["status"] == "ingested"
        assert sediment["mode"] == "agent_chat"
        assert agent.world.concepts.resolve("model serving") is not None
        assert agent.world.concepts.resolve("fastapi") is not None

    def test_agent_chat_respects_sediment_interval(self, tmp_store: Path) -> None:
        llm = FakeLLM(responses=[
            json.dumps({
                "concepts": [
                    {"name": "model serving", "description": "runtime endpoint"},
                    {"name": "deployment", "description": "release path"},
                    {"name": "latency", "description": "runtime performance signal"},
                ],
                "relations": [
                    {"source": "model serving", "target": "deployment", "type": "depends_on"},
                    {"source": "deployment", "target": "latency", "type": "related_to"},
                ],
            }),
        ])
        agent = PKMAgent(store_path=tmp_store, llm=llm)
        agent._runtime_settings["dialogue_sediment_interval"] = 2
        agent._chat_provider = FakeChatProvider([
            ChatResponse(
                content="Model serving depends on deployment.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
            ChatResponse(
                content="Latency is the current deployment concern.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ])

        first = agent.agent_chat("What release path matters for model serving?")
        assert "deployment" in first.lower()
        first_sediment = agent.latest_dialogue_sediment()
        assert first_sediment is not None
        assert first_sediment["status"] == "pending"
        assert first_sediment["pending_turns"] == 1
        assert agent.world.concepts.resolve("model serving") is None

        second = agent.agent_chat("What signal tells us the runtime is unhealthy?")
        assert "latency" in second.lower()
        second_sediment = agent.latest_dialogue_sediment()
        assert second_sediment is not None
        assert second_sediment["status"] == "ingested"
        assert second_sediment["pending_turns"] == 2
        assert second_sediment["required_turns"] == 2
        assert agent.world.concepts.resolve("model serving") is not None
        assert agent.world.concepts.resolve("latency") is not None


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

    def test_ask_stores_projection_snapshot(self, agent: PKMAgent) -> None:
        obs = Observation(
            concepts=["fastapi", "python", "postgresql"],
            relations=[("fastapi", "python", "depends_on")],
        )
        agent.learn_structured(obs)
        agent.ask("fastapi backend")
        snapshot = agent.latest_projection_snapshot()
        assert snapshot is not None
        assert snapshot["query"] == "fastapi backend"
        assert any(item["name"] == "fastapi" for item in snapshot["concepts"])


class TestResearch:
    def test_search_web_with_filters_and_fetch(
        self,
        tmp_store: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from world0.agents import research as research_utils

        llm = FakeResearchLLM()
        agent = PKMAgent(store_path=tmp_store, llm=llm)

        def fake_search(query, limit=5, timeout=15, domains=None):
            assert query == "agent mcp"
            assert domains == ["docs.anthropic.com", "modelcontextprotocol.io"]
            return [
                research_utils.SearchResult(
                    title="Claude Code MCP",
                    url="https://docs.anthropic.com/en/docs/claude-code/mcp",
                    snippet="Claude Code can connect to local MCP servers.",
                    domain="docs.anthropic.com",
                ),
                research_utils.SearchResult(
                    title="MCP Spec",
                    url="https://modelcontextprotocol.io/introduction",
                    snippet="MCP standardizes tool and resource access for models.",
                    domain="modelcontextprotocol.io",
                ),
            ]

        monkeypatch.setattr(research_utils, "search_web", fake_search)
        monkeypatch.setattr(
            research_utils,
            "fetch_web_document",
            lambda url, max_chars=4000: research_utils.FetchedDocument(
                title="Fetched " + url.rsplit("/", 1)[-1],
                url=url,
                text="Search results can be upgraded into grounded source glimpses for agents.",
            ),
        )

        result = agent.search_web(
            "agent",
            focus="mcp",
            domains="docs.anthropic.com, modelcontextprotocol.io",
            fetch_pages=True,
        )

        assert "web search: agent" in result.lower()
        assert "domains:" in result.lower()
        assert "search brief" in result.lower()
        assert "source glimpses" in result.lower()

    def test_research_topic(self, tmp_store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from world0.agents import research as research_utils

        llm = FakeResearchLLM()
        agent = PKMAgent(store_path=tmp_store, llm=llm)

        monkeypatch.setattr(
            research_utils,
            "search_web",
            lambda query, limit=5, timeout=15, domains=None: [
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

    def test_search_web_recovers_without_domain_filters(
        self,
        tmp_store: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from world0.agents import research as research_utils

        llm = FakeResearchLLM()
        agent = PKMAgent(store_path=tmp_store, llm=llm)
        calls: list[tuple[str, list[str] | None]] = []

        def fake_search(query, limit=5, timeout=15, domains=None):
            calls.append((query, list(domains) if domains else None))
            if domains:
                raise RuntimeError("temporary network failure")
            return [
                research_utils.SearchResult(
                    title="Recovered result",
                    url="https://example.com/recovered",
                    snippet="Recovered after removing domain filters.",
                    domain="example.com",
                ),
            ]

        monkeypatch.setattr(research_utils, "search_web", fake_search)

        result = agent.search_web(
            "agent",
            focus="mcp",
            domains="docs.anthropic.com",
        )

        assert len(calls) >= 2
        assert "### Recovery" in result
        assert "without domain filters" in result
        assert agent.latest_failure() is None

    def test_prepare_session_for_agentic_compacts_old_messages(
        self,
        tmp_store: Path,
    ) -> None:
        llm = FakeResearchLLM()
        agent = PKMAgent(store_path=tmp_store, llm=llm)
        session = agent.session
        for i in range(60):
            role = "user" if i % 2 == 0 else "assistant"
            session.add_message(role, f"message {i} about agent research and citations")

        agent._prepare_session_for_agentic()

        assert session.compaction is not None
        assert session.compaction.covered_messages > 0
        msgs = session.to_llm_messages(max_messages=10)
        assert msgs[0]["role"] == "system"
        assert "Session Summary" in msgs[0]["content"]


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

    def test_agent_state_without_llm_is_blocked(self, agent: PKMAgent) -> None:
        state = agent.agent_state()
        assert state.status == AgentLifecycleStatus.BLOCKED
        assert state.reason == "No LLM provider configured."
        assert state.agentic_ready is False

    def test_agent_state_with_latest_turn_failure_is_degraded(
        self,
        agent_with_llm: PKMAgent,
    ) -> None:
        agent_with_llm._chat_provider = SimpleNamespace(
            provider_name="openai",
            model="openai/gpt-5.4",
        )
        agent_with_llm.session.add_turn_summary(TurnSummary(
            stop_reason="end_turn",
            failure_class="provider_rate_limit",
            rounds=2,
            tool_count=1,
            failed_tools=["web_search"],
            user_input_preview="Research retry policy",
            assistant_output_preview="Search failed due to rate limit.",
        ))
        state = agent_with_llm.agent_state()
        assert state.status == AgentLifecycleStatus.DEGRADED
        assert state.latest_failure_class == "provider_rate_limit"
        assert state.failed_tools == ["web_search"]
        assert "latest_turn" in state.degraded_sources

    def test_agent_state_with_runtime_failure_is_degraded(
        self,
        tmp_store: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from world0.agents import research as research_utils

        llm = FakeResearchLLM()
        agent = PKMAgent(store_path=tmp_store, llm=llm)
        agent._chat_provider = SimpleNamespace(
            provider_name="openai",
            model="openai/gpt-5.4",
        )
        monkeypatch.setattr(
            research_utils,
            "search_web",
            lambda query, limit=5, timeout=15, domains=None: (_ for _ in ()).throw(
                RuntimeError("temporary network failure")
            ),
        )
        result = agent.search_web("agent", focus="mcp", domains="docs.anthropic.com")
        state = agent.agent_state()
        assert "no web results found" in result.lower()
        assert state.status == AgentLifecycleStatus.DEGRADED
        assert "runtime_failure" in state.degraded_sources


class TestProjectionFeedback:
    def test_apply_projection_feedback_adjusts_projection_items(self, agent: PKMAgent) -> None:
        obs = Observation(
            concepts=["fastapi", "python", "orm"],
            relations=[("fastapi", "python", "depends_on"), ("fastapi", "orm", "supports")],
        )
        agent.learn_structured(obs)
        agent.ask("fastapi backend")

        fastapi = agent.world.concepts.resolve("fastapi")
        orm = agent.world.concepts.resolve("orm")
        rel = agent.world.relations.find_between(fastapi.id, orm.id)
        before_noise_conf = orm.confidence
        before_rel_weight = rel.weight

        result = agent.apply_projection_feedback(
            useful=True,
            missing_concepts=["postgresql"],
            noisy_concepts=["orm"],
            weak_relations=["fastapi -> supports -> orm"],
            notes="Need database concepts, ORM was too prominent.",
        )

        postgres = agent.world.concepts.resolve("postgresql")
        assert postgres is not None
        assert "postgresql" in result["created_missing_concepts"]
        assert "orm" in result["demoted_noisy_concepts"]
        assert "fastapi -> supports -> orm" in result["weakened_relations"]
        assert agent.world.concepts.resolve("orm").confidence < before_noise_conf
        assert agent.world.relations.get(rel.id).weight < before_rel_weight
        assert agent.latest_projection_feedback() is not None
        assert agent.latest_projection_feedback().useful is True


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
