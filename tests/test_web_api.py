"""Tests for the PKM Agent web API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from world0.agents.session import TurnSummary
from world0.agents.web import create_app
from world0.llm.base import LLMProvider


class FakeLLM(LLMProvider):
    def complete_json(self, system: str, user: str) -> str:
        return json.dumps({
            "concepts": [
                {"name": "python", "description": "programming language"},
                {"name": "web framework", "description": "HTTP server toolkit"},
            ],
            "relations": [
                {"source": "python", "target": "web framework", "type": "enables"},
            ],
        })


@pytest.fixture
def client(tmp_path: Path):
    llm = FakeLLM()
    app = create_app(store_path=tmp_path / "test_web", llm=llm)
    return TestClient(app)


def test_index(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "World 0" in resp.text
    assert "settings-prompt-select" in resp.text


def test_status_empty(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_concepts"] == 0
    assert data["total_relations"] == 0


def test_learn(client):
    resp = client.post("/api/learn", json={"text": "Python is great for web development"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] in ("text", "markdown")
    assert "python" in data["message"].lower()


def test_learn_then_status(client):
    client.post("/api/learn", json={"text": "Python is great"})
    resp = client.get("/api/status")
    data = resp.json()
    assert data["total_concepts"] > 0


def test_concepts(client):
    client.post("/api/learn", json={"text": "Python is great"})
    resp = client.get("/api/concepts")
    data = resp.json()
    assert len(data["concepts"]) > 0
    assert any(c["name"] == "python" for c in data["concepts"])


def test_concept_card(client):
    client.post("/api/learn", json={"text": "Python supports web frameworks"})
    resp = client.get("/api/concepts/python/card")
    assert resp.status_code == 200
    data = resp.json()["card"]
    assert data["name"] == "python"
    assert "relations" in data
    assert "recent_activity" in data


def test_explore(client):
    client.post("/api/learn", json={"text": "Python and web"})
    resp = client.get("/api/explore/python")
    assert resp.status_code == 200
    data = resp.json()
    assert "python" in data["message"].lower()


def test_explore_not_found(client):
    resp = client.get("/api/explore/nonexistent")
    data = resp.json()
    assert "not found" in data["message"].lower()


def test_search(client):
    client.post("/api/learn", json={"text": "Python and web"})
    resp = client.get("/api/search", params={"q": "python"})
    data = resp.json()
    assert len(data["results"]) > 0


def test_search_empty(client):
    resp = client.get("/api/search", params={"q": ""})
    data = resp.json()
    assert data["results"] == []


def test_connect(client):
    resp = client.post("/api/connect", json={
        "source": "python",
        "target": "django",
        "relation_type": "enables",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "connected" in data["message"].lower()


def test_reflect(client):
    resp = client.post("/api/reflect")
    assert resp.status_code == 200
    data = resp.json()
    assert "reflection" in data["message"].lower()


def test_graph(client):
    client.post("/api/learn", json={"text": "Python and ML"})
    resp = client.get("/api/graph")
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) > 0


def test_relation_types(client):
    resp = client.get("/api/relation_types")
    data = resp.json()
    assert "membership" in data["types"]
    assert "conflict" in data["types"]
    assert "generic_relation" in data["types"]
    assert data["axes"] == ["positive", "negative", "parallel"]


def test_agent_status(client):
    resp = client.get("/api/agent/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "agentic_ready" in data
    assert data["llm_enabled"] is True
    assert data["state"]["status"] == "blocked"
    assert "Agentic mode unavailable" in data["state"]["reason"]
    assert "store_path" in data
    assert "current_session" in data
    assert data["current_session"]["state"]["status"] == "active"
    assert "tool_count" in data
    assert "languages" in data
    assert "providers" in data
    assert "provider_env" in data
    assert "external_agents" in data
    assert "last_external_consultation" in data
    assert "codex" in data["provider_env"]
    assert "claude" in data["provider_env"]
    assert "openai" in data["provider_env"]
    provider_ids = {item["id"] for item in data["providers"]}
    assert "codex" in provider_ids
    assert "claude" in provider_ids


def test_sessions_flow(client):
    learn_resp = client.post("/api/learn", json={"text": "Python supports APIs"})
    assert learn_resp.status_code == 200

    sessions_resp = client.get("/api/sessions")
    assert sessions_resp.status_code == 200
    sessions = sessions_resp.json()["sessions"]
    assert len(sessions) == 1
    session_id = sessions[0]["id"]

    detail_resp = client.get(f"/api/sessions/{session_id}")
    assert detail_resp.status_code == 200
    session = detail_resp.json()["session"]
    assert session["id"] == session_id
    assert session["state"]["status"] == "active"
    assert len(session["messages"]) >= 2

    resume_resp = client.post("/api/sessions/resume", json={"session_id": session_id})
    assert resume_resp.status_code == 200
    resume_data = resume_resp.json()
    assert resume_data["success"] is True
    assert resume_data["session"]["id"] == session_id


def test_session_rename_endpoint(client):
    client.post("/api/learn", json={"text": "Python supports APIs"})
    session_id = client.get("/api/sessions").json()["sessions"][0]["id"]
    resp = client.post("/api/sessions/rename", json={
        "session_id": session_id,
        "title": "Research Session",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["session"]["title"] == "Research Session"


def test_session_compact_endpoint(client):
    for i in range(10):
        client.post("/api/learn", json={"text": f"Python supports APIs {i}"})
    resp = client.post("/api/sessions/compact", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["session"]["state"]["status"] == "compacted"
    assert data["covered_messages"] > 0


def test_latest_failure_endpoint(client):
    from world0.agents import web as web_module

    web_module._agent.session.add_turn_summary(TurnSummary(
        stop_reason="end_turn",
        failure_class="tool_runtime",
        rounds=2,
        tool_count=1,
        failed_tools=["web_fetch"],
        user_input_preview="Fetch a source",
        assistant_output_preview="The fetch failed.",
    ))
    resp = client.get("/api/agent/latest_failure")
    assert resp.status_code == 200
    data = resp.json()
    assert data["failure"]["failure_class"] == "tool_runtime"
    assert "manual_compact" not in (data["failure"]["recovery_actions"] or [])


def test_projection_feedback_flow(client):
    client.post("/api/learn", json={"text": "FastAPI depends on Python and supports ORM usage"})
    ask_resp = client.post("/api/ask", json={"query": "python web framework"})
    assert ask_resp.status_code == 200

    last_projection = client.get("/api/projection/last")
    assert last_projection.status_code == 200
    projection = last_projection.json()["projection"]
    assert projection is not None
    assert projection["query"] == "python web framework"

    feedback_resp = client.post("/api/projection/feedback", json={
        "useful": True,
        "missing_concepts": ["postgresql"],
        "noisy_concepts": ["web framework"],
        "notes": "Need database concepts.",
    })
    assert feedback_resp.status_code == 200
    data = feedback_resp.json()
    assert data["success"] is True
    assert "postgresql" in data["created_missing_concepts"]
    latest = client.get("/api/projection/last").json()["latest_feedback"]
    assert latest is not None
    assert latest["useful"] is True


def test_settings_roundtrip(client):
    resp = client.post("/api/settings", json={
        "language": "zh",
        "provider": "none",
        "model": "",
        "auto_sediment_dialogue": False,
        "dialogue_sediment_interval": 3,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["settings"]["language"] == "zh"
    assert data["settings"]["provider"] == "none"
    assert data["settings"]["auto_sediment_dialogue"] is False
    assert data["settings"]["dialogue_sediment_interval"] == 3
    assert data["agentic_ready"] is False
    assert data["state"]["status"] == "blocked"

    status_resp = client.get("/api/agent/status")
    status = status_resp.json()
    assert status["language"] == "zh"
    assert status["llm_enabled"] is False
    assert status["unavailable_reason"] == "No LLM provider configured."
    assert status["state"]["status"] == "blocked"
    assert status["state"]["reason"] == "No LLM provider configured."


def test_prompt_registry_api_roundtrip(client):
    prompt_id = "agent.answer.system"

    list_resp = client.get("/api/prompts")
    assert list_resp.status_code == 200
    prompt_list = list_resp.json()
    assert any(item["id"] == prompt_id for item in prompt_list["prompts"])
    assert prompt_list["config_path"].endswith("prompts.json")

    update_resp = client.post(
        f"/api/prompts/{prompt_id}",
        json={"template": "Custom web answer prompt."},
    )
    assert update_resp.status_code == 200
    update_data = update_resp.json()
    assert update_data["success"] is True
    assert update_data["prompt"]["template"] == "Custom web answer prompt."
    assert update_data["prompt"]["is_overridden"] is True
    assert update_data["prompt"]["has_active_override"] is True

    path = Path(update_data["config_path"])
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["prompts"][prompt_id]["template"] == "Custom web answer prompt."

    show_resp = client.get(f"/api/prompts/{prompt_id}")
    assert show_resp.status_code == 200
    assert show_resp.json()["prompt"]["template"] == "Custom web answer prompt."

    export_resp = client.get("/api/prompts/export")
    assert export_resp.status_code == 200
    assert export_resp.json()["prompts"][prompt_id]["template"] == (
        "Custom web answer prompt."
    )

    reset_resp = client.post(f"/api/prompts/{prompt_id}/reset")
    assert reset_resp.status_code == 200
    reset_data = reset_resp.json()
    assert reset_data["success"] is True
    assert reset_data["prompt"]["has_active_override"] is False
    assert reset_data["prompt"]["template"] == reset_data["prompt"]["default_template"]
    reset_saved = json.loads(path.read_text(encoding="utf-8"))
    assert prompt_id not in reset_saved["prompts"]


def test_research_endpoint(client, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "world0.agents.web._agent.research_topic",
        lambda topic, focus="", max_sources=4, save_findings=True: (
            f"## Research Brief\n\n**Topic:** {topic}\n\n**Focus:** {focus}"
        ),
    )
    resp = client.post("/api/research", json={
        "topic": "independent research agents",
        "focus": "citations",
        "max_sources": 2,
        "save_findings": True,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "research brief" in data["message"].lower()
