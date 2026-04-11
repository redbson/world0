"""Tests for the PKM Agent web API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

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
                {"source": "python", "target": "web framework", "type": "supports"},
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
        "relation_type": "supports",
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
    assert "related_to" in data["types"]
    assert "supports" in data["types"]


def test_agent_status(client):
    resp = client.get("/api/agent/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "agentic_ready" in data
    assert "store_path" in data
    assert "current_session" in data
    assert "tool_count" in data
    assert "languages" in data
    assert "providers" in data
    assert "provider_env" in data
    assert "openai" in data["provider_env"]


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
    assert len(session["messages"]) >= 2

    resume_resp = client.post("/api/sessions/resume", json={"session_id": session_id})
    assert resume_resp.status_code == 200
    resume_data = resume_resp.json()
    assert resume_data["success"] is True
    assert resume_data["session"]["id"] == session_id


def test_settings_roundtrip(client):
    resp = client.post("/api/settings", json={
        "language": "zh",
        "provider": "none",
        "model": "",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["settings"]["language"] == "zh"
    assert data["settings"]["provider"] == "none"

    status_resp = client.get("/api/agent/status")
    status = status_resp.json()
    assert status["language"] == "zh"


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
