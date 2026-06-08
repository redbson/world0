"""Tests for the tool registry and PKM tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from world0.agents.pkm import PKMAgent
from world0.agents.tools.registry import (
    Permission,
    Tool,
    ToolParam,
    ToolRegistry,
    ToolResult,
)
from world0.agents.tools.pkm_tools import build_pkm_tools
from world0.llm.base import LLMProvider


class FakeLLM(LLMProvider):
    def complete_json(self, system: str, user: str) -> str:
        return json.dumps({
            "concepts": [
                {"name": "python", "description": "programming language"},
                {"name": "testing", "description": "software quality assurance"},
            ],
            "relations": [
                {"source": "python", "target": "testing", "type": "supports"},
            ],
        })


# ── ToolRegistry tests ──

class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = Tool(
            name="test_tool",
            description="A test",
            parameters=[],
            handler=lambda: ToolResult(success=True, output="ok"),
        )
        reg.register(tool)
        assert "test_tool" in reg
        assert reg.get("test_tool") is tool
        assert len(reg) == 1

    def test_execute(self):
        reg = ToolRegistry()
        reg.register(Tool(
            name="greet",
            description="Greet",
            parameters=[ToolParam("name", "Name to greet")],
            handler=lambda name: ToolResult(success=True, output=f"Hello {name}"),
        ))
        result = reg.execute("greet", {"name": "World"})
        assert result.success
        assert "Hello World" in result.output

    def test_execute_unknown_tool(self):
        reg = ToolRegistry()
        result = reg.execute("nonexistent", {})
        assert not result.success
        assert "Unknown tool" in result.output

    def test_permission_denied(self):
        reg = ToolRegistry()
        reg.register(Tool(
            name="admin_tool",
            description="Admin only",
            parameters=[],
            handler=lambda: ToolResult(success=True, output="ok"),
            permission=Permission.ADMIN,
        ))
        result = reg.execute("admin_tool", {}, max_permission=Permission.READ)
        assert not result.success
        assert "Permission denied" in result.output

    def test_permission_allowed(self):
        reg = ToolRegistry()
        reg.register(Tool(
            name="read_tool",
            description="Read",
            parameters=[],
            handler=lambda: ToolResult(success=True, output="ok"),
            permission=Permission.READ,
        ))
        result = reg.execute("read_tool", {}, max_permission=Permission.READ)
        assert result.success

    def test_openai_specs(self):
        reg = ToolRegistry()
        reg.register(Tool(
            name="search",
            description="Search stuff",
            parameters=[
                ToolParam("query", "Query string", required=True),
                ToolParam("limit", "Max results", type="integer", required=False),
            ],
            handler=lambda **kw: ToolResult(success=True, output=""),
        ))
        specs = reg.openai_specs()
        assert len(specs) == 1
        assert specs[0]["type"] == "function"
        assert specs[0]["function"]["name"] == "search"
        assert "query" in specs[0]["function"]["parameters"]["properties"]

    def test_anthropic_specs(self):
        reg = ToolRegistry()
        reg.register(Tool(
            name="learn",
            description="Learn stuff",
            parameters=[ToolParam("text", "Input text")],
            handler=lambda **kw: ToolResult(success=True, output=""),
        ))
        specs = reg.anthropic_specs()
        assert len(specs) == 1
        assert specs[0]["name"] == "learn"
        assert "input_schema" in specs[0]

    def test_permission_filter_on_specs(self):
        reg = ToolRegistry()
        reg.register(Tool(
            name="read_op", description="Read", parameters=[],
            handler=lambda: ToolResult(success=True, output=""),
            permission=Permission.READ,
        ))
        reg.register(Tool(
            name="admin_op", description="Admin", parameters=[],
            handler=lambda: ToolResult(success=True, output=""),
            permission=Permission.ADMIN,
        ))
        read_specs = reg.openai_specs(max_permission=Permission.READ)
        assert len(read_specs) == 1
        assert read_specs[0]["function"]["name"] == "read_op"

    def test_handler_exception(self):
        reg = ToolRegistry()
        reg.register(Tool(
            name="bad", description="Fails", parameters=[],
            handler=lambda: 1 / 0,
        ))
        result = reg.execute("bad", {})
        assert not result.success
        assert "Tool error" in result.output


# ── PKM Tools integration tests ──

class TestPKMTools:
    @pytest.fixture
    def agent(self, tmp_path: Path) -> PKMAgent:
        return PKMAgent(store_path=tmp_path / "tools_test", llm=FakeLLM())

    @pytest.fixture
    def registry(self, agent: PKMAgent) -> ToolRegistry:
        return build_pkm_tools(agent)

    def test_all_tools_registered(self, registry: ToolRegistry):
        names = registry.names()
        assert "learn" in names
        assert "ask" in names
        assert "explore" in names
        assert "search" in names
        assert "connect" in names
        assert "status" in names
        assert "reflect" in names
        assert "list_concepts" in names
        assert "batch_learn" in names
        # New autonomy tools
        assert "run_skill" in names
        assert "list_skills" in names
        assert "consult_claude_code" in names
        assert "consult_codex" in names
        assert "web_search" in names
        assert "web_fetch" in names
        assert "research_topic" in names

    def test_consult_codex_tool(
        self,
        registry: ToolRegistry,
        agent: PKMAgent,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            agent,
            "consult_external_agent",
            lambda provider, prompt, **kwargs: f"{provider}:{prompt}",
        )
        result = registry.execute("consult_codex", {"prompt": "inspect tests"})
        assert result.success
        assert result.output == "codex:inspect tests"

    def test_list_skills_tool(self, registry: ToolRegistry):
        result = registry.execute("list_skills", {})
        assert result.success
        assert "digest_article" in result.output
        assert "analyze_topic" in result.output

    def test_run_skill_without_agentic_mode(self, registry: ToolRegistry):
        # Without init_agentic, run_skill should fail gracefully
        result = registry.execute("run_skill", {"skill_name": "summarize_world"})
        assert result.success is False
        assert "agentic mode" in result.output.lower() or "init_agentic" in result.output.lower()

    def test_web_fetch_invalid_url(self, registry: ToolRegistry):
        result = registry.execute("web_fetch", {"url": "http://invalid.invalid.invalid.example/"})
        assert result.success is False
        assert "error" in result.output.lower()

    def test_web_search_empty(self, registry: ToolRegistry):
        result = registry.execute("web_search", {"query": ""})
        assert result.success
        assert "provide a search query" in result.output.lower()

    def test_web_search_fetch_pages(
        self,
        registry: ToolRegistry,
        agent: PKMAgent,
        monkeypatch: pytest.MonkeyPatch,
    ):
        calls: dict[str, object] = {}

        def fake_search_web(query, *, focus="", max_results=5, domains=None, fetch_pages=False):
            calls["query"] = query
            calls["focus"] = focus
            calls["max_results"] = max_results
            calls["domains"] = domains
            calls["fetch_pages"] = fetch_pages
            return "## Web Search: MCP\n\n### Source Glimpses"

        monkeypatch.setattr(agent, "search_web", fake_search_web)
        result = registry.execute("web_search", {
            "query": "MCP",
            "focus": "Claude Code",
            "domains": "docs.anthropic.com",
            "fetch_pages": True,
        })

        assert result.success
        assert "source glimpses" in result.output.lower()
        assert calls["query"] == "MCP"
        assert calls["focus"] == "Claude Code"
        assert calls["domains"] == "docs.anthropic.com"
        assert calls["fetch_pages"] is True

    def test_learn_tool(self, registry: ToolRegistry):
        result = registry.execute("learn", {"text": "Python is great for ML"})
        assert result.success
        assert "python" in result.output.lower()

    def test_status_tool(self, registry: ToolRegistry):
        result = registry.execute("status", {})
        assert result.success
        assert "concepts" in result.output.lower()

    def test_explore_not_found(self, registry: ToolRegistry):
        result = registry.execute("explore", {"concept_name": "nonexistent"})
        assert not result.success

    def test_connect_tool(self, registry: ToolRegistry):
        result = registry.execute("connect", {
            "source": "python",
            "target": "django",
            "relation_type": "supports",
        })
        assert result.success
        assert "connected" in result.output.lower()

    def test_search_tool(self, registry: ToolRegistry):
        registry.execute("learn", {"text": "Python is great"})
        result = registry.execute("search", {"query": "python"})
        assert result.success

    def test_reflect_tool(self, registry: ToolRegistry):
        result = registry.execute("reflect", {})
        assert result.success

    def test_list_concepts_tool(self, registry: ToolRegistry):
        registry.execute("learn", {"text": "Python is great"})
        result = registry.execute("list_concepts", {})
        assert result.success

    def test_batch_learn_tool(self, registry: ToolRegistry):
        result = registry.execute("batch_learn", {
            "texts": "Python is great === Rust is fast === Go is simple"
        })
        assert result.success
        assert "Chunk 1" in result.output
