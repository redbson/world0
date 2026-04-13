"""Tests for the agentic conversation loop."""

from __future__ import annotations

from world0.agents.loop import AgentLoop
from world0.agents.provider import ChatResponse, ToolCall
from world0.agents.session import Session
from world0.agents.tools.registry import Tool, ToolParam, ToolRegistry, ToolResult


class FakeChatProvider:
    provider_name = "openai"

    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)

    def chat(self, messages, *, system=None, tools=None):  # noqa: ANN001
        if not self._responses:
            raise AssertionError("No more fake chat responses configured")
        return self._responses.pop(0)


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(
        name="search",
        description="Search knowledge",
        parameters=[ToolParam("query", "query")],
        handler=lambda query: ToolResult(success=True, output=f"Found result for {query}"),
    ))
    registry.register(Tool(
        name="failing_tool",
        description="Always fails",
        parameters=[],
        handler=lambda: ToolResult(success=False, output="tool exploded"),
    ))
    return registry


def test_agent_loop_records_successful_turn_summary():
    session = Session()
    provider = FakeChatProvider([
        ChatResponse(
            content="I will search first.",
            tool_calls=[ToolCall(id="tool1", name="search", arguments={"query": "mcp"})],
            stop_reason="tool_use",
        ),
        ChatResponse(
            content="Search complete.",
            tool_calls=[],
            stop_reason="end_turn",
        ),
    ])

    loop = AgentLoop(provider, build_registry(), session)
    result = loop.run("Find MCP docs")

    assert result == "Search complete."
    latest = session.latest_turn_summary()
    assert latest is not None
    assert latest.stop_reason == "end_turn"
    assert latest.failure_class == "none"
    assert latest.rounds == 2
    assert latest.tool_count == 1


def test_agent_loop_records_failed_tool_summary():
    session = Session()
    provider = FakeChatProvider([
        ChatResponse(
            content="Trying a tool.",
            tool_calls=[ToolCall(id="tool2", name="failing_tool", arguments={})],
            stop_reason="tool_use",
        ),
        ChatResponse(
            content="Tool failed, but here is a fallback answer.",
            tool_calls=[],
            stop_reason="end_turn",
        ),
    ])

    loop = AgentLoop(provider, build_registry(), session)
    result = loop.run("Do the thing")

    assert "fallback answer" in result
    latest = session.latest_turn_summary()
    assert latest is not None
    assert latest.failure_class == "tool_runtime"
    assert latest.failed_tools == ["failing_tool"]
