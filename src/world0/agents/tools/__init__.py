"""Tool registry for the PKM Agent — inspired by claw-code's tool system.

Each tool is a typed, self-describing operation that the LLM can invoke
autonomously during an agentic conversation loop.
"""

from world0.agents.tools.registry import ToolRegistry, Tool, ToolResult

__all__ = ["ToolRegistry", "Tool", "ToolResult"]
