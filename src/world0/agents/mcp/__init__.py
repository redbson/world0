"""MCP (Model Context Protocol) integration for the PKM Agent.

Connects to external MCP servers, discovers their tools, and bridges
them into the PKM Agent's ToolRegistry for agentic use.
"""

from world0.agents.mcp.client import McpClient, McpServerConfig
from world0.agents.mcp.bridge import McpToolBridge
from world0.agents.mcp.manager import McpManager

__all__ = ["McpClient", "McpServerConfig", "McpToolBridge", "McpManager"]
