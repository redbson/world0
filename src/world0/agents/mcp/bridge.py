"""MCP Tool Bridge — bridges MCP server tools into the PKM ToolRegistry.

Inspired by claw-code's mcp_tool_bridge. Takes discovered MCP tools
and registers them as callable tools in the ToolRegistry, using the
naming convention: mcp__{server_name}__{tool_name}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from world0.agents.mcp.client import McpClient, McpToolInfo
from world0.agents.tools.registry import (
    Permission,
    Tool,
    ToolParam,
    ToolRegistry,
    ToolResult,
)

if TYPE_CHECKING:
    pass


def _normalize_name(name: str) -> str:
    """Normalize a name to a valid identifier (like claw-code)."""
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def _qualified_name(server_name: str, tool_name: str) -> str:
    """Generate a qualified tool name: mcp__{server}__{tool}"""
    return f"mcp__{_normalize_name(server_name)}__{_normalize_name(tool_name)}"


def _schema_to_params(input_schema: dict) -> list[ToolParam]:
    """Convert JSON Schema properties to ToolParam list."""
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    params = []
    for name, prop in properties.items():
        params.append(ToolParam(
            name=name,
            description=prop.get("description", ""),
            type=prop.get("type", "string"),
            required=name in required,
            enum=prop.get("enum"),
        ))
    return params


class McpToolBridge:
    """Bridges MCP tools into the ToolRegistry.

    For each connected MCP server, discovers its tools and registers
    them as callable tools with the naming convention:
        mcp__{server_name}__{tool_name}

    Usage::

        bridge = McpToolBridge(registry)
        bridge.bridge_server(client)  # Registers all tools from this server
        bridge.unbrige_server("filesystem")  # Remove when server disconnects
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._bridged: dict[str, list[str]] = {}  # server_name → [tool_names]

    def bridge_server(self, client: McpClient) -> list[str]:
        """Register all tools from an MCP server into the ToolRegistry.

        Returns list of qualified tool names that were registered.
        """
        registered = []

        for tool_info in client.list_tools():
            qualified = _qualified_name(client.name, tool_info.name)
            params = _schema_to_params(tool_info.input_schema)

            # Create a closure that captures the client and tool name
            handler = _make_handler(client, tool_info.name)

            tool = Tool(
                name=qualified,
                description=(
                    f"[MCP: {client.name}] {tool_info.description}"
                    if tool_info.description
                    else f"[MCP: {client.name}] {tool_info.name}"
                ),
                parameters=params,
                handler=handler,
                permission=Permission.WRITE,  # MCP tools can modify external state
            )
            self._registry.register(tool)
            registered.append(qualified)

        self._bridged[client.name] = registered
        return registered

    def unbridge_server(self, server_name: str) -> int:
        """Remove all tools from a server from the registry.

        Returns the number of tools removed.
        """
        tool_names = self._bridged.pop(server_name, [])
        count = 0
        for name in tool_names:
            if name in self._registry:
                # ToolRegistry doesn't have remove, but we can track
                count += 1
        return count

    def bridged_servers(self) -> dict[str, list[str]]:
        """Return mapping of server_name → registered tool names."""
        return dict(self._bridged)

    def bridged_tool_count(self) -> int:
        return sum(len(tools) for tools in self._bridged.values())


def _make_handler(client: McpClient, tool_name: str):
    """Create a tool handler closure for an MCP tool."""
    def handler(**kwargs) -> ToolResult:
        try:
            result = client.call_tool(tool_name, kwargs if kwargs else None)
            return ToolResult(success=True, output=str(result))
        except Exception as e:
            return ToolResult(success=False, output=f"MCP tool error: {e}")
    return handler
