"""MCP Manager — orchestrates multiple MCP server connections.

Inspired by claw-code's McpServerManager and lifecycle management.
Handles configuration loading, server startup, health monitoring,
and graceful degradation when servers fail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from world0.agents.mcp.bridge import McpToolBridge
from world0.agents.mcp.client import (
    McpClient,
    McpError,
    McpServerConfig,
    McpStatus,
)
from world0.agents.tools.registry import ToolRegistry


@dataclass
class McpServerStatus:
    """Status of a managed MCP server."""
    name: str
    status: str
    tool_count: int = 0
    resource_count: int = 0
    error: str | None = None


@dataclass
class McpHealthReport:
    """Health report across all managed MCP servers."""
    total_servers: int = 0
    connected: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    total_tools: int = 0
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return len(self.failed) == 0

    @property
    def degraded(self) -> bool:
        return 0 < len(self.failed) < self.total_servers

    def summary(self) -> str:
        lines = [f"MCP Health: {len(self.connected)}/{self.total_servers} servers connected, {self.total_tools} tools available"]
        if self.failed:
            lines.append(f"Failed: {', '.join(self.failed)}")
            for name, err in self.errors.items():
                lines.append(f"  {name}: {err}")
        return "\n".join(lines)


class McpManager:
    """Manages multiple MCP server connections and bridges their tools.

    Usage::

        manager = McpManager(tool_registry)

        # Add servers from config
        manager.add_server(McpServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        ))

        # Or load from config file
        manager.load_config("~/.pkm_world/mcp.json")

        # Connect all servers
        manager.connect_all()

        # Check health
        report = manager.health()
        print(report.summary())

        # Cleanup
        manager.disconnect_all()
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._bridge = McpToolBridge(registry)
        self._clients: dict[str, McpClient] = {}
        self._configs: dict[str, McpServerConfig] = {}

    def add_server(self, config: McpServerConfig) -> None:
        """Register a server configuration (does not connect yet)."""
        self._configs[config.name] = config

    def remove_server(self, name: str) -> None:
        """Remove and disconnect a server."""
        client = self._clients.pop(name, None)
        if client:
            self._bridge.unbridge_server(name)
            client.disconnect()
        self._configs.pop(name, None)

    def load_config(self, config_path: str | Path) -> int:
        """Load MCP server configurations from a JSON file.

        Expected format::

            {
              "mcpServers": {
                "server_name": {
                  "command": "npx",
                  "args": ["-y", "@modelcontextprotocol/server-xxx"],
                  "env": {"KEY": "value"}
                }
              }
            }

        Returns number of servers loaded.
        """
        path = Path(config_path).expanduser()
        if not path.exists():
            return 0

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return 0

        servers = data.get("mcpServers", {})
        count = 0
        for name, cfg in servers.items():
            if not isinstance(cfg, dict):
                continue
            command = cfg.get("command", "")
            if not command:
                continue

            self.add_server(McpServerConfig(
                name=name,
                command=command,
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
                timeout=cfg.get("timeout", 30.0),
            ))
            count += 1

        return count

    def connect_server(self, name: str) -> bool:
        """Connect a single server by name. Returns True on success."""
        config = self._configs.get(name)
        if not config:
            return False

        client = McpClient(config)
        try:
            client.connect()
            self._clients[name] = client
            # Bridge tools into registry
            self._bridge.bridge_server(client)
            return True
        except McpError:
            # Store client even on failure for status reporting
            self._clients[name] = client
            return False

    def connect_all(self) -> McpHealthReport:
        """Connect all configured servers. Returns health report.

        Uses best-effort: connects as many as possible, reports failures.
        """
        report = McpHealthReport(total_servers=len(self._configs))

        for name in self._configs:
            if self.connect_server(name):
                client = self._clients[name]
                report.connected.append(name)
                report.total_tools += len(client.tools)
            else:
                client = self._clients.get(name)
                report.failed.append(name)
                if client and client.error:
                    report.errors[name] = client.error

        return report

    def disconnect_server(self, name: str) -> None:
        """Disconnect a single server."""
        client = self._clients.pop(name, None)
        if client:
            self._bridge.unbridge_server(name)
            client.disconnect()

    def disconnect_all(self) -> None:
        """Disconnect all servers."""
        for name in list(self._clients.keys()):
            self.disconnect_server(name)

    def reconnect_server(self, name: str) -> bool:
        """Disconnect and reconnect a server."""
        self.disconnect_server(name)
        return self.connect_server(name)

    def health(self) -> McpHealthReport:
        """Get current health status of all servers."""
        report = McpHealthReport(total_servers=len(self._configs))

        for name, config in self._configs.items():
            client = self._clients.get(name)
            if client and client.status == McpStatus.CONNECTED:
                report.connected.append(name)
                report.total_tools += len(client.tools)
            else:
                report.failed.append(name)
                if client and client.error:
                    report.errors[name] = client.error

        return report

    def server_statuses(self) -> list[McpServerStatus]:
        """Get detailed status for each server."""
        statuses = []
        for name, config in self._configs.items():
            client = self._clients.get(name)
            if client:
                statuses.append(McpServerStatus(
                    name=name,
                    status=client.status.value,
                    tool_count=len(client.tools),
                    resource_count=len(client.resources),
                    error=client.error,
                ))
            else:
                statuses.append(McpServerStatus(
                    name=name,
                    status="not_started",
                ))
        return statuses

    def get_client(self, name: str) -> McpClient | None:
        return self._clients.get(name)

    def call_tool(self, qualified_name: str, arguments: dict | None = None) -> Any:
        """Call a tool by its qualified name (mcp__{server}__{tool})."""
        # Parse qualified name
        parts = qualified_name.split("__")
        if len(parts) < 3 or parts[0] != "mcp":
            raise McpError(f"Invalid MCP tool name: {qualified_name}")

        server_name = parts[1]
        tool_name = "__".join(parts[2:])  # Handle tool names with underscores

        client = self._clients.get(server_name)
        if not client:
            # Try fuzzy match
            for name, c in self._clients.items():
                if name.replace("-", "_").replace(" ", "_").lower() == server_name:
                    client = c
                    break

        if not client:
            raise McpError(f"MCP server not found: {server_name}")

        return client.call_tool(tool_name, arguments)

    @property
    def connected_count(self) -> int:
        return sum(
            1 for c in self._clients.values()
            if c.status == McpStatus.CONNECTED
        )

    @property
    def total_tools(self) -> int:
        return sum(len(c.tools) for c in self._clients.values())

    def __len__(self) -> int:
        return len(self._configs)

    def __del__(self):
        self.disconnect_all()
