"""Tests for MCP modules (client, bridge, manager)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from world0.agents.mcp.client import (
    McpClient,
    McpError,
    McpServerConfig,
    McpStatus,
    McpToolInfo,
    McpResourceInfo,
)
from world0.agents.mcp.bridge import (
    McpToolBridge,
    _normalize_name,
    _qualified_name,
    _schema_to_params,
)
from world0.agents.mcp.manager import (
    McpManager,
    McpHealthReport,
    McpServerStatus,
)
from world0.agents.tools.registry import ToolRegistry


# ── McpServerConfig ─────────────────────────────────────────────────


class TestMcpServerConfig:
    def test_basic(self):
        cfg = McpServerConfig(name="test", command="echo")
        assert cfg.name == "test"
        assert cfg.command == "echo"
        assert cfg.args == []
        assert cfg.env == {}
        assert cfg.timeout == 30.0

    def test_with_args(self):
        cfg = McpServerConfig(
            name="fs", command="npx",
            args=["-y", "@mcp/server-fs", "/tmp"],
            env={"NODE_ENV": "production"},
            timeout=60.0,
        )
        assert len(cfg.args) == 3
        assert cfg.env["NODE_ENV"] == "production"
        assert cfg.timeout == 60.0


# ── McpToolInfo / McpResourceInfo ───────────────────────────────────


class TestMcpInfoTypes:
    def test_tool_info(self):
        info = McpToolInfo(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        assert info.name == "read_file"
        assert "path" in info.input_schema["properties"]

    def test_resource_info(self):
        info = McpResourceInfo(uri="file:///tmp", name="tmp", description="Temp dir")
        assert info.uri == "file:///tmp"


# ── McpClient (unit tests, no real subprocess) ─────────────────────


class TestMcpClient:
    def test_initial_state(self):
        cfg = McpServerConfig(name="test", command="echo")
        client = McpClient(cfg)
        assert client.name == "test"
        assert client.status == McpStatus.DISCONNECTED
        assert client.tools == []
        assert client.resources == []
        assert client.error is None

    def test_disconnect_noop_when_not_connected(self):
        cfg = McpServerConfig(name="test", command="echo")
        client = McpClient(cfg)
        client.disconnect()  # Should not raise
        assert client.status == McpStatus.DISCONNECTED

    def test_list_tools_returns_empty_when_disconnected(self):
        cfg = McpServerConfig(name="test", command="echo")
        client = McpClient(cfg)
        assert client.list_tools() == []

    def test_call_tool_raises_when_disconnected(self):
        cfg = McpServerConfig(name="test", command="echo")
        client = McpClient(cfg)
        with pytest.raises(McpError, match="not connected"):
            client.call_tool("anything")


# ── Bridge helpers ──────────────────────────────────────────────────


class TestBridgeHelpers:
    def test_normalize_name(self):
        assert _normalize_name("my-server") == "my_server"
        assert _normalize_name("Tool Name!") == "tool_name"
        assert _normalize_name("__leading__") == "leading"
        assert _normalize_name("UPPER") == "upper"

    def test_qualified_name(self):
        assert _qualified_name("fs-server", "read_file") == "mcp__fs_server__read_file"
        assert _qualified_name("My Server", "Do Thing") == "mcp__my_server__do_thing"

    def test_schema_to_params_empty(self):
        params = _schema_to_params({})
        assert params == []

    def test_schema_to_params(self):
        schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "encoding": {"type": "string", "description": "Encoding", "enum": ["utf-8", "ascii"]},
            },
            "required": ["path"],
        }
        params = _schema_to_params(schema)
        assert len(params) == 2
        path_param = next(p for p in params if p.name == "path")
        assert path_param.required is True
        assert path_param.type == "string"
        enc_param = next(p for p in params if p.name == "encoding")
        assert enc_param.required is False
        assert enc_param.enum == ["utf-8", "ascii"]


# ── McpToolBridge ───────────────────────────────────────────────────


class TestMcpToolBridge:
    def _make_mock_client(self, name="testserver", tools=None):
        client = MagicMock(spec=McpClient)
        client.name = name
        if tools is None:
            tools = [
                McpToolInfo(
                    name="read_file",
                    description="Read a file",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "Path"}},
                        "required": ["path"],
                    },
                ),
                McpToolInfo(
                    name="write_file",
                    description="Write a file",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                ),
            ]
        client.list_tools.return_value = tools
        client.call_tool.return_value = "result"
        return client

    def test_bridge_server(self):
        registry = ToolRegistry()
        bridge = McpToolBridge(registry)
        client = self._make_mock_client()
        registered = bridge.bridge_server(client)
        assert len(registered) == 2
        assert "mcp__testserver__read_file" in registered
        assert "mcp__testserver__write_file" in registered
        # Tools should be in the registry
        assert registry.get("mcp__testserver__read_file") is not None

    def test_bridge_server_tool_callable(self):
        registry = ToolRegistry()
        bridge = McpToolBridge(registry)
        client = self._make_mock_client()
        bridge.bridge_server(client)
        tool = registry.get("mcp__testserver__read_file")
        result = tool.execute(path="/tmp/test.txt")
        assert result.success is True
        client.call_tool.assert_called_with("read_file", {"path": "/tmp/test.txt"})

    def test_bridge_server_tool_error(self):
        registry = ToolRegistry()
        bridge = McpToolBridge(registry)
        client = self._make_mock_client()
        client.call_tool.side_effect = McpError("connection lost")
        bridge.bridge_server(client)
        tool = registry.get("mcp__testserver__read_file")
        result = tool.execute(path="/tmp/test.txt")
        assert result.success is False
        assert "MCP tool error" in result.output

    def test_unbridge_server(self):
        registry = ToolRegistry()
        bridge = McpToolBridge(registry)
        client = self._make_mock_client()
        bridge.bridge_server(client)
        count = bridge.unbridge_server("testserver")
        assert count == 2

    def test_unbridge_nonexistent(self):
        registry = ToolRegistry()
        bridge = McpToolBridge(registry)
        count = bridge.unbridge_server("nope")
        assert count == 0

    def test_bridged_servers(self):
        registry = ToolRegistry()
        bridge = McpToolBridge(registry)
        client = self._make_mock_client()
        bridge.bridge_server(client)
        servers = bridge.bridged_servers()
        assert "testserver" in servers
        assert len(servers["testserver"]) == 2

    def test_bridged_tool_count(self):
        registry = ToolRegistry()
        bridge = McpToolBridge(registry)
        assert bridge.bridged_tool_count() == 0
        bridge.bridge_server(self._make_mock_client("s1", [
            McpToolInfo(name="t1", description="d", input_schema={}),
        ]))
        assert bridge.bridged_tool_count() == 1


# ── McpHealthReport ────────────────────────────────────────────────


class TestMcpHealthReport:
    def test_healthy(self):
        report = McpHealthReport(total_servers=2, connected=["a", "b"])
        assert report.healthy is True
        assert report.degraded is False

    def test_degraded(self):
        report = McpHealthReport(total_servers=2, connected=["a"], failed=["b"])
        assert report.healthy is False
        assert report.degraded is True

    def test_all_failed(self):
        report = McpHealthReport(total_servers=2, failed=["a", "b"])
        assert report.healthy is False
        assert report.degraded is False

    def test_summary(self):
        report = McpHealthReport(
            total_servers=2, connected=["a"], failed=["b"],
            total_tools=5, errors={"b": "timeout"},
        )
        s = report.summary()
        assert "1/2" in s
        assert "5 tools" in s
        assert "timeout" in s


# ── McpManager ──────────────────────────────────────────────────────


class TestMcpManager:
    def test_add_server(self):
        mgr = McpManager(ToolRegistry())
        mgr.add_server(McpServerConfig(name="fs", command="npx"))
        assert len(mgr) == 1

    def test_remove_server(self):
        mgr = McpManager(ToolRegistry())
        mgr.add_server(McpServerConfig(name="fs", command="npx"))
        mgr.remove_server("fs")
        assert len(mgr) == 0

    def test_remove_nonexistent(self):
        mgr = McpManager(ToolRegistry())
        mgr.remove_server("nope")  # Should not raise

    def test_load_config(self):
        data = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@mcp/server-fs"],
                    "env": {"HOME": "/tmp"},
                },
                "git": {
                    "command": "mcp-git",
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            mgr = McpManager(ToolRegistry())
            count = mgr.load_config(f.name)

        assert count == 2
        assert len(mgr) == 2

    def test_load_config_nonexistent(self):
        mgr = McpManager(ToolRegistry())
        assert mgr.load_config("/nonexistent.json") == 0

    def test_load_config_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("bad json!")
            f.flush()
            mgr = McpManager(ToolRegistry())
            assert mgr.load_config(f.name) == 0

    def test_load_config_skips_invalid_entries(self):
        data = {
            "mcpServers": {
                "valid": {"command": "echo"},
                "no_command": {"args": ["a"]},
                "not_dict": "invalid",
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            mgr = McpManager(ToolRegistry())
            count = mgr.load_config(f.name)

        assert count == 1

    def test_server_statuses_not_started(self):
        mgr = McpManager(ToolRegistry())
        mgr.add_server(McpServerConfig(name="fs", command="npx"))
        statuses = mgr.server_statuses()
        assert len(statuses) == 1
        assert statuses[0].status == "not_started"

    def test_health_no_servers(self):
        mgr = McpManager(ToolRegistry())
        report = mgr.health()
        assert report.total_servers == 0
        assert report.healthy is True

    def test_connected_count_and_total_tools(self):
        mgr = McpManager(ToolRegistry())
        assert mgr.connected_count == 0
        assert mgr.total_tools == 0

    def test_call_tool_invalid_name(self):
        mgr = McpManager(ToolRegistry())
        with pytest.raises(McpError, match="Invalid MCP tool name"):
            mgr.call_tool("bad_name")

    def test_call_tool_server_not_found(self):
        mgr = McpManager(ToolRegistry())
        with pytest.raises(McpError, match="server not found"):
            mgr.call_tool("mcp__nonexistent__tool")
