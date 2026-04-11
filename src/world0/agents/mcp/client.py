"""MCP Client — JSON-RPC 2.0 over stdio transport.

Inspired by claw-code's McpStdioProcess. Connects to MCP servers
via subprocess, performs protocol handshake, discovers tools, and
executes tool calls.

Protocol: Content-Length framed JSON-RPC 2.0 messages over stdin/stdout.
"""

from __future__ import annotations

import json
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class McpStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class McpToolInfo:
    """A tool discovered from an MCP server."""
    name: str
    description: str = ""
    input_schema: dict = field(default_factory=dict)


@dataclass
class McpResourceInfo:
    """A resource discovered from an MCP server."""
    uri: str
    name: str = ""
    description: str = ""
    mime_type: str = ""


@dataclass
class McpServerConfig:
    """Configuration for connecting to an MCP server."""
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0


class McpError(Exception):
    """MCP protocol or transport error."""


class McpClient:
    """MCP client using stdio transport (JSON-RPC 2.0).

    Usage::

        config = McpServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
        client = McpClient(config)
        client.connect()
        tools = client.list_tools()
        result = client.call_tool("read_file", {"path": "/tmp/test.txt"})
        client.disconnect()
    """

    def __init__(self, config: McpServerConfig) -> None:
        self.config = config
        self.status = McpStatus.DISCONNECTED
        self.server_info: dict = {}
        self.tools: list[McpToolInfo] = []
        self.resources: list[McpResourceInfo] = []
        self.error: str | None = None

        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.config.name

    def connect(self) -> None:
        """Spawn the MCP server subprocess and perform initialization handshake."""
        if self.status == McpStatus.CONNECTED:
            return

        self.status = McpStatus.CONNECTING
        try:
            env = None
            if self.config.env:
                import os
                env = {**os.environ, **self.config.env}

            self._process = subprocess.Popen(
                [self.config.command] + self.config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            # Initialize handshake
            response = self._request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "world0-pkm-agent",
                    "version": "0.2.0",
                },
            })

            self.server_info = response.get("result", {})

            # Send initialized notification
            self._notify("notifications/initialized", {})

            self.status = McpStatus.CONNECTED
            self.error = None

            # Auto-discover tools
            self._discover_tools()
            self._discover_resources()

        except Exception as e:
            self.status = McpStatus.ERROR
            self.error = str(e)
            self.disconnect()
            raise McpError(f"Failed to connect to {self.name}: {e}") from e

    def disconnect(self) -> None:
        """Shut down the MCP server subprocess."""
        if self._process:
            try:
                self._process.stdin.close()
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
        self.status = McpStatus.DISCONNECTED

    def list_tools(self) -> list[McpToolInfo]:
        """Return discovered tools."""
        return list(self.tools)

    def list_resources(self) -> list[McpResourceInfo]:
        """Return discovered resources."""
        return list(self.resources)

    def call_tool(self, tool_name: str, arguments: dict | None = None) -> Any:
        """Execute a tool call on the MCP server."""
        if self.status != McpStatus.CONNECTED:
            raise McpError(f"Server {self.name} is not connected (status: {self.status.value})")

        params: dict[str, Any] = {"name": tool_name}
        if arguments:
            params["arguments"] = arguments

        response = self._request("tools/call", params)

        if "error" in response:
            err = response["error"]
            raise McpError(f"Tool call failed: {err.get('message', err)}")

        result = response.get("result", {})
        # Extract text content from MCP response format
        content = result.get("content", [])
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
            elif isinstance(item, str):
                texts.append(item)

        return "\n".join(texts) if texts else json.dumps(result)

    def read_resource(self, uri: str) -> str:
        """Read a resource from the MCP server."""
        if self.status != McpStatus.CONNECTED:
            raise McpError(f"Server {self.name} is not connected")

        response = self._request("resources/read", {"uri": uri})

        if "error" in response:
            raise McpError(f"Resource read failed: {response['error']}")

        result = response.get("result", {})
        contents = result.get("contents", [])
        texts = []
        for item in contents:
            if isinstance(item, dict):
                texts.append(item.get("text", ""))
        return "\n".join(texts) if texts else json.dumps(result)

    # ── Internal protocol methods ─────────────────────────────────

    def _discover_tools(self) -> None:
        """Discover available tools from the server."""
        try:
            response = self._request("tools/list", {})
            raw_tools = response.get("result", {}).get("tools", [])
            self.tools = [
                McpToolInfo(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                )
                for t in raw_tools
                if t.get("name")
            ]
        except Exception:
            self.tools = []

    def _discover_resources(self) -> None:
        """Discover available resources from the server."""
        try:
            response = self._request("resources/list", {})
            raw = response.get("result", {}).get("resources", [])
            self.resources = [
                McpResourceInfo(
                    uri=r.get("uri", ""),
                    name=r.get("name", ""),
                    description=r.get("description", ""),
                    mime_type=r.get("mimeType", ""),
                )
                for r in raw
                if r.get("uri")
            ]
        except Exception:
            self.resources = []

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and read the response."""
        with self._lock:
            if not self._process or not self._process.stdin or not self._process.stdout:
                raise McpError("No active connection")

            msg = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": method,
                "params": params,
            }
            self._write_message(msg)
            return self._read_message()

    def _notify(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        with self._lock:
            if not self._process or not self._process.stdin:
                return
            msg = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
            self._write_message(msg)

    def _write_message(self, msg: dict) -> None:
        """Write a content-length framed JSON-RPC message to stdin."""
        body = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        self._process.stdin.write(header + body)
        self._process.stdin.flush()

    def _read_message(self) -> dict:
        """Read a content-length framed JSON-RPC message from stdout."""
        stdout = self._process.stdout

        # Read headers until empty line
        content_length = 0
        while True:
            line = stdout.readline()
            if not line:
                raise McpError("Connection closed unexpectedly")
            line_str = line.decode("utf-8").strip()
            if not line_str:
                break
            if line_str.lower().startswith("content-length:"):
                content_length = int(line_str.split(":", 1)[1].strip())

        if content_length == 0:
            raise McpError("Missing Content-Length header")

        # Read body
        body = stdout.read(content_length)
        if len(body) < content_length:
            raise McpError("Incomplete message body")

        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise McpError(f"Invalid JSON response: {e}") from e

    def __del__(self):
        self.disconnect()

    def __repr__(self) -> str:
        return f"McpClient({self.name!r}, status={self.status.value})"
