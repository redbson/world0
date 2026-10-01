"""World 0 as an MCP server (``docs/world0-api.md`` §5; TODO P2-11).

Exposes the unified operations as tools named ``world0.<op>`` over stdio
JSON-RPC 2.0 with Content-Length framing — the same transport
``world0.agents.mcp.client.McpClient`` speaks, so one World 0 can be a
tool of another Agent (or of another World 0).  Tool input schemas are
the request models of ``world0.ops``; a ``project`` result is returned as
the prompt-ready text content with the structured JSON as
``structuredContent``.

    python -m world0.agents.mcp.server --store .world0
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, BinaryIO

from world0 import __version__
from world0.api import API_VERSION
from world0.ops import OPERATIONS, ApiError, call, describe, input_schema
from world0.world import World

PROTOCOL_VERSION = "2024-11-05"
TOOL_PREFIX = "world0."


def tool_list() -> list[dict[str, Any]]:
    return [
        {"name": f"{TOOL_PREFIX}{op}", "description": describe(op), "inputSchema": input_schema(op)}
        for op in OPERATIONS
    ]


def call_tool(world: World, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    """MCP ``tools/call`` result for one tool: text content(s), ``isError`` on failure."""
    if not name.startswith(TOOL_PREFIX):
        return {"content": [{"type": "text", "text": json.dumps(
            ApiError("unknown_operation", f"unknown tool {name!r}").as_json())}], "isError": True}
    op = name[len(TOOL_PREFIX):]
    try:
        result = call(world, op, arguments or {})
    except ApiError as exc:
        return {"content": [{"type": "text", "text": json.dumps(exc.as_json(), ensure_ascii=False)}],
                "isError": True}
    # ``project``'s text content is the prompt-ready render (what a tool
    # caller pastes into its context); every result is also returned whole
    # as ``structuredContent``.
    text = result["text"] if op == "project" else json.dumps(result, ensure_ascii=False)
    return {"content": [{"type": "text", "text": text}], "isError": False, "structuredContent": result}


class McpServer:
    """A minimal MCP server: initialize, ping, tools/list, tools/call."""

    def __init__(self, world: World, stdin: BinaryIO | None = None, stdout: BinaryIO | None = None) -> None:
        self.world = world
        self._in = stdin or sys.stdin.buffer
        self._out = stdout or sys.stdout.buffer

    # ── protocol ──────────────────────────────────────────────────────
    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        method = msg.get("method", "")
        params = msg.get("params") or {}
        msg_id = msg.get("id")
        if method == "initialize":
            return self._reply(msg_id, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "world0", "version": __version__, "api": API_VERSION},
            })
        if method == "notifications/initialized" or msg_id is None:
            return None  # notifications have no reply
        if method == "ping":
            return self._reply(msg_id, {})
        if method == "tools/list":
            return self._reply(msg_id, {"tools": tool_list()})
        if method == "tools/call":
            return self._reply(msg_id, call_tool(self.world, params.get("name", ""), params.get("arguments")))
        if method == "resources/list":
            return self._reply(msg_id, {"resources": []})
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"method not found: {method}"}}

    @staticmethod
    def _reply(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    # ── transport ─────────────────────────────────────────────────────
    def serve_forever(self) -> None:
        while True:
            msg = self._read()
            if msg is None:
                break
            reply = self.handle(msg)
            if reply is not None:
                self._write(reply)

    def _write(self, msg: dict[str, Any]) -> None:
        body = json.dumps(msg, ensure_ascii=False).encode("utf-8")
        self._out.write(f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8") + body)
        self._out.flush()

    def _read(self) -> dict[str, Any] | None:
        length = 0
        while True:
            line = self._in.readline()
            if not line:
                return None
            text = line.decode("utf-8").strip()
            if not text:
                break
            if text.lower().startswith("content-length:"):
                length = int(text.split(":", 1)[1].strip())
        if length == 0:
            return None
        body = self._in.read(length)
        if len(body) < length:
            return None
        return json.loads(body.decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m world0.agents.mcp.server", description="World 0 as an MCP server (stdio)")
    p.add_argument("--store", default=".world0")
    p.add_argument("--backend", default="auto", choices=["auto", "json", "sqlite"])
    ns = p.parse_args(argv)
    world = World(store_path=ns.store, backend=ns.backend)
    try:
        McpServer(world).serve_forever()
    finally:
        world.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
