"""World 0 as an MCP server (``docs/world0-api.md`` §5; TODO P2-11).

Exposes the unified operations as tools named ``world0_<op>`` over stdio
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

# The newest protocol this server speaks; ``initialize`` echoes the client's
# version when it is one we know (``structuredContent`` needs 2025-06-18).
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
PROTOCOL_VERSION = PROTOCOL_VERSIONS[0]
# Tool names use an underscore: hosts' function-name patterns reject dots.
TOOL_PREFIX = "world0_"


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
        result = call(world, op, arguments if arguments is not None else {})
    except ApiError as exc:
        return {"content": [{"type": "text", "text": json.dumps(exc.as_json(), ensure_ascii=False)}],
                "isError": True}
    except Exception as exc:  # never let a tool call take the server down
        err = ApiError("internal_error", f"{type(exc).__name__}: {exc}")
        return {"content": [{"type": "text", "text": json.dumps(err.as_json(), ensure_ascii=False)}],
                "isError": True}
    # Text content first, as the spec asks (serialised JSON for clients
    # without ``structuredContent``); ``project`` leads with the prompt-ready
    # render so a text-only client can paste it as is.
    contents = []
    if op == "project":
        contents.append({"type": "text", "text": result["text"]})
    contents.append({"type": "text", "text": json.dumps(result, ensure_ascii=False)})
    return {"content": contents, "isError": False, "structuredContent": result}


class _FrameError(Exception):
    """A frame that cannot be read as a JSON-RPC message."""


class McpServer:
    """A minimal MCP server: initialize, ping, tools/list, tools/call."""

    def __init__(self, world: World, stdin: BinaryIO | None = None, stdout: BinaryIO | None = None) -> None:
        self.world = world
        self._in = stdin or sys.stdin.buffer
        self._out = stdout or sys.stdout.buffer

    # ── protocol ──────────────────────────────────────────────────────
    def handle(self, msg: Any) -> dict[str, Any] | None:
        if not isinstance(msg, dict):
            return self._error(None, -32600, "invalid request: expected a JSON-RPC object")
        method = msg.get("method")
        params = msg.get("params") or {}
        msg_id = msg.get("id")
        if not isinstance(method, str):
            return None  # a response or malformed object: nothing to answer
        if msg_id is None:
            return None  # notifications (initialized, cancelled, ...) have no reply
        if not isinstance(params, dict):
            return self._error(msg_id, -32602, "params must be an object")
        if method == "initialize":
            asked = params.get("protocolVersion")
            return self._reply(msg_id, {
                "protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "world0", "version": __version__, "api": API_VERSION},
            })
        if method == "ping":
            return self._reply(msg_id, {})
        if method == "tools/list":
            return self._reply(msg_id, {"tools": tool_list()})
        if method == "tools/call":
            return self._reply(msg_id, call_tool(self.world, params.get("name", ""), params.get("arguments")))
        if method == "resources/list":
            return self._reply(msg_id, {"resources": []})
        return self._error(msg_id, -32601, f"method not found: {method}")

    @staticmethod
    def _reply(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}

    # ── transport ─────────────────────────────────────────────────────
    def serve_forever(self) -> None:
        """Serve until stdin closes.  A frame that cannot be parsed is
        answered with a JSON-RPC parse error and the loop continues."""
        while True:
            try:
                msg = self._read()
            except _FrameError as exc:
                self._write(self._error(None, -32700, f"parse error: {exc}"))
                continue
            if msg is None:
                break
            try:
                reply = self.handle(msg)
            except Exception as exc:  # a bug in a handler must not end the session
                reply = self._error(msg.get("id") if isinstance(msg, dict) else None, -32603,
                                    f"internal error: {type(exc).__name__}: {exc}")
            if reply is not None:
                self._write(reply)

    def _write(self, msg: dict[str, Any]) -> None:
        body = json.dumps(msg, ensure_ascii=False).encode("utf-8")
        self._out.write(f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8") + body)
        self._out.flush()

    def _read(self) -> Any:
        """One framed message, or None at end of input; ``_FrameError`` on a
        bad header or body."""
        length = -1
        saw_header = False
        while True:
            line = self._in.readline()
            if not line:
                return None
            saw_header = True
            text = line.decode("utf-8", "replace").strip()
            if not text:
                break
            if text.lower().startswith("content-length:"):
                try:
                    length = int(text.split(":", 1)[1].strip())
                except ValueError as exc:
                    raise _FrameError(f"bad Content-Length header {text!r}") from exc
        if length < 0:
            if not saw_header:
                return None
            raise _FrameError("missing Content-Length header")
        body = self._in.read(length)
        if len(body) < length:
            return None  # input closed mid-message
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise _FrameError(f"body is not JSON: {exc}") from exc


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
