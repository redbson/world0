"""The unified API's entry points agree (docs/world0-api.md §5, 0.5 step).

The golden sequence of §8 is written through the operation table, then
read through every surface — Python ``ops.call``, the ``world0`` CLI with
``--json``, HTTP ``/v1``, and the MCP server over stdio through the
project's own ``McpClient`` — and the JSON they return must be identical.
Errors carry the same code everywhere.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest

from world0 import API_VERSION, World
from world0.cli import main as cli_main
from world0.ops import ERROR_CODES, HTTP_STATUS, OPERATIONS, ApiError, call, input_schema

READS = [
    ("project", {"seeds": ["api"], "task": "backend"}),
    ("card", {"concept": "db"}),
    ("claims", {"concept": "api", "task": ""}),
    ("find", {"q": "api", "limit": 5, "min_similarity": 0.3}),
    ("status", {}),
]


def _write_golden(store) -> None:
    w = World(store_path=store)
    call(w, "ingest", {
        "concepts": ["api", "db", "cache"],
        "statements": [{"source": "api", "relation": "depends_on", "target": "db"},
                       {"source": "api", "relation": "conflict", "target": "cache"}],
        "task": "backend", "source": "design review", "source_id": "review-2026-10-01#3",
    })
    call(w, "withdraw", {"source": "api", "relation": "conflict", "target": "cache", "task": "backend"})
    w.close()


@pytest.fixture
def store(tmp_path):
    path = tmp_path / ".world0"
    _write_golden(path)
    return path


def _via_python(store, op, params):
    w = World(store_path=store)
    try:
        return call(w, op, params)
    finally:
        w.close()


def _via_cli(store, op, params):
    argv = ["--store", str(store), "--json", op.replace("_", "-")]
    if op == "project":
        argv += [*params["seeds"], "--task", params["task"]]
    elif op in ("card", "claims"):
        argv += [params["concept"]] + (["--task", params["task"]] if params.get("task") else [])
    elif op == "find":
        argv += [params["q"], "--limit", str(params["limit"])]
    out = io.StringIO()
    code = cli_main(argv, stdout=out)
    return code, json.loads(out.getvalue())


def _http_client(store):
    from fastapi.testclient import TestClient

    from world0.http import create_app
    world = World(store_path=store)
    return TestClient(create_app(world)), world


def _via_http(client, op, params):
    if op == "project":
        return client.post("/v1/project", json=params)
    if op == "card":
        return client.get(f"/v1/card/{params['concept']}")
    if op == "claims":
        return client.get(f"/v1/claims/{params['concept']}", params={"task": params["task"]})
    if op == "find":
        return client.get("/v1/find", params={"q": params["q"], "limit": params["limit"]})
    if op == "status":
        return client.get("/v1/status")
    raise AssertionError(op)


class TestSurfacesAgree:
    def test_cli_returns_the_python_result(self, store):
        for op, params in READS:
            expected = _via_python(store, op, params)
            code, got = _via_cli(store, op, params)
            assert code == 0 and got == expected, op

    def test_http_returns_the_python_result(self, store):
        client, world = _http_client(store)
        try:
            for op, params in READS:
                expected = _via_python(store, op, params)
                resp = _via_http(client, op, params)
                assert resp.status_code == 200 and resp.json() == expected, op
        finally:
            world.close()

    def test_mcp_server_returns_the_python_result(self, store):
        from world0.agents.mcp.client import McpClient, McpServerConfig
        expected = {op: _via_python(store, op, params) for op, params in READS}
        client = McpClient(McpServerConfig(name="world0", command=sys.executable,
                                           args=["-m", "world0.agents.mcp.server", "--store", str(store)]))
        client.connect()
        try:
            tools = {t.name: t for t in client.list_tools()}
            assert set(tools) == {f"world0.{op}" for op in OPERATIONS}
            assert tools["world0.project"].input_schema["required"] == ["seeds"]
            for op, params in READS:
                result = client.call_tool_raw(f"world0.{op}", params)
                assert result["isError"] is False
                assert result["structuredContent"] == expected[op], op
                if op != "project":
                    assert json.loads(result["content"][0]["text"]) == expected[op]
            # the text an Agent sees for a projection is the render itself
            assert client.call_tool("world0.project", READS[0][1]) == expected["project"]["text"]
            err = client.call_tool_raw("world0.card", {"concept": "nobody"})
            assert err["isError"] is True and json.loads(err["content"][0]["text"])["error"]["code"] == "not_found"
        finally:
            client.disconnect()

    def test_project_result_carries_text_and_views(self, store):
        data = _via_python(store, "project", READS[0][1])
        assert data["api"] == API_VERSION
        assert data["claims"][0]["text"] == "api depends on db"
        assert data["no_longer_holds"][0]["status"] == "withdrawn"
        assert data["text"].startswith("## Cognitive Context")


class TestErrorsAgree:
    def test_error_codes_everywhere(self, store):
        cases = [
            ("card", {"concept": "nobody"}, "not_found"),
            ("state", {"source": "a", "relation": "frobnicates", "target": "b"}, "unknown_relation"),
            ("ingest", {"statements": [{"source": "a", "relation": "frobnicates", "target": "b"}]}, "unknown_relation"),
            ("ingest", {"concepts": "not-a-list"}, "invalid_observation"),
            ("project", {"seeds": []}, "invalid_request"),
            ("merge", {"keeper": "api", "absorbed": "api"}, "identity_conflict"),
            ("ingest_text", {"text": "x"}, "llm_unavailable"),
            ("nope", {}, "unknown_operation"),
        ]
        w = World(store_path=store)
        try:
            for op, params, code in cases:
                with pytest.raises(ApiError) as exc:
                    call(w, op, params)
                assert exc.value.code == code, (op, exc.value.message)
                assert exc.value.as_json()["error"]["code"] == code
        finally:
            w.close()
        assert set(HTTP_STATUS) == set(ERROR_CODES)

    def test_http_error_status_and_body(self, store):
        client, world = _http_client(store)
        try:
            resp = client.get("/v1/card/nobody")
            assert resp.status_code == 404 and resp.json()["error"]["code"] == "not_found"
            resp = client.post("/v1/project", json={"seeds": []})
            assert resp.status_code == 400 and resp.json()["error"]["code"] == "invalid_request"
            resp = client.post("/v1/ingest-text", json={"text": "x"})
            assert resp.status_code == 503
            index = client.get("/v1").json()
            assert index["api"] == API_VERSION and set(index["operations"]) == set(OPERATIONS)
        finally:
            world.close()

    def test_cli_error_exit_code_and_json(self, store):
        out = io.StringIO()
        assert cli_main(["--store", str(store), "card", "nobody", "--json"], stdout=out) == 1
        assert json.loads(out.getvalue())["error"]["code"] == "not_found"


class TestWritesThroughSurfaces:
    def test_cli_and_http_writes_match_python(self, tmp_path):
        """The same write through each surface yields the same IngestResult."""
        results = []
        # python
        w = World(store_path=tmp_path / "a"); results.append(call(w, "state", {"source": "x", "relation": "enables", "target": "y", "task": "t"})); w.close()
        # cli
        out = io.StringIO()
        assert cli_main(["--store", str(tmp_path / "b"), "--json", "state", "x", "enables", "y", "--task", "t"], stdout=out) == 0
        results.append(json.loads(out.getvalue()))
        # http
        client, world = _http_client(tmp_path / "c")
        try:
            results.append(client.post("/v1/state", json={"source": "x", "relation": "enables", "target": "y", "task": "t"}).json())
        finally:
            world.close()
        for r in results:
            r.pop("prediction", None)
        assert results[0] == results[1] == results[2]
        assert results[0]["new_relations"] == ["x → enables → y"]

    def test_ingest_from_stdin_and_file(self, tmp_path, monkeypatch):
        obs = {"statements": [{"source": "a", "relation": "contains", "target": "b"}], "task": "t"}
        path = tmp_path / "obs.json"
        path.write_text(json.dumps(obs))
        out = io.StringIO()
        assert cli_main(["--store", str(tmp_path / "s1"), "ingest", str(path), "--json"], stdout=out) == 0
        assert json.loads(out.getvalue())["new_relations"] == ["a → inclusion → b"]
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(obs)))
        out = io.StringIO()
        assert cli_main(["--store", str(tmp_path / "s2"), "ingest", "-"], stdout=out) == 0
        assert "new relations: a → inclusion → b" in out.getvalue()


class TestSchemas:
    def test_every_operation_has_a_schema_and_a_description(self):
        from world0.ops import describe
        for op in OPERATIONS:
            schema = input_schema(op)
            assert schema.get("type") == "object", op
            assert describe(op) and describe(op) != op, op
