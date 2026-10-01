"""HTTP surface of the unified API: ``/v1/<op>`` (``docs/world0-api.md`` §5).

``v1_router(world)`` is a FastAPI router that the Agent shell's web app
mounts next to its own ``/api/*`` routes (passing a callable so it follows
the shell's current world), and ``create_app(world)`` serves it alone.
Requests are serialised under one lock: ``World`` is not thread-safe.
Bodies and responses are the JSON of ``world0.ops``; an error is
``{"api": ..., "error": {"code", "message"}}`` with the status of
``HTTP_STATUS``.  FastAPI is an optional dependency (``pip install
world0[web]``); importing this module without it raises ``ImportError``.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Union

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from world0.api import API_VERSION
from world0.ops import HTTP_STATUS, OPERATIONS, ApiError, call, describe
from world0.world import World

_LOCK = threading.Lock()
WorldSource = Union[World, Callable[[], World]]


def _resolve(source: WorldSource) -> World:
    return source() if callable(source) else source


def _run(source: WorldSource, op: str, params: dict[str, Any]) -> JSONResponse:
    """One operation under one lock: ``World`` is not thread-safe and even a
    read settles and may reap, so requests are serialised here."""
    try:
        with _LOCK:
            return JSONResponse(call(_resolve(source), op, params))
    except ApiError as exc:
        return JSONResponse(exc.as_json(), status_code=HTTP_STATUS[exc.code])


async def _body(request: Request) -> dict[str, Any]:
    """The JSON object of the request body; ``{}`` for an empty body."""
    raw = await request.body()
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ApiError("invalid_request", f"body is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ApiError("invalid_request", f"body must be a JSON object, not {type(data).__name__}")
    return data


def _query(**params: Any) -> dict[str, Any]:
    """Query parameters as given (strings); the request model coerces them."""
    return {k: v for k, v in params.items() if v is not None}


def v1_router(world: WorldSource) -> APIRouter:
    """The ``/v1`` routes over ``world`` — a ``World`` or a callable returning
    the current one (the Agent shell swaps worlds when it changes space)."""
    router = APIRouter(prefix="/v1", tags=["world0 API v1"])

    @router.get("")
    async def index() -> dict[str, Any]:
        return {"api": API_VERSION, "operations": {op: describe(op) for op in OPERATIONS}}

    async def posted(op: str, request: Request) -> JSONResponse:
        try:
            params = await _body(request)
        except ApiError as exc:
            return JSONResponse(exc.as_json(), status_code=HTTP_STATUS[exc.code])
        return _run(world, op, params)

    @router.post("/ingest")
    async def ingest(request: Request) -> JSONResponse:
        return await posted("ingest", request)

    @router.post("/ingest-text")
    async def ingest_text(request: Request) -> JSONResponse:
        return await posted("ingest_text", request)

    @router.post("/project")
    async def project(request: Request) -> JSONResponse:
        return await posted("project", request)

    @router.get("/card/{concept}")
    async def card(concept: str) -> JSONResponse:
        return _run(world, "card", {"concept": concept})

    @router.get("/claims/{concept}")
    async def claims(concept: str, task: str = "") -> JSONResponse:
        return _run(world, "claims", {"concept": concept, "task": task})

    @router.get("/find")
    async def find(q: str = "", limit: str | None = None, min_similarity: str | None = None) -> JSONResponse:
        return _run(world, "find", _query(q=q, limit=limit, min_similarity=min_similarity))

    @router.get("/status")
    async def status() -> JSONResponse:
        return _run(world, "status", {})

    @router.post("/reflect")
    async def reflect(request: Request) -> JSONResponse:
        return await posted("reflect", request)

    for verb in ("merge", "split", "weaken", "state", "withdraw", "deny"):
        def _make(op: str):
            async def handler(request: Request) -> JSONResponse:
                return await posted(op, request)
            handler.__name__ = op
            return handler
        router.add_api_route(f"/{verb}", _make(verb), methods=["POST"])

    return router


def create_app(world: World) -> FastAPI:
    app = FastAPI(title="World 0 API", version=API_VERSION, on_shutdown=[world.close])
    app.include_router(v1_router(world))
    return app


__all__ = ["create_app", "v1_router"]
