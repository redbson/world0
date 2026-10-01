"""HTTP surface of the unified API: ``/v1/<op>`` (``docs/world0-api.md`` §5).

``v1_router(world)`` is a FastAPI router that the Agent shell's web app
mounts next to its own ``/api/*`` routes, and ``create_app(world)`` serves
it alone.  Bodies and responses are the JSON of ``world0.ops``; an error is
``{"api": ..., "error": {"code", "message"}}`` with the status of
``HTTP_STATUS``.  FastAPI is an optional dependency (``pip install
world0[web]``); importing this module without it raises ``ImportError``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from world0.api import API_VERSION
from world0.ops import HTTP_STATUS, ApiError, call
from world0.world import World

GET_OPS = {"card", "claims", "find", "status"}


def _run(world: World, op: str, params: dict[str, Any]) -> JSONResponse:
    try:
        return JSONResponse(call(world, op, params))
    except ApiError as exc:
        return JSONResponse(exc.as_json(), status_code=HTTP_STATUS[exc.code])


def v1_router(world: World) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["world0 API v1"])

    @router.get("")
    def index() -> dict[str, Any]:
        from world0.ops import OPERATIONS, describe
        return {"api": API_VERSION, "operations": {op: describe(op) for op in OPERATIONS}}

    @router.post("/ingest")
    async def ingest(request: Request) -> JSONResponse:
        return _run(world, "ingest", await _body(request))

    @router.post("/ingest-text")
    async def ingest_text(request: Request) -> JSONResponse:
        return _run(world, "ingest_text", await _body(request))

    @router.post("/project")
    async def project(request: Request) -> JSONResponse:
        return _run(world, "project", await _body(request))

    @router.get("/card/{concept}")
    def card(concept: str) -> JSONResponse:
        return _run(world, "card", {"concept": concept})

    @router.get("/claims/{concept}")
    def claims(concept: str, task: str = "") -> JSONResponse:
        return _run(world, "claims", {"concept": concept, "task": task})

    @router.get("/find")
    def find(q: str = "", limit: int = 5, min_similarity: float = 0.3) -> JSONResponse:
        return _run(world, "find", {"q": q, "limit": limit, "min_similarity": min_similarity})

    @router.get("/status")
    def status() -> JSONResponse:
        return _run(world, "status", {})

    @router.post("/reflect")
    async def reflect(request: Request) -> JSONResponse:
        return _run(world, "reflect", await _body(request))

    for verb in ("merge", "split", "weaken", "state", "withdraw", "deny"):
        def _make(op: str):
            async def handler(request: Request) -> JSONResponse:
                return _run(world, op, await _body(request))
            handler.__name__ = op
            return handler
        router.add_api_route(f"/{verb}", _make(verb), methods=["POST"])

    return router


async def _body(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if not raw.strip():
        return {}
    try:
        data = await request.json()
    except ValueError:
        return {"__invalid__": raw.decode("utf-8", "replace")}
    return data if isinstance(data, dict) else {"__invalid__": data}


def create_app(world: World) -> FastAPI:
    app = FastAPI(title="World 0 API", version=API_VERSION, on_shutdown=[world.close])
    app.include_router(v1_router(world))
    return app


__all__ = ["create_app", "v1_router"]
