"""The unified operation table behind every entry point (``docs/world0-api.md`` §3, §5).

Python callers use ``World`` directly.  The CLI (``world0.cli``), the HTTP
surface (``world0.http``) and the MCP server (``world0.agents.mcp.server``)
all go through ``call(world, op, params)`` here, so they cannot drift:
one request model per operation, one JSON result shape, one error
vocabulary.  **Stable** together with the API shapes.

Results are plain JSON-able dicts.  Every result carries ``api`` (the
wire-format version); ``project`` also carries ``text``, the prompt-ready
render, so a surface without a renderer of its own can still show it.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from world0.api import API_VERSION, Statement
from world0.schemas.relation import is_known_relation_type
from world0.schemas.types import Observation
from world0.world import World

ERROR_CODES = (
    "not_found",            # the concept named does not exist
    "invalid_observation",  # the observation does not validate
    "unknown_relation",     # a relation label the world does not know
    "identity_conflict",    # merge / split refused (same concept, missing endpoint, ...)
    "llm_unavailable",      # ingest_text without a provider
    "invalid_request",      # anything else wrong with the parameters
    "unknown_operation",
)

HTTP_STATUS = {
    "not_found": 404, "invalid_observation": 400, "unknown_relation": 400,
    "identity_conflict": 409, "llm_unavailable": 503, "invalid_request": 400,
    "unknown_operation": 404,
}


class ApiError(Exception):
    """An operation failed; ``code`` is one of ``ERROR_CODES``."""

    def __init__(self, code: str, message: str) -> None:
        assert code in ERROR_CODES, code
        super().__init__(message)
        self.code = code
        self.message = message

    def as_json(self) -> dict[str, Any]:
        return {"api": API_VERSION, "error": {"code": self.code, "message": self.message}}


# ── request models (one per operation; their JSON Schema is the MCP inputSchema) ──

class ProjectRequest(BaseModel):
    """Generate the task-relevant local view around ``seeds``."""
    seeds: list[str] = Field(min_length=1, description="Concept names to start from")
    task: str = Field(default="", description="The current task; conditions relevance and splits claims by context")
    perspective: str = Field(default="", description="A perspective profile name (dependency_map, impact_map, taxonomy, analogy, contrast)")
    max_concepts: int = Field(default=15, ge=1, le=200)
    max_depth: int = Field(default=2, ge=1, le=6)


class ConceptRequest(BaseModel):
    """Name (or alias, or id) of one concept."""
    concept: str = Field(min_length=1)


class ClaimsRequest(ConceptRequest):
    """Claims about one concept; ``task`` keeps only those in its context."""
    task: str = ""


class FindRequest(BaseModel):
    """Concepts resembling a text."""
    q: str = Field(min_length=1, description="Text to match against names, aliases and signatures")
    limit: int = Field(default=5, ge=1, le=50)
    min_similarity: float = Field(default=0.3, ge=0.0, le=1.0)


class IngestTextRequest(BaseModel):
    """Extract concepts and claims from text with the configured LLM, then ingest."""
    text: str = Field(min_length=1)
    task: str = ""
    source: str = ""


class ReflectRequest(BaseModel):
    """Consolidate: communities, colour, physical deletion.  ``light`` skips the heavy passes."""
    light: bool = False


class MergeRequest(BaseModel):
    """Two concepts are one; ``absorbed`` folds into ``keeper`` (names or ids)."""
    keeper: str = Field(min_length=1)
    absorbed: str = Field(min_length=1)


class SplitRequest(BaseModel):
    """One concept is two: ``new_name`` is split off ``source``."""
    source: str = Field(min_length=1)
    new_name: str = Field(min_length=1)
    aliases_to_move: list[str] = Field(default_factory=list)
    description: str = ""


class WeakenRequest(ConceptRequest):
    """Doubt about a concept itself (not about a claim)."""
    source: str = ""
    task: str = ""


class StatementRequest(BaseModel):
    """One statement, read as "<source> <relation> <target>"."""
    source: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    target: str = Field(min_length=1)
    task: str = ""
    source_label: str = Field(default="", description="Where the statement comes from")
    belief: float | None = Field(default=None, ge=0.0, le=1.0)


class EmptyRequest(BaseModel):
    pass


# ── operations ──────────────────────────────────────────────────────────

def _check_relations(observation: Observation) -> None:
    for src, tgt, rel in (*observation.relations, *observation.retracted_relations,
                          *observation.contradicted_relations):
        if not is_known_relation_type(rel):
            raise ApiError("unknown_relation", f"unknown relation label {rel!r} in {src!r} → {tgt!r}")


def _observation(params: dict[str, Any]) -> Observation:
    try:
        obs = Observation.model_validate(params)
    except ValidationError as exc:
        raise ApiError("invalid_observation", str(exc)) from exc
    _check_relations(obs)
    return obs


def op_ingest(world: World, params: dict[str, Any]) -> dict[str, Any]:
    """Write one observation (statements, withdrawals, denials, cards)."""
    return world.ingest(_observation(params)).model_dump(mode="json")


def op_ingest_text(world: World, params: dict[str, Any]) -> dict[str, Any]:
    """Extract concepts and claims from text with the configured LLM, then ingest."""
    req = _request(IngestTextRequest, params)
    try:
        result = world.ingest_text(req.text, task=req.task, source=req.source)
    except RuntimeError as exc:
        raise ApiError("llm_unavailable", str(exc)) from exc
    return result.model_dump(mode="json")


def op_project(world: World, params: dict[str, Any]) -> dict[str, Any]:
    """Generate the task-relevant local view around the seeds (claims, cards, what to hold loosely)."""
    req = _request(ProjectRequest, params)
    try:
        projection = world.project(req.seeds, task=req.task, perspective=req.perspective or None,
                                   max_concepts=req.max_concepts, max_depth=req.max_depth)
    except (KeyError, ValueError) as exc:
        raise ApiError("invalid_request", str(exc)) from exc
    data = projection.model_dump(mode="json")
    # Activation is settled at read time and the cognitive clock drifts
    # slowly with the calendar, so two reads seconds apart differ in the
    # ninth decimal; the wire form carries the quantum the projection
    # itself decides ties on (1e-6 of the strongest activation).
    data["activation_scores"] = {k: round(v, 6) for k, v in data["activation_scores"].items()}
    data["text"] = projection.render()
    return data


def op_card(world: World, params: dict[str, Any]) -> dict[str, Any]:
    """One concept's card: name, aliases, sense, description, maturity, evidence, tasks."""
    req = _request(ConceptRequest, params)
    card = world.card(req.concept)
    if card is None:
        raise ApiError("not_found", f"no concept named {req.concept!r}")
    return {"api": API_VERSION, **card.model_dump(mode="json")}


def op_claims(world: World, params: dict[str, Any]) -> dict[str, Any]:
    """Every claim about one concept (current, withdrawn, co-occurrence), optionally in one task's context."""
    req = _request(ClaimsRequest, params)
    card = world.card(req.concept)
    if card is None:
        raise ApiError("not_found", f"no concept named {req.concept!r}")
    claims = world.claims(req.concept, task=req.task)
    return {"api": API_VERSION, "concept": card.name, "concept_id": card.id, "task": req.task,
            "claims": [c.model_dump(mode="json") for c in claims]}


def op_find(world: World, params: dict[str, Any]) -> dict[str, Any]:
    """Concepts whose name, alias or signature resembles a text."""
    req = _request(FindRequest, params)
    matches = world.find(req.q, limit=req.limit, min_similarity=req.min_similarity)
    return {"api": API_VERSION, "q": req.q,
            "matches": [{"card": card.model_dump(mode="json"), "score": round(score, 4)} for card, score in matches]}


def op_status(world: World, params: dict[str, Any]) -> dict[str, Any]:
    """World status: cognitive tick, counts by maturity, communities."""
    _request(EmptyRequest, params)
    return world.status().model_dump(mode="json")


def op_reflect(world: World, params: dict[str, Any]) -> dict[str, Any]:
    """Consolidate: communities, colour, physical deletion of dead objects."""
    req = _request(ReflectRequest, params)
    return world.reflect(light=req.light).model_dump(mode="json")


def op_merge(world: World, params: dict[str, Any]) -> dict[str, Any]:
    """Two concepts are one: fold `absorbed` into `keeper`."""
    req = _request(MergeRequest, params)
    cards = []
    for name in (req.keeper, req.absorbed):
        card = world.card(name)
        if card is None:
            raise ApiError("not_found", f"no concept named {name!r}")
        cards.append(card)
    if cards[0].id == cards[1].id:
        raise ApiError("identity_conflict", f"{req.keeper!r} and {req.absorbed!r} are the same concept")
    if not world.merge(req.keeper, req.absorbed):
        raise ApiError("identity_conflict", f"cannot merge {req.absorbed!r} into {req.keeper!r}")
    return {"api": API_VERSION, "merged": True, "keeper": world.card(req.keeper).model_dump(mode="json")}


def op_split(world: World, params: dict[str, Any]) -> dict[str, Any]:
    """One concept is two: split `new_name` off `source`."""
    req = _request(SplitRequest, params)
    if world.card(req.source) is None:
        raise ApiError("not_found", f"no concept named {req.source!r}")
    new_id = world.split(req.source, req.new_name, aliases_to_move=req.aliases_to_move or None,
                         description=req.description)
    if new_id is None:
        raise ApiError("identity_conflict", f"cannot split {req.new_name!r} off {req.source!r}")
    return {"api": API_VERSION, "split": True, "new": world.card(new_id).model_dump(mode="json")}


def op_weaken(world: World, params: dict[str, Any]) -> dict[str, Any]:
    """Doubt about a concept itself (not about a claim)."""
    req = _request(WeakenRequest, params)
    if world.card(req.concept) is None:
        raise ApiError("not_found", f"no concept named {req.concept!r}")
    ok = world.weaken(req.concept, source=req.source, task=req.task)
    return {"api": API_VERSION, "weakened": bool(ok), "card": world.card(req.concept).model_dump(mode="json")}


def _statement_op(kind: str) -> Callable[[World, dict[str, Any]], dict[str, Any]]:
    def op(world: World, params: dict[str, Any]) -> dict[str, Any]:
        req = _request(StatementRequest, params)
        if not is_known_relation_type(req.relation):
            raise ApiError("unknown_relation", f"unknown relation label {req.relation!r}")
        statement = Statement(req.source, req.relation, req.target, belief=req.belief)
        field = {"state": "statements", "withdraw": "withdrawals", "deny": "denials"}[kind]
        obs = Observation(**{field: [statement]}, task=req.task, source=req.source_label)
        return world.ingest(obs).model_dump(mode="json")
    op.__doc__ = {
        "state": "Ingest one statement: the claim holds.",
        "withdraw": "Ingest one withdrawal: the claim no longer holds and leaves views.",
        "deny": "Ingest one denial: lowers the claim's belief; a denial of a claim nobody made changes nothing.",
    }[kind]
    return op


def _request(model: type[BaseModel], params: dict[str, Any]) -> BaseModel:
    try:
        return model.model_validate(params or {})
    except ValidationError as exc:
        raise ApiError("invalid_request", str(exc)) from exc


#: Operation name → (handler, request model or None for a raw Observation, mutates?).
OPERATIONS: dict[str, tuple[Callable[[World, dict[str, Any]], dict[str, Any]], type[BaseModel] | None, bool]] = {
    "ingest": (op_ingest, None, True),
    "ingest_text": (op_ingest_text, IngestTextRequest, True),
    "project": (op_project, ProjectRequest, False),
    "card": (op_card, ConceptRequest, False),
    "claims": (op_claims, ClaimsRequest, False),
    "find": (op_find, FindRequest, False),
    "status": (op_status, EmptyRequest, False),
    "reflect": (op_reflect, ReflectRequest, True),
    "merge": (op_merge, MergeRequest, True),
    "split": (op_split, SplitRequest, True),
    "weaken": (op_weaken, WeakenRequest, True),
    "state": (_statement_op("state"), StatementRequest, True),
    "withdraw": (_statement_op("withdraw"), StatementRequest, True),
    "deny": (_statement_op("deny"), StatementRequest, True),
}


def call(world: World, op: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run operation ``op`` with ``params`` against ``world``; raises ``ApiError``."""
    entry = OPERATIONS.get(op.replace("-", "_"))
    if entry is None:
        raise ApiError("unknown_operation", f"unknown operation {op!r}; one of {sorted(OPERATIONS)}")
    handler, _model, _mutates = entry
    return handler(world, dict(params or {}))


def input_schema(op: str) -> dict[str, Any]:
    """JSON Schema of ``op``'s parameters (the raw ``Observation`` schema for ``ingest``)."""
    _handler, model, _mutates = OPERATIONS[op.replace("-", "_")]
    schema = (model or Observation).model_json_schema()
    if model is None:
        # The pipeline fields are accepted too, but the API names are what
        # a caller should see first.
        schema["description"] = (
            "An observation: concepts mentioned, statements (source, relation, target[, belief]), "
            "withdrawals, denials, cards, task, source, source_id."
        )
    return schema


def describe(op: str) -> str:
    handler, _model, _mutates = OPERATIONS[op.replace("-", "_")]
    return (handler.__doc__ or op).strip().splitlines()[0]


__all__ = ["ApiError", "ERROR_CODES", "HTTP_STATUS", "OPERATIONS", "call", "describe", "input_schema"]
