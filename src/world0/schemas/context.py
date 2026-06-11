"""Perspective — a first-class task/role context for cognitive projection.

The design philosophy calls for World 0 to be *context sensitive, not
globally static*.  The legacy ``task: str`` parameter reduced context
to a label used only for a fixed task-affinity boost; a Perspective
instead carries everything that should actually shift relevance for
the current view:

- ``task`` / ``role``: who is asking and for what
- ``active_domains``: which domain colors count as in-focus
- ``relation_type_weights``: an overlay on the global
  RELATION_TYPE_FACTOR dict, so the *same* concept-world can produce
  different projections under ``debug`` vs ``design`` frames without
  editing any relation

Perspectives are immutable-by-convention Pydantic models.  Short-lived
ones can be built inline for a single ``project()`` call; stable ones
(e.g. per agent role) can be pickled alongside the store.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from world0.schemas.relation import (
    NEGATIVE_VISIBILITY_DEFAULTS,
    SEMANTIC_RELATION_SPECS,
    normalize_semantic_relation,
)


class Perspective(BaseModel):
    """A task- or role-conditioned view over the concept-world.

    Usage::

        p = Perspective(
            name="debug",
            role="on-call engineer",
            task="triage prod latency",
            active_domains=["observability", "infra"],
            relation_type_weights={
                "positive": 1.2,  # attraction matters most in this frame
                "parallel": 0.7,  # resonance is useful but secondary
                "negative": 1.0,  # repulsion/counter-evidence is fully valued
            },
        )
        proj = world.project(["latency"], perspective=p)
    """

    name: str = "default"
    role: str = ""
    task: str = ""
    description: str = ""
    active_domains: list[str] = Field(default_factory=list)
    # Keyed by the string value of RelationType for JSON-friendliness.
    # Missing keys fall back to the default RELATION_TYPE_FACTOR values
    # used by the activation engine.
    relation_type_weights: dict[str, float] = Field(default_factory=dict)
    # Keyed by canonical semantic relation name ("dependence", "overlap",
    # ...).  A semantic-level weight *replaces* the axis-level weight for
    # edges carrying that relation — it does not stack multiplicatively,
    # so a single override stays explainable instead of compounding with
    # RELATION_TYPE_FACTOR in surprising ways.
    semantic_relation_weights: dict[str, float] = Field(default_factory=dict)
    # Multiplier applied to concepts whose dominant domain appears in
    # ``active_domains``.  Stacks on top of the task-affinity boost.
    domain_affinity_boost: float = 1.3
    # Render style hint consumed by Projection.render() ("default",
    # "compact", "detailed").  Unknown styles fall back to "default".
    render_style: str = "default"
    # Per-semantic-relation projection visibility for negative edges
    # ("suppress" | "expose").  Overrides NEGATIVE_VISIBILITY_DEFAULTS —
    # this is how a role opts in to "conditional" relations (a debug
    # perspective exposing ``conflict``, an ontology-repair perspective
    # exposing ``incompatible_ontology``).
    negative_visibility: dict[str, str] = Field(default_factory=dict)

    @field_validator("semantic_relation_weights", mode="before")
    @classmethod
    def _drop_unknown_semantic_keys(cls, value: object) -> object:
        """Keep only canonical semantic relation names.

        Keys are normalized through the alias table; anything that would
        normalize to ``generic_relation`` without literally *being*
        ``generic_relation`` is a typo or unknown label and is dropped —
        silently coercing it would let a misspelled key override every
        untyped edge in the world.
        """
        if not isinstance(value, dict):
            return value
        cleaned: dict[str, float] = {}
        for key, weight in value.items():
            canonical = normalize_semantic_relation(key)
            # Normalize the literal the same way the alias table does
            # (space → underscore) so "related to" is recognized as the
            # genuine generic_relation alias, not a typo to drop.
            key_norm = str(key).strip().lower().replace(" ", "_")
            if canonical == "generic_relation" and key_norm not in (
                "generic_relation",
                "related_to",
                "parallel",
            ):
                continue
            if canonical in SEMANTIC_RELATION_SPECS:
                cleaned[canonical] = float(weight)
        return cleaned

    def weight_for(self, relation_type: str, default: float) -> float:
        """Resolve the propagation weight for a relation type under this view."""
        if not self.relation_type_weights:
            return default
        return float(self.relation_type_weights.get(relation_type, default))

    def weight_for_relation(
        self, semantic_relation: str, axis: str, default: float
    ) -> float:
        """Resolve the propagation weight for one edge under this view.

        Resolution order: semantic-level override > axis-level override >
        global default.  The semantic weight is a *replacement* on the
        same scale as the axis weight, not a multiplier on top of it.
        """
        if semantic_relation and self.semantic_relation_weights:
            override = self.semantic_relation_weights.get(semantic_relation)
            if override is not None:
                return float(override)
        return self.weight_for(axis, default)

    def visibility_for(self, semantic_relation: str) -> str:
        """Resolve projection visibility for one negative relation.

        Resolution: perspective override > NEGATIVE_VISIBILITY_DEFAULTS
        > "suppress".  "conditional" resolves to "suppress" — it means
        "expose only when a perspective opts in", and an opting-in
        perspective sets an explicit "expose" override.
        """
        policy = self.negative_visibility.get(semantic_relation)
        if policy is None:
            policy = NEGATIVE_VISIBILITY_DEFAULTS.get(
                semantic_relation, "suppress"
            )
        return "expose" if policy == "expose" else "suppress"

    def domain_match(self, domain_label: str) -> bool:
        """True if the given domain label matches this perspective's focus."""
        if not self.active_domains or not domain_label:
            return False
        norm = domain_label.strip().lower()
        return any(d.strip().lower() == norm for d in self.active_domains)


class Context(BaseModel):
    """The active condition for one cognitive operation.

    CLAUDE.md names Context as a top-level concept distinct from
    Perspective: the Perspective is the *lens* (role-conditioned
    weights), the Context is the *moment* — which task is live, through
    which lens, at what time.  It is deliberately tiny; anything that
    looks like memory or workflow state does not belong here.
    """

    task: str = ""
    perspective: Perspective = Field(default_factory=Perspective)
    # Injectable clock for deterministic evaluation.  ``None`` means
    # "use the wall clock", which is the production default.
    now: datetime | None = None

    @property
    def effective_task(self) -> str:
        """The task label that should drive affinity scoring."""
        return self.perspective.task or self.task
