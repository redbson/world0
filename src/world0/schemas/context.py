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

from pydantic import BaseModel, Field


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
    active_domains: list[str] = Field(default_factory=list)
    # Keyed by the string value of RelationType for JSON-friendliness.
    # Missing keys fall back to the default RELATION_TYPE_FACTOR values
    # used by the activation engine.
    relation_type_weights: dict[str, float] = Field(default_factory=dict)
    # Multiplier applied to concepts whose dominant domain appears in
    # ``active_domains``.  Stacks on top of the task-affinity boost.
    domain_affinity_boost: float = 1.3

    def weight_for(self, relation_type: str, default: float) -> float:
        """Resolve the propagation weight for a relation type under this view."""
        if not self.relation_type_weights:
            return default
        return float(self.relation_type_weights.get(relation_type, default))

    def domain_match(self, domain_label: str) -> bool:
        """True if the given domain label matches this perspective's focus."""
        if not self.active_domains or not domain_label:
            return False
        norm = domain_label.strip().lower()
        return any(d.strip().lower() == norm for d in self.active_domains)
