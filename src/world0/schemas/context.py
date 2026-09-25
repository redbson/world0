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

from pydantic import BaseModel, Field, field_validator

from world0.schemas.relation import (
    RelationType,
    canonical_relation_label,
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
                "dependence": 1.4,  # semantic relation: what latency relies on
                "positive": 1.0,  # every other attraction relation
                "parallel": 0.7,  # resonance is useful but secondary
                "negative": 1.0,  # repulsion/counter-evidence is fully valued
            },
            direction_weights={"forward": 1.0, "backward": 0.4},
        )
        proj = world.project(["latency"], perspective=p)

    ``relation_type_weights`` is keyed by **semantic relation** names
    (``dependence``, ``inclusion``, ``enables`` … and their aliases such
    as ``depends_on`` / ``contains``) or by **axis** (``positive`` /
    ``negative`` / ``parallel``).  A semantic key wins over its axis key;
    an unknown key is rejected at construction so a typo cannot silently
    leave the perspective inert.  Ready-made role profiles live in
    ``world0.perspectives``.
    """

    name: str = "default"
    role: str = ""
    task: str = ""
    active_domains: list[str] = Field(default_factory=list)
    # Keyed by semantic relation name (or alias) or by axis value, for
    # JSON-friendliness.  Missing keys fall back to the default
    # RELATION_TYPE_FACTOR of the relation's axis.
    relation_type_weights: dict[str, float] = Field(default_factory=dict)
    # Multiplier applied to concepts whose dominant domain appears in
    # ``active_domains``.  Stacks on top of the task-affinity boost.
    domain_affinity_boost: float = 1.3
    # Traversal-direction multipliers for propagation.  ``"forward"``
    # follows a relation from its source to its target (``A depends_on B``
    # traversed from A to B: "what do I depend on?"), ``"backward"`` the
    # reverse ("who depends on me?").  Missing keys are neutral (1.0), so
    # the default perspective keeps activation undirected.
    direction_weights: dict[str, float] = Field(default_factory=dict)

    @field_validator("relation_type_weights", mode="before")
    @classmethod
    def _check_relation_keys(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        canonical: dict[str, float] = {}
        unknown: list[str] = []
        for key, weight in value.items():
            try:
                canonical[canonical_relation_label(str(key))] = float(weight)
            except KeyError:
                unknown.append(str(key))
        if unknown:
            raise ValueError(
                "unknown relation label(s) in relation_type_weights: "
                f"{sorted(unknown)} — use a semantic relation name "
                "(e.g. 'dependence', 'inclusion') or an axis "
                "('positive', 'negative', 'parallel')"
            )
        return canonical

    def weight_for(
        self, relation_type: str, default: float, semantic_relation: str = ""
    ) -> float:
        """Resolve the propagation weight for a relation under this view.

        Lookup order: the relation's canonical semantic name, then its
        axis, then ``default`` (keys were canonicalized at construction).
        """
        weights = self.relation_type_weights
        if not weights:
            return default
        if semantic_relation:
            canonical = normalize_semantic_relation(semantic_relation)
            if canonical in weights:
                return float(weights[canonical])
        axis = relation_type.value if isinstance(relation_type, RelationType) else str(relation_type)
        return float(weights.get(axis, default))

    def weight_for_direction(self, direction: str) -> float:
        """Resolve the traversal-direction multiplier (``forward``/``backward``)."""
        if not self.direction_weights:
            return 1.0
        return float(self.direction_weights.get(direction, 1.0))

    def domain_match(self, domain_label: str) -> bool:
        """True if the given domain label matches this perspective's focus."""
        if not self.active_domains or not domain_label:
            return False
        norm = domain_label.strip().lower()
        return any(d.strip().lower() == norm for d in self.active_domains)
