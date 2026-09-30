"""RelationEdge — discovered, reinforced, typed connections between concepts."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

from world0.schemas.clock import cognitive_elapsed, wall_now
from world0.schemas.concept import TaskVocabulary, normalize_task_label, task_match_score


class RelationType(str, Enum):
    """Axis-aligned relation categories.

    World 0 models concept links as three cognitive axes:

    - positive: attraction, trust, co-creation, future coupling,
      mutual reinforcement
    - negative: repulsion, conflict, incompatible ontology,
      instability, adversarial prediction
    - parallel: resonance, mutual understanding, conceptual overlap,
      recursive co-modeling, persistent attention allocation
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
    PARALLEL = "parallel"

    # Backward-compatible enum aliases.  Iterating RelationType still yields
    # only the three canonical axes above, while older code comparing against
    # RelationType.SUPPORTS / CONTRASTS / RELATED_TO keeps working.
    CONTAINS = "positive"
    PART_OF = "positive"
    DEPENDS_ON = "positive"
    SUPPORTS = "positive"
    ACTIVATES = "positive"
    PRECEDES = "positive"
    DERIVED_FROM = "positive"
    CONTRASTS = "negative"
    SIMILAR_TO = "parallel"
    RELATED_TO = "parallel"


_LEGACY_RELATION_TYPE_MAP: dict[str, RelationType] = {
    "positive": RelationType.POSITIVE,
    "attraction": RelationType.POSITIVE,
    "trust": RelationType.POSITIVE,
    "co_creation": RelationType.POSITIVE,
    "co-creation": RelationType.POSITIVE,
    "future_coupling": RelationType.POSITIVE,
    "mutual_reinforcement": RelationType.POSITIVE,
    "supports": RelationType.POSITIVE,
    "depends_on": RelationType.POSITIVE,
    "contains": RelationType.POSITIVE,
    "part_of": RelationType.POSITIVE,
    "activates": RelationType.POSITIVE,
    "precedes": RelationType.POSITIVE,
    "derived_from": RelationType.POSITIVE,
    "negative": RelationType.NEGATIVE,
    "repulsion": RelationType.NEGATIVE,
    "conflict": RelationType.NEGATIVE,
    "incompatible_ontology": RelationType.NEGATIVE,
    "instability": RelationType.NEGATIVE,
    "adversarial_prediction": RelationType.NEGATIVE,
    "contrasts": RelationType.NEGATIVE,
    "parallel": RelationType.PARALLEL,
    "resonance": RelationType.PARALLEL,
    "mutual_understanding": RelationType.PARALLEL,
    "deep_conceptual_overlap": RelationType.PARALLEL,
    "recursive_co_modeling": RelationType.PARALLEL,
    "persistent_attention_allocation": RelationType.PARALLEL,
    "similar_to": RelationType.PARALLEL,
    "related_to": RelationType.PARALLEL,
}


# Share of the remaining doubt removed by one explicit re-statement of a
# relation without an attached probability (see ``RelationEdge.confirm``).
EXPLICIT_CONFIRMATION_GAIN: float = 0.05

# Belief that a *negative-axis* claim is correct at the moment it is first
# stated and the extractor attached no probability of its own.
#
# ``SemanticRelationSpec.propagation_strength`` is how strongly activation
# flows along an edge.  On the negative axis it is the gain of the
# *inhibition* channel and is deliberately small (0.05-0.12); reading it as
# a belief made a stated conflict start with a seventh of the belief of a
# stated dependence, lose every contested pair, and — because the relation
# floor is ``RELATION_FLOOR_SHARE x belief`` — fall under the prune
# threshold from the first tick (docs/paper §4.2, note after Proposition
# 4.3).  0.70 is the belief a once-stated ``dependence`` (the paper's
# reference relation, and the weakest positive-axis claim) already carries,
# so a stated claim has the same standing on either axis.  It is a belief
# only: the inhibition gain (``weight``) is untouched.
NEGATIVE_CLAIM_PRIOR: float = 0.70

# A stored negative claim is rebased onto ``NEGATIVE_CLAIM_PRIOR`` only if its
# belief is explained (within this tolerance) by the legacy default seed and
# its own confirm / weaken counters; anything else was set by an extractor or
# by feedback and is kept as stored (``RelationEdge.adopt_claim_prior``).
LEGACY_BELIEF_TOLERANCE: float = 0.02


def disconfirmation_penalty(count: int) -> float:
    """Absolute penalty of the ``count``-th disconfirmation (diminishing)."""
    return 0.06 * (1.0 / (1.0 + count * 0.10))


def _replay_belief(
    seed: float, confirms: int, weakens: int, *, weakens_first: bool = False
) -> float:
    """Belief reached from ``seed`` after bare confirmations / disconfirmations.

    Mirrors ``RelationEdge.confirm`` (closed form) and ``RelationEdge.weaken``
    (with its 0.01 clamp); the order in which they happened is not stored, so
    callers bound it with both orderings.
    """

    def confirm(p: float) -> float:
        return 1.0 - (1.0 - p) * (1.0 - EXPLICIT_CONFIRMATION_GAIN) ** max(0, confirms)

    def weaken(p: float) -> float:
        for i in range(1, max(0, weakens) + 1):
            if p <= 0.01:
                break
            p = max(0.01, p - disconfirmation_penalty(i))
        return p

    p = min(1.0, max(0.0, seed))
    return confirm(weaken(p)) if weakens_first else weaken(confirm(p))


@dataclass(frozen=True)
class SemanticRelationSpec:
    """Deterministic mapping from language relation to axis + scores."""

    name: str
    axis: RelationType
    structural_strength: float
    propagation_strength: float
    description: str

    @property
    def claim_prior(self) -> float:
        """Belief that one explicit statement of this relation is correct.

        ``propagation_strength`` for positive and parallel relations (the
        long-standing default, unchanged); ``NEGATIVE_CLAIM_PRIOR`` for the
        negative axis, whose propagation strength is an inhibition gain and
        not a belief.  Belief and gain are separate quantities: this one
        seeds ``RelationEdge.probability``, the other ``weight``.
        """
        if self.axis == RelationType.NEGATIVE:
            return NEGATIVE_CLAIM_PRIOR
        return self.propagation_strength


SEMANTIC_RELATION_SPECS: dict[str, SemanticRelationSpec] = {
    # Positive / attraction axis
    "membership": SemanticRelationSpec(
        "membership", RelationType.POSITIVE, 0.94, 0.88, "x belongs to A"
    ),
    "inclusion": SemanticRelationSpec(
        "inclusion", RelationType.POSITIVE, 0.92, 0.86, "A contains B"
    ),
    "proper_inclusion": SemanticRelationSpec(
        "proper_inclusion", RelationType.POSITIVE, 0.93, 0.87, "A strictly contains B"
    ),
    "functional_map": SemanticRelationSpec(
        "functional_map", RelationType.POSITIVE, 0.90, 0.84, "f(x) maps to y"
    ),
    "co_creation": SemanticRelationSpec(
        "co_creation", RelationType.POSITIVE, 0.88, 0.82, "concepts jointly produce or shape each other"
    ),
    "mutual_reinforcement": SemanticRelationSpec(
        "mutual_reinforcement", RelationType.POSITIVE, 0.86, 0.82, "concepts strengthen each other's relevance"
    ),
    "future_coupling": SemanticRelationSpec(
        "future_coupling", RelationType.POSITIVE, 0.84, 0.78, "future states or trajectories become coupled"
    ),
    "enables": SemanticRelationSpec(
        "enables", RelationType.POSITIVE, 0.82, 0.76, "one concept enables another"
    ),
    "dependence": SemanticRelationSpec(
        "dependence", RelationType.POSITIVE, 0.78, 0.70, "one concept depends on another under context"
    ),
    # Negative / repulsion axis
    "disjointness": SemanticRelationSpec(
        "disjointness", RelationType.NEGATIVE, 0.95, 0.05, "sets or roles are mutually exclusive"
    ),
    "complement": SemanticRelationSpec(
        "complement", RelationType.NEGATIVE, 0.88, 0.10, "one concept occupies the complement of another"
    ),
    "exclusion": SemanticRelationSpec(
        "exclusion", RelationType.NEGATIVE, 0.90, 0.08, "one concept excludes another"
    ),
    "incompatible_ontology": SemanticRelationSpec(
        "incompatible_ontology", RelationType.NEGATIVE, 0.90, 0.06, "concepts use incompatible modeling commitments"
    ),
    "violates_constraint": SemanticRelationSpec(
        "violates_constraint", RelationType.NEGATIVE, 0.86, 0.08, "a concept violates a constraint or validity region"
    ),
    "conflict": SemanticRelationSpec(
        "conflict", RelationType.NEGATIVE, 0.84, 0.10, "concepts conflict or contradict"
    ),
    "instability": SemanticRelationSpec(
        "instability", RelationType.NEGATIVE, 0.78, 0.12, "one concept destabilizes another"
    ),
    "adversarial_prediction": SemanticRelationSpec(
        "adversarial_prediction", RelationType.NEGATIVE, 0.76, 0.10, "one concept predicts against another"
    ),
    # Parallel / resonance axis
    "equivalence": SemanticRelationSpec(
        "equivalence", RelationType.PARALLEL, 0.96, 0.92, "same under an abstraction, not absolute identity"
    ),
    "quotient_map": SemanticRelationSpec(
        "quotient_map", RelationType.PARALLEL, 0.93, 0.90, "maps into a shared equivalence class"
    ),
    "approximate_equivalence": SemanticRelationSpec(
        "approximate_equivalence", RelationType.PARALLEL, 0.82, 0.74, "near-equivalent under a weaker abstraction"
    ),
    "overlap": SemanticRelationSpec(
        "overlap", RelationType.PARALLEL, 0.66, 0.60, "non-empty conceptual intersection"
    ),
    "similarity_kernel": SemanticRelationSpec(
        "similarity_kernel", RelationType.PARALLEL, 0.70, 0.64, "metric or kernel-induced similarity"
    ),
    "recursive_co_modeling": SemanticRelationSpec(
        "recursive_co_modeling", RelationType.PARALLEL, 0.86, 0.78, "concepts recursively model each other"
    ),
    "persistent_attention": SemanticRelationSpec(
        "persistent_attention", RelationType.PARALLEL, 0.78, 0.70, "concepts persistently allocate attention to each other"
    ),
    "co_membership": SemanticRelationSpec(
        "co_membership", RelationType.PARALLEL, 0.50, 0.45, "concepts share a set or context"
    ),
    "generic_relation": SemanticRelationSpec(
        "generic_relation", RelationType.PARALLEL, 0.55, 0.45, "generic relation incidence without stronger structure"
    ),
}


# Semantic relations on a directed axis whose meaning is nevertheless
# symmetric: "A conflicts with B" says the same as "B conflicts with A".
# They are matched in either orientation and are not scaled by
# direction-conditioned perspectives (docs §7.18).
SYMMETRIC_SEMANTIC_RELATIONS: frozenset[str] = frozenset({
    "co_creation",
    "mutual_reinforcement",
    "future_coupling",
    "conflict",
    "disjointness",
    "complement",
    "incompatible_ontology",
})

_SEMANTIC_RELATION_ALIASES: dict[str, str] = {
    # Canonical names
    **{name: name for name in SEMANTIC_RELATION_SPECS},
    # Axis words default to generic language relations for that axis.
    "positive": "mutual_reinforcement",
    "attraction": "mutual_reinforcement",
    "negative": "conflict",
    "repulsion": "conflict",
    "parallel": "generic_relation",
    "resonance": "overlap",
    # Prior semantic labels.
    "trust": "mutual_reinforcement",
    "co-creation": "co_creation",
    "future_coupling": "future_coupling",
    "mutual_reinforcement": "mutual_reinforcement",
    "conflict": "conflict",
    "incompatible_ontology": "incompatible_ontology",
    "instability": "instability",
    "adversarial_prediction": "adversarial_prediction",
    "mutual_understanding": "equivalence",
    "deep_conceptual_overlap": "overlap",
    "recursive_co_modeling": "recursive_co_modeling",
    "persistent_attention_allocation": "persistent_attention",
    # Legacy relation labels.
    "supports": "enables",
    "depends_on": "dependence",
    "contains": "inclusion",
    "part_of": "membership",
    "activates": "enables",
    "precedes": "dependence",
    "derived_from": "dependence",
    "contrasts": "conflict",
    "similar_to": "similarity_kernel",
    "related_to": "generic_relation",
}


def normalize_relation_type(value: str | RelationType | None) -> RelationType:
    """Coerce current or legacy relation labels onto the three-axis model."""
    if isinstance(value, RelationType):
        return value
    raw = str(value or "").strip().lower()
    if not raw:
        return RelationType.PARALLEL
    normalized = raw.replace(" ", "_")
    return _LEGACY_RELATION_TYPE_MAP.get(normalized, RelationType.PARALLEL)


def is_known_relation_type(value: str | RelationType | None) -> bool:
    """Return True when a label is a canonical axis or known legacy alias."""
    if isinstance(value, RelationType):
        return True
    raw = str(value or "").strip().lower()
    if not raw:
        return False
    normalized = raw.replace(" ", "_")
    return (
        normalized in _LEGACY_RELATION_TYPE_MAP
        or normalized in _SEMANTIC_RELATION_ALIASES
    )


def is_known_relation_label(value: str | RelationType | None) -> bool:
    """True for any label a ``Perspective`` may weight: an axis
    (``positive`` / ``negative`` / ``parallel``), a canonical semantic
    relation (``dependence``, ``inclusion`` …) or one of its aliases
    (``depends_on``, ``contains`` …)."""
    return is_known_relation_type(value)


def canonical_relation_label(value: str | RelationType | None) -> str:
    """Canonical form of a label a ``Perspective`` may weight.

    An axis stays an axis value (``"positive"``); a semantic relation or
    one of its aliases becomes its canonical semantic name
    (``"depends_on"`` → ``"dependence"``).  Raises ``KeyError`` for an
    unknown label.
    """
    if isinstance(value, RelationType):
        return value.value
    raw = str(value or "").strip().lower().replace(" ", "_")
    if raw in {axis.value for axis in RelationType}:
        return raw
    if raw in _SEMANTIC_RELATION_ALIASES:
        return _SEMANTIC_RELATION_ALIASES[raw]
    if raw in _LEGACY_RELATION_TYPE_MAP:
        return _LEGACY_RELATION_TYPE_MAP[raw].value
    raise KeyError(raw)


def normalize_semantic_relation(value: str | None) -> str:
    """Normalize a language relation label to a canonical semantic relation."""
    raw = str(value or "").strip().lower()
    if not raw:
        return "generic_relation"
    key = raw.replace(" ", "_")
    return _SEMANTIC_RELATION_ALIASES.get(key, "generic_relation")


_T = TypeVar("_T")

# Legacy labels phrased from the other end: "A precedes B" states that B
# depends on A.  They are stored in the canonical direction, so a claim
# reads the same whichever label stated it.
_REVERSED_ALIASES: frozenset[str] = frozenset({"precedes"})


def orient_relation(source: _T, target: _T, label: str | None) -> tuple[_T, _T, str]:
    """``(source, target, canonical semantic relation)`` for a stated claim,
    with the endpoints swapped for a label phrased from the other end."""
    key = str(label or "").strip().lower().replace(" ", "_")
    canonical = normalize_semantic_relation(label)
    if key in _REVERSED_ALIASES:
        return target, source, canonical
    return source, target, canonical


def semantic_relation_spec(value: str | None) -> SemanticRelationSpec:
    """Return the score/axis mapping for a language relation label."""
    return SEMANTIC_RELATION_SPECS[normalize_semantic_relation(value)]


# How a claim reads in plain language, source first: "<source> <phrase>
# <target>".  Used by the compact projection render, the form an Agent's
# prompt receives; one entry per canonical semantic relation.
RELATION_PHRASES: dict[str, str] = {
    "membership": "belongs to",
    "inclusion": "contains",
    "proper_inclusion": "strictly contains",
    "functional_map": "maps to",
    "co_creation": "co-creates",
    "mutual_reinforcement": "reinforces",
    "future_coupling": "is coupled in future with",
    "enables": "enables",
    "dependence": "depends on",
    "disjointness": "is disjoint from",
    "complement": "is the complement of",
    "exclusion": "excludes",
    "incompatible_ontology": "is incompatible with",
    "violates_constraint": "violates",
    "conflict": "conflicts with",
    "instability": "destabilizes",
    "adversarial_prediction": "predicts against",
    "equivalence": "is equivalent to",
    "quotient_map": "maps into the same class as",
    "approximate_equivalence": "is roughly equivalent to",
    "overlap": "overlaps with",
    "similarity_kernel": "is similar to",
    "recursive_co_modeling": "co-models",
    "persistent_attention": "attends to",
    "co_membership": "shares a group with",
    "generic_relation": "is related to",
}


def relation_phrase(value: str | None) -> str:
    """Plain-language phrase for a relation label (canonical or alias)."""
    return RELATION_PHRASES[normalize_semantic_relation(value)]


def semantic_relation_names(axis: RelationType | str | None = None) -> list[str]:
    """List canonical language relations, optionally restricted to one axis."""
    if axis is None:
        return sorted(SEMANTIC_RELATION_SPECS)
    relation_axis = normalize_relation_type(axis)
    return sorted(
        name
        for name, spec in SEMANTIC_RELATION_SPECS.items()
        if spec.axis == relation_axis
    )


def relation_axis_descriptions() -> dict[str, list[str]]:
    """Human-facing descriptions for each relation axis."""
    return {
        "positive": [
            "trust",
            "co-creation",
            "future coupling",
            "mutual reinforcement",
        ],
        "negative": [
            "conflict",
            "incompatible ontology",
            "instability",
            "adversarial prediction",
        ],
        "parallel": [
            "mutual understanding",
            "deep conceptual overlap",
            "recursive co-modeling",
            "persistent attention allocation",
        ],
    }


# Most recent distinct tasks a claim is remembered to have been stated under.
MAX_CLAIM_TASKS: int = 32


class RelationEdge(BaseModel):
    """A relation is discovered through the Agent's work, not declared upfront.

    It has provenance, reinforcement history, and can strengthen or weaken.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_id: str
    target_id: str
    relation_type: RelationType = RelationType.PARALLEL
    semantic_relation: str = ""
    structural_strength: float = Field(default=0.55, ge=0.0, le=1.0)
    propagation_strength: float = Field(default=0.45, ge=0.0, le=1.0)
    probability: float = Field(default=0.3, ge=0.0, le=1.0)
    probability_observation_count: int = 0
    # Belief an explicit claim started from: ``SemanticRelationSpec.claim_prior``
    # or the extractor's own prior.  ``None`` marks an edge stored before belief
    # was separated from the propagation gain (a legacy negative claim is
    # rebased once on load, see ``adopt_claim_prior``); it also stays ``None``
    # for co-occurrence edges, which assert nothing.
    belief_prior: float | None = Field(default=None, ge=0.0, le=1.0)
    weight: float = Field(default=0.3, ge=0.0, le=1.0)
    is_explicit: bool = False  # True if declared by Agent, False if Hebbian

    confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    reinforcement_count: int = 0
    disconfirmation_count: int = 0
    last_reinforced: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_weakened: datetime | None = None
    # Cognitive-time coordinates (see ``schemas/clock.py``).
    discovered_tick: int = 0
    last_reinforced_tick: int = 0
    # Tick at which the claim was withdrawn (``Observation.retracted_relations``:
    # "X no longer depends on Y").  A retracted claim is kept as history but
    # is no longer part of the world: it carries no activation, is not a
    # connection, is not in projections, and its weight relaxes toward zero.
    # Restating the claim clears it.
    retracted_tick: int | None = None
    # Instant (both coordinates) at which time decay was last applied (see
    # ``ConceptNode.last_decayed_at``).
    last_decayed_at: datetime | None = None
    last_decayed_tick: int | None = None
    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance: str = ""
    task_history: list[str] = Field(default_factory=list)
    # Tasks under which the claim was *stated* (explicit statements only;
    # ``task_history`` also collects co-occurrence provenance).  The context
    # of a claim is decided by these (projection ``CONTEXT_MATCH``).
    claim_tasks: list[str] = Field(default_factory=list)

    @field_validator("relation_type", mode="before")
    @classmethod
    def _coerce_relation_type(cls, value: object) -> RelationType:
        return normalize_relation_type(value if value is not None else None)

    @field_validator("semantic_relation", mode="before")
    @classmethod
    def _coerce_semantic_relation(cls, value: object) -> str:
        return normalize_semantic_relation(str(value or ""))

    @model_validator(mode="after")
    def _apply_semantic_profile(self) -> "RelationEdge":
        if self.semantic_relation:
            spec = semantic_relation_spec(self.semantic_relation)
        else:
            inferred = {
                RelationType.POSITIVE: "mutual_reinforcement",
                RelationType.NEGATIVE: "conflict",
                RelationType.PARALLEL: "generic_relation",
            }[self.relation_type]
            spec = semantic_relation_spec(inferred)
            self.semantic_relation = spec.name
        self.relation_type = spec.axis
        self.structural_strength = spec.structural_strength
        self.propagation_strength = spec.propagation_strength
        if (
            self.probability_observation_count == 0
            and self.probability == 0.3
            and self.weight == 0.3
            and self.confidence == 0.3
            # A stamped edge was initialised on purpose (an extractor prior
            # of exactly 0.3 is not "unset").
            and self.belief_prior is None
        ):
            # Belief and operational strength are separate quantities: the
            # weight is the propagation (or inhibition) gain of the label; the
            # belief is its claim prior for an explicit statement.
            self.probability = (
                spec.claim_prior if self.is_explicit else spec.propagation_strength
            )
            if self.is_explicit:
                self.belief_prior = self.probability
            self.weight = spec.propagation_strength
            self.confidence = spec.structural_strength
        return self

    def ensure_probability(self) -> None:
        """Backfill probability for relations saved before this field existed."""
        if (
            self.probability_observation_count == 0
            and self.probability == 0.3
            and self.confidence != 0.3
            # Stamped edges postdate the probability field: a belief of
            # exactly 0.3 (an extractor's prior) is a belief, not "missing".
            and self.belief_prior is None
        ):
            self.probability = self.confidence

    def adopt_claim_prior(self) -> bool:
        """Migrate an explicit negative claim stored before belief and gain
        were separated.

        Such an edge started at its inhibition gain (0.05-0.12) as *belief*,
        so a stated conflict carried a seventh of the standing of a stated
        dependence.  If the stored belief is what that legacy seed and the
        edge's own ``confirm`` / ``weaken`` counters explain (within
        ``LEGACY_BELIEF_TOLERANCE``, for either order of the two), the belief
        is rebased: the same counters are replayed from the label's claim
        prior instead (confirmations first, then disconfirmations), and the
        result is kept only if it is higher.  A belief the legacy seed does
        not explain came from an extractor's own probability or from
        feedback, and is kept as stored.  ``weight`` (the inhibition gain) is
        never touched, and neither is any positive, parallel or co-occurrence
        edge.  The edge is stamped with ``belief_prior`` so this runs once.
        Returns True when the edge was stamped (and so should be persisted).
        """
        if (
            self.belief_prior is not None
            or not self.is_explicit
            or self.relation_type != RelationType.NEGATIVE
        ):
            return False
        confirms = self.probability_observation_count
        weakens = self.disconfirmation_count
        seed = self.propagation_strength  # what the legacy code seeded belief from
        ends = (
            _replay_belief(seed, confirms, weakens),
            _replay_belief(seed, confirms, weakens, weakens_first=True),
        )
        explained = (
            min(ends) - LEGACY_BELIEF_TOLERANCE
            <= self.probability
            <= max(ends) + LEGACY_BELIEF_TOLERANCE
        )
        if explained:
            prior = semantic_relation_spec(self.semantic_relation).claim_prior
            self.probability = max(
                self.probability, _replay_belief(prior, confirms, weakens)
            )
            self.belief_prior = prior
        else:
            self.belief_prior = self.probability
        return True

    def involves(self, concept_id: str) -> bool:
        return self.source_id == concept_id or self.target_id == concept_id

    def connects(self, id_a: str, id_b: str, *, directed: bool = False) -> bool:
        """Whether this edge links ``id_a`` and ``id_b``.

        With ``directed=True`` a directed (positive / negative) edge must
        run ``id_a → id_b``; a parallel edge has no meaningful orientation
        and matches either way.
        """
        if directed and self.is_directed:
            return self.source_id == id_a and self.target_id == id_b
        return {self.source_id, self.target_id} == {id_a, id_b}

    def opposes(self, other_axis: RelationType, other_semantic: str = "") -> bool:
        """Whether a claim on ``other_axis`` contradicts this edge's claim.

        A negative claim (conflict, exclusion, …) about a pair contradicts
        a positive or parallel one and vice versa.  ``generic_relation``
        asserts nothing beyond "related", so it neither contradicts nor is
        contradicted — pass the other claim's ``other_semantic`` so the
        relation stays symmetric (docs/paper, Proposition 4.4).
        """
        if self.semantic_relation == "generic_relation":
            return False
        if other_semantic and normalize_semantic_relation(other_semantic) == "generic_relation":
            return False
        mine_negative = self.relation_type == RelationType.NEGATIVE
        return mine_negative != (other_axis == RelationType.NEGATIVE)

    def decay_reference_time(self) -> datetime:
        """Wall-clock instant from which the next decay interval is measured."""
        if self.last_decayed_at and self.last_decayed_at > self.last_reinforced:
            return self.last_decayed_at
        return self.last_reinforced

    def decay_reference_tick(self) -> int:
        """Tick from which the next decay interval is measured."""
        if (
            self.last_decayed_tick is not None
            and self.last_decayed_tick > self.last_reinforced_tick
        ):
            return self.last_decayed_tick
        return self.last_reinforced_tick

    def elapsed_since_reinforced(
        self, now_tick: int | None = None, now: datetime | None = None
    ) -> float:
        """Cognitive time since the last reinforcement, in ticks."""
        return cognitive_elapsed(
            self.last_reinforced_tick if now_tick is None else now_tick,
            self.last_reinforced_tick,
            now or wall_now(),
            self.last_reinforced,
        )

    def decay_elapsed(self, now_tick: int, now: datetime | None = None) -> float:
        """Cognitive time since decay was last applied (or since reinforcement)."""
        return cognitive_elapsed(
            now_tick,
            self.decay_reference_tick(),
            now or wall_now(),
            self.decay_reference_time(),
        )

    def task_affinity(self, task: str, vocabulary: TaskVocabulary | None = None) -> float:
        """Graded association between this relation and ``task`` in [0, 1].

        Word-level match against the tasks under which the relation was
        observed (``task_history``); see ``task_match_score``.
        """
        best = 0.0
        for label in self.task_history:
            score = task_match_score(task, label, vocabulary)
            if score > best:
                best = score
                if best >= 1.0:
                    break
        return best

    @property
    def is_retracted(self) -> bool:
        return self.retracted_tick is not None

    def record_claim(self, task: str) -> None:
        """Note that the claim was stated under ``task``."""
        label = normalize_task_label(task)
        if label and label not in self.claim_tasks:
            self.claim_tasks.append(label)
            if len(self.claim_tasks) > MAX_CLAIM_TASKS:
                del self.claim_tasks[: len(self.claim_tasks) - MAX_CLAIM_TASKS]

    @property
    def support(self) -> int:
        """Explicit statements of this claim (0 for a co-occurrence edge)."""
        return self.probability_observation_count + 1 if self.is_explicit else 0

    def claim_affinity(self, task: str, vocabulary: TaskVocabulary | None = None) -> float | None:
        """Best match of ``task`` against the tasks the claim was stated under;
        None when the claim was never stated under a task (neutral)."""
        if not self.claim_tasks:
            return None
        return max(task_match_score(task, label, vocabulary) for label in self.claim_tasks)

    def other_end(self, concept_id: str) -> str | None:
        if self.source_id == concept_id:
            return self.target_id
        if self.target_id == concept_id:
            return self.source_id
        return None

    @property
    def is_directed(self) -> bool:
        """Whether source→target orientation carries meaning.

        Positive and negative relations are directed (``A depends_on B``,
        ``A excludes B``); parallel relations (equivalence, overlap,
        Hebbian co-occurrence) are symmetric and their stored orientation
        is arbitrary, so direction-conditioned propagation ignores them.
        So are the symmetric semantics on a directed axis (``conflict``,
        ``mutual_reinforcement``, … — ``SYMMETRIC_SEMANTIC_RELATIONS``).
        """
        return (
            self.relation_type != RelationType.PARALLEL
            and self.semantic_relation not in SYMMETRIC_SEMANTIC_RELATIONS
        )

    def reinforce(self, provenance: str = "", *, tick: int | None = None) -> None:
        """Strengthen this relation through repeated observation.

        ``tick`` is the world's cognitive time at which the observation
        happened; engines always pass it.
        """
        self.reinforcement_count += 1
        self.last_reinforced = datetime.now(timezone.utc)
        if tick is not None:
            self.last_reinforced_tick = int(tick)
        if provenance:
            self.provenance = provenance
            if provenance not in self.task_history:
                self.task_history.append(provenance)
        # Weight grows with reinforcement (diminishing returns)
        # Hebbian (auto-discovered) relations use steeper diminishing returns
        # and are capped at 0.7 to preserve distinction from explicit relations.
        if self.is_explicit:
            boost = 0.08 * (1.0 / (1.0 + self.reinforcement_count * 0.05))
            cap = 1.0
        else:
            boost = 0.06 * (1.0 / (1.0 + self.reinforcement_count * 0.15))
            cap = 0.7
        self.weight = min(cap, self.weight + boost)
        self.confidence = min(cap, self.confidence + boost)

    def confirm(self, *, gain: float = EXPLICIT_CONFIRMATION_GAIN) -> None:
        """An explicit re-statement of this relation is semantic evidence.

        ``reinforce()`` strengthens the operational weight (Hebbian
        co-occurrence does that too); ``confirm()`` is reserved for an
        Agent or extractor asserting the typed relation again, and moves
        the belief that it is *correct* toward 1 with diminishing returns.
        Twenty bare confirmations take a 0.70 relation to ≈0.89.
        """
        step = max(0.0, min(1.0, gain))
        self.probability = min(1.0, self.probability + (1.0 - self.probability) * step)
        self.probability_observation_count += 1

    def weaken(self, provenance: str = "") -> None:
        """Disconfirmation evidence against this relation.

        Mirrors `reinforce()` in the negative direction.  Hebbian and
        explicit relations use the same diminishing-returns penalty
        profile — the cap asymmetry only matters for growth, not decay.
        """
        self.disconfirmation_count += 1
        self.last_weakened = datetime.now(timezone.utc)
        penalty = disconfirmation_penalty(self.disconfirmation_count)
        self.weight = max(0.01, self.weight - penalty)
        self.confidence = max(0.01, self.confidence - penalty)
        # Disconfirmation is semantic evidence, so the belief that the
        # relation is correct drops by the same penalty.  It must not be
        # overwritten with ``confidence`` — that is a structural-strength
        # scale and copying it could *raise* the probability.
        self.probability = max(0.01, self.probability - penalty)
        if provenance and provenance not in self.task_history:
            self.task_history.append(provenance)

    def update_probability(
        self,
        *,
        evidence_probability: float | None = None,
        prior_probability: float | None = None,
        prior_strength: float = 1.0,
        evidence_strength: float = 2.0,
        provenance: str = "",
        tick: int | None = None,
    ) -> None:
        """Recalculate relation probability from prior + evidence.

        The current probability is treated as accumulated world belief.
        Optional preset probability acts as a lightweight prior for the
        current extraction pass.  Optional evidence probability is the
        extraction model's assessment for this relation occurrence.
        """
        self.ensure_probability()
        total_strength = max(
            1.0,
            2.0
            + float(self.reinforcement_count)
            + float(self.disconfirmation_count)
            + float(self.probability_observation_count),
        )
        total = self.probability * total_strength

        if prior_probability is not None and prior_strength > 0:
            prior = min(1.0, max(0.0, prior_probability))
            total += prior * prior_strength
            total_strength += prior_strength

        if evidence_probability is not None and evidence_strength > 0:
            evidence = min(1.0, max(0.0, evidence_probability))
            total += evidence * evidence_strength
            total_strength += evidence_strength
            self.probability_observation_count += 1
            if evidence >= 0.5:
                self.reinforcement_count += 1
                self.last_reinforced = datetime.now(timezone.utc)
                if tick is not None:
                    self.last_reinforced_tick = int(tick)
            else:
                self.disconfirmation_count += 1
                self.last_weakened = datetime.now(timezone.utc)

        if total_strength <= 0:
            return
        self.probability = min(1.0, max(0.0, total / total_strength))
        # Evidence about the claim re-scales the operational strengths of a
        # positive / parallel edge.  On the negative axis ``weight`` is the
        # inhibition gain (0.05-0.12 by design), not a belief: writing a
        # belief of ~0.7 into it would multiply the inhibition of an
        # extractor-restated conflict by ~3 now that stated negative claims
        # carry a real belief, so the gain is left alone.
        if self.relation_type != RelationType.NEGATIVE:
            self.weight = self.probability
            self.confidence = self.probability
        if provenance:
            self.provenance = provenance
            if provenance not in self.task_history:
                self.task_history.append(provenance)

    def beta_posterior(
        self, prior_alpha: float = 1.0, prior_beta: float = 1.0
    ) -> tuple[float, float]:
        """Beta(α, β) posterior from evidence counts."""
        alpha = prior_alpha + float(self.reinforcement_count)
        beta = prior_beta + float(self.disconfirmation_count)
        return alpha, beta

    def evidence_balance(self) -> float:
        """Posterior mean of reinforcement vs disconfirmation in [0, 1]."""
        alpha, beta = self.beta_posterior()
        total = alpha + beta
        if total <= 0:
            return 0.5
        return alpha / total

    def hours_since_reinforced(self, now: datetime | None = None) -> float:
        reference = now or datetime.now(timezone.utc)
        delta = reference - self.last_reinforced
        return delta.total_seconds() / 3600.0

    def temporal_relevance(
        self,
        half_life: float = 72.0,
        *,
        now_tick: int | None = None,
        now: datetime | None = None,
    ) -> float:
        """Freshness score in [0, 1] as a function of cognitive time.

        Returns 1.0 for a just-reinforced relation and decays
        exponentially in ticks (observations).  More reinforced relations
        use a longer effective half-life (the same scaling used by
        DecayEngine).  A floor of 0.15 keeps structurally significant but
        old relations from disappearing completely during activation.

        Args:
            half_life: Base half-life in ticks (default 72).
            now_tick: The world's current tick; engines pass one value
                for a whole pass.
            now: Wall-clock reference for the drift term.
        """
        if half_life <= 0:
            return 1.0
        elapsed = self.elapsed_since_reinforced(now_tick, now)
        if elapsed <= 0:
            return 1.0
        effective_hl = half_life * (1.0 + self.reinforcement_count * 0.5)
        raw = math.pow(0.5, elapsed / effective_hl)
        return max(0.15, raw)
