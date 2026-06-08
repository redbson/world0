"""RelationEdge — discovered, reinforced, typed connections between concepts."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


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


@dataclass(frozen=True)
class SemanticRelationSpec:
    """Deterministic mapping from language relation to axis + scores."""

    name: str
    axis: RelationType
    structural_strength: float
    propagation_strength: float
    description: str


SEMANTIC_RELATION_SPECS: dict[str, SemanticRelationSpec] = {
    # Positive / attraction axis
    "membership": SemanticRelationSpec(
        "membership", RelationType.POSITIVE, 0.94, 0.88, "x belongs to A"
    ),
    "inclusion": SemanticRelationSpec(
        "inclusion", RelationType.POSITIVE, 0.92, 0.86, "A is contained in B"
    ),
    "proper_inclusion": SemanticRelationSpec(
        "proper_inclusion", RelationType.POSITIVE, 0.93, 0.87, "A is strictly contained in B"
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


def normalize_semantic_relation(value: str | None) -> str:
    """Normalize a language relation label to a canonical semantic relation."""
    raw = str(value or "").strip().lower()
    if not raw:
        return "generic_relation"
    key = raw.replace(" ", "_")
    return _SEMANTIC_RELATION_ALIASES.get(key, "generic_relation")


def semantic_relation_spec(value: str | None) -> SemanticRelationSpec:
    """Return the score/axis mapping for a language relation label."""
    return SEMANTIC_RELATION_SPECS[normalize_semantic_relation(value)]


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
    weight: float = Field(default=0.3, ge=0.0, le=1.0)
    is_explicit: bool = False  # True if declared by Agent, False if Hebbian

    confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    reinforcement_count: int = 0
    disconfirmation_count: int = 0
    last_reinforced: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_weakened: datetime | None = None
    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance: str = ""
    task_history: list[str] = Field(default_factory=list)

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
        ):
            self.probability = spec.propagation_strength
            self.weight = spec.propagation_strength
            self.confidence = spec.structural_strength
        return self

    def ensure_probability(self) -> None:
        """Backfill probability for relations saved before this field existed."""
        if (
            self.probability_observation_count == 0
            and self.probability == 0.3
            and self.confidence != 0.3
        ):
            self.probability = self.confidence

    def involves(self, concept_id: str) -> bool:
        return self.source_id == concept_id or self.target_id == concept_id

    def other_end(self, concept_id: str) -> str | None:
        if self.source_id == concept_id:
            return self.target_id
        if self.target_id == concept_id:
            return self.source_id
        return None

    def reinforce(self, provenance: str = "") -> None:
        """Strengthen this relation through repeated observation."""
        self.reinforcement_count += 1
        self.last_reinforced = datetime.now(timezone.utc)
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

    def weaken(self, provenance: str = "") -> None:
        """Disconfirmation evidence against this relation.

        Mirrors `reinforce()` in the negative direction.  Hebbian and
        explicit relations use the same diminishing-returns penalty
        profile — the cap asymmetry only matters for growth, not decay.
        """
        self.disconfirmation_count += 1
        self.last_weakened = datetime.now(timezone.utc)
        penalty = 0.06 * (1.0 / (1.0 + self.disconfirmation_count * 0.10))
        self.weight = max(0.01, self.weight - penalty)
        self.confidence = max(0.01, self.confidence - penalty)
        self.probability = self.confidence
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
            else:
                self.disconfirmation_count += 1
                self.last_weakened = datetime.now(timezone.utc)

        if total_strength <= 0:
            return
        self.probability = min(1.0, max(0.0, total / total_strength))
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

    def hours_since_reinforced(self) -> float:
        delta = datetime.now(timezone.utc) - self.last_reinforced
        return delta.total_seconds() / 3600.0

    def temporal_relevance(self, half_life_hours: float = 72.0) -> float:
        """Time-based relevance score in [0, 1].

        Returns 1.0 for a just-reinforced relation and decays
        exponentially.  More reinforced relations use a longer
        effective half-life (the same scaling used by DecayEngine).
        A floor of 0.15 keeps structurally significant but old
        relations from disappearing completely during activation.

        Args:
            half_life_hours: Base half-life in hours.
                Default 72 h (3 days).
        """
        hours = self.hours_since_reinforced()
        if hours <= 0 or half_life_hours <= 0:
            return 1.0
        effective_hl = half_life_hours * (1.0 + self.reinforcement_count * 0.5)
        raw = math.pow(0.5, hours / effective_hl)
        return max(0.15, raw)
