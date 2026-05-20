"""RelationEdge — discovered, reinforced, typed connections between concepts."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class RelationType(str, Enum):
    """Typed relation categories.

    RELATED_TO is the default when a relation is first discovered.
    The Agent can refine it to a more specific type later.
    """

    CONTAINS = "contains"
    PART_OF = "part_of"
    DEPENDS_ON = "depends_on"
    SUPPORTS = "supports"
    CONTRASTS = "contrasts"
    SIMILAR_TO = "similar_to"
    ACTIVATES = "activates"
    PRECEDES = "precedes"
    DERIVED_FROM = "derived_from"
    RELATED_TO = "related_to"  # default / fallback


class RelationEdge(BaseModel):
    """A relation is discovered through the Agent's work, not declared upfront.

    It has provenance, reinforcement history, and can strengthen or weaken.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_id: str
    target_id: str
    relation_type: RelationType = RelationType.RELATED_TO
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
        if provenance and provenance not in self.task_history:
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
