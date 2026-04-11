"""RelationEdge — discovered, reinforced, typed connections between concepts."""

from __future__ import annotations

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
    last_reinforced: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
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

    def hours_since_reinforced(self) -> float:
        delta = datetime.now(timezone.utc) - self.last_reinforced
        return delta.total_seconds() / 3600.0
