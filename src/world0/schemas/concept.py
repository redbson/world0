"""ConceptNode — a living cognitive unit with lifecycle."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Maturity(str, Enum):
    """Concept lifecycle stages.

    embryonic  → just extracted, low confidence, might be noise
    developing → reinforced multiple times, stabilizing
    established → high confidence, reliable part of cognition
    core       → central concept, high-frequency activation, dense connections
    fading     → not activated for a long time, decaying
    """

    EMBRYONIC = "embryonic"
    DEVELOPING = "developing"
    ESTABLISHED = "established"
    CORE = "core"
    FADING = "fading"


class ReinforcementEntry(BaseModel):
    """A record of one reinforcement event."""

    timestamp: datetime
    source: str = ""
    task: str = ""


class ConceptNode(BaseModel):
    """A concept is not a static card — it is a living cognitive unit.

    It has confidence, maturity, activation history, and origin. It grows,
    sharpens, merges, or fades as the Agent works.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    domain: str = ""
    tags: list[str] = Field(default_factory=list)

    # Cognitive properties
    confidence: float = Field(default=0.15, ge=0.0, le=1.0)
    maturity: Maturity = Maturity.EMBRYONIC
    activation_count: int = 0
    last_activated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    origin: str = ""
    reinforcement_log: list[ReinforcementEntry] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def normalized_name(self) -> str:
        return self.name.strip().lower()

    def all_names(self) -> list[str]:
        return [self.normalized_name()] + [a.strip().lower() for a in self.aliases]

    def activate(self, source: str = "", task: str = "") -> None:
        """Record an activation event."""
        now = datetime.now(timezone.utc)
        self.activation_count += 1
        self.last_activated = now
        self.reinforcement_log.append(
            ReinforcementEntry(timestamp=now, source=source, task=task)
        )
        # Each activation reinforces confidence (diminishing returns)
        # Tuned so that ~15 activations can reach 0.6 (established threshold)
        boost = 0.06 * (1.0 / (1.0 + self.activation_count * 0.08))
        self.confidence = min(1.0, self.confidence + boost)

        # If fading, revive to developing
        if self.maturity == Maturity.FADING:
            self.maturity = Maturity.DEVELOPING

    def hours_since_activation(self) -> float:
        delta = datetime.now(timezone.utc) - self.last_activated
        return delta.total_seconds() / 3600.0

    def temporal_relevance(self, half_life_hours: float = 168.0) -> float:
        """Time-based relevance score in [0, 1].

        Returns 1.0 for a just-activated concept and decays exponentially
        with a configurable half-life.  A floor of 0.1 prevents ancient
        but structurally important concepts from being completely invisible.

        Args:
            half_life_hours: Hours after which relevance halves.
                Default 168 h (1 week).
        """
        hours = self.hours_since_activation()
        if hours <= 0 or half_life_hours <= 0:
            return 1.0
        raw = math.pow(0.5, hours / half_life_hours)
        return max(0.1, raw)
