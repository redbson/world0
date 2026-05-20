"""ConceptNode — a living cognitive unit with lifecycle."""

from __future__ import annotations

import colorsys
import hashlib
import math
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# Signature tokenization: lowercase word tokens ≥2 chars, common English
# stopwords removed.  Keeps the set small while preserving domain terms.
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "in", "on",
    "at", "to", "for", "with", "from", "by", "as", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those",
    "it", "its", "has", "have", "had", "not", "no", "do", "does", "did",
    "can", "will", "would", "should", "could", "may", "might", "than",
})


def tokenize_signature(text: str) -> set[str]:
    """Produce the signature token set for a piece of text.

    Two-character tokens are preserved (covers 'go', 'ai', 'ml', etc.).
    """
    if not text:
        return set()
    return {
        tok.lower()
        for tok in _TOKEN_RE.findall(text)
        if len(tok) >= 2 and tok.lower() not in _STOPWORDS
    }


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
    domain_profile: dict[str, float] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    # Cognitive properties
    confidence: float = Field(default=0.15, ge=0.0, le=1.0)
    maturity: Maturity = Maturity.EMBRYONIC
    activation_count: int = 0
    disconfirmation_count: int = 0
    last_activated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_weakened: datetime | None = None
    origin: str = ""
    reinforcement_log: list[ReinforcementEntry] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def normalized_name(self) -> str:
        return self.name.strip().lower()

    def all_names(self) -> list[str]:
        return [self.normalized_name()] + [a.strip().lower() for a in self.aliases]

    def activate(self, source: str = "", task: str = "") -> None:
        """Record an activation event (confirmation evidence)."""
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

    def weaken(self, source: str = "", task: str = "") -> None:
        """Record a disconfirmation event — evidence *against* the concept.

        Mirrors `activate()` but in the opposite direction: the
        diminishing-returns boost becomes a diminishing-returns penalty
        bounded at 0.01.  Lets confidence reflect the balance of
        positive and negative evidence instead of only accumulating.
        """
        now = datetime.now(timezone.utc)
        self.disconfirmation_count += 1
        self.last_weakened = now
        penalty = 0.06 * (1.0 / (1.0 + self.disconfirmation_count * 0.08))
        self.confidence = max(0.01, self.confidence - penalty)

    def beta_posterior(
        self, prior_alpha: float = 1.0, prior_beta: float = 1.0
    ) -> tuple[float, float]:
        """Beta(α, β) posterior from evidence counts.

        Returns the raw (alpha, beta) so downstream code can compute
        posterior mean `α/(α+β)` or credible intervals.  `confidence`
        on the node remains the soft cognitive score used by decay/
        projection; this method exposes the underlying evidence balance
        for callers that need principled uncertainty (e.g. reflection).
        """
        alpha = prior_alpha + float(self.activation_count)
        beta = prior_beta + float(self.disconfirmation_count)
        return alpha, beta

    def evidence_balance(self) -> float:
        """Posterior mean of confirmation vs disconfirmation in [0, 1].

        ``activate()`` uniformly raises this; ``weaken()`` uniformly
        lowers it.  Independent of the soft `confidence` field (which
        is also affected by decay).
        """
        alpha, beta = self.beta_posterior()
        total = alpha + beta
        if total <= 0:
            return 0.5
        return alpha / total

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

    def signature_tokens(self) -> set[str]:
        """Token set used for signature-based identity resolution.

        Combines name, aliases, and description under the same stopword
        filter.  Used by `ConceptManager` to detect likely duplicates
        (e.g. `PostgreSQL` vs `postgres database`) without relying on
        string-equality of `name`.
        """
        tokens = tokenize_signature(self.name)
        for alias in self.aliases:
            tokens |= tokenize_signature(alias)
        if self.description:
            tokens |= tokenize_signature(self.description)
        return tokens

    def signature_similarity(self, other: "ConceptNode") -> float:
        """Jaccard similarity of signature tokens, gated by domain.

        - identical or empty domain on either side → full Jaccard
        - different domains → heavy 0.3 discount (concepts in separate
          domains are rarely the same underlying unit)
        """
        a = self.signature_tokens()
        b = other.signature_tokens()
        if not a or not b:
            return 0.0
        jac = len(a & b) / len(a | b)

        domain_a = self.domain.strip().lower()
        domain_b = other.domain.strip().lower()
        if domain_a and domain_b and domain_a != domain_b:
            jac *= 0.3
        return jac

    def domain_strength(self, domain_label: str) -> float:
        return self.domain_profile.get(domain_label.strip().lower(), 0.0)

    def dominant_domain_strength(self) -> float:
        if not self.domain_profile:
            return 0.0
        return max(self.domain_profile.values())

    def sorted_domain_profile(self) -> list[tuple[str, float]]:
        return sorted(
            self.domain_profile.items(),
            key=lambda item: item[1],
            reverse=True,
        )

    def color_purity(self) -> float:
        """Top component's share of total color load — 1.0 = pure.

        Returns 1.0 for concepts with no color at all: "no color" is
        trivially pure (doc §12.1 "节点颜色纯度").
        """
        if not self.domain_profile:
            return 1.0
        total = sum(self.domain_profile.values())
        if total <= 0:
            return 1.0
        return max(self.domain_profile.values()) / total

    def is_bridge(
        self, *, min_ratio: float = 0.55, min_second: float = 0.08
    ) -> bool:
        """Two or more comparable color components → edge-layer node.

        Doc §8: "桥接概念不是污染, 而是边界层".  We mark a concept as a
        bridge when its second-strongest color is at least
        ``min_ratio`` of the strongest AND itself exceeds a small
        absolute floor so that near-zero noise does not qualify.
        """
        profile = self.sorted_domain_profile()
        if len(profile) < 2:
            return False
        top, second = profile[0][1], profile[1][1]
        if top <= 0 or second < min_second:
            return False
        return (second / top) >= min_ratio

    @staticmethod
    def domain_color_for(domain_label: str) -> str:
        normalized = domain_label.strip().lower()
        if not normalized:
            return "#64748b"

        digest = hashlib.sha1(normalized.encode("utf-8")).digest()
        hue = int.from_bytes(digest[:2], "big") % 360
        saturation = 0.55 + (digest[2] / 255.0) * 0.15
        lightness = 0.48 + (digest[3] / 255.0) * 0.12
        red, green, blue = colorsys.hls_to_rgb(
            hue / 360.0,
            lightness,
            saturation,
        )
        return "#{:02x}{:02x}{:02x}".format(
            round(red * 255),
            round(green * 255),
            round(blue * 255),
        )
