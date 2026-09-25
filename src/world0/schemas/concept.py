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

from pydantic import BaseModel, Field, model_validator

from world0.schemas.clock import cognitive_elapsed, wall_now

# Signature tokenization: lowercase word tokens ≥2 chars, common English
# stopwords removed.  Keeps the set small while preserving domain terms.
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_REPRESENTATION_PART_RE = re.compile(r"[^\w]+", re.UNICODE)
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "in", "on",
    "at", "to", "for", "with", "from", "by", "as", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those",
    "it", "its", "has", "have", "had", "not", "no", "do", "does", "did",
    "can", "will", "would", "should", "could", "may", "might", "than",
})


def normalize_identity_part(value: str) -> str:
    """Normalize one semantic identity component for stable comparison."""
    lowered = value.strip().lower()
    compact = re.sub(r"[_\W]+", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", compact).strip()


def build_concept_identity_key(
    *,
    name: str,
    kind: str = "",
    sense: str = "",
    domain: str = "",
) -> str:
    """Build a stable semantic identity key from non-token context.

    The key intentionally combines lexical label with disambiguating
    semantic fields.  This lets `Apple` as fruit and `Apple` as company
    coexist while repeated observations of the same sense reinforce the
    same concept UID.
    """
    normalized_kind = normalize_identity_part(kind)
    if normalized_kind in {"core", "supporting", "background"}:
        normalized_kind = ""
    parts = [
        normalize_identity_part(name),
        normalized_kind,
        normalize_identity_part(domain),
        normalize_identity_part(sense),
    ]
    payload = "\x1f".join(parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"ck_{digest}"


def representation_part(value: str, *, fallback: str = "concept") -> str:
    """Normalize a value for `token.feature.uid` display keys."""
    compact = _REPRESENTATION_PART_RE.sub("-", value.strip().lower())
    compact = re.sub(r"-+", "-", compact).strip("-")
    return compact or fallback


def tokenize_signature(text: str) -> set[str]:
    """Produce the signature token set for a piece of text.

    Two-character tokens are preserved (covers 'go', 'ai', 'ml', etc.), and
    purely numeric tokens of any length are kept: a version or generation
    number is often the *only* thing separating two concepts ("GPT 4" vs
    "GPT 5", "v1" vs "v2"), and dropping it would make them signature
    twins that consolidation merges.
    """
    if not text:
        return set()
    return {
        tok.lower()
        for tok in _TOKEN_RE.findall(text)
        if (len(tok) >= 2 or tok.isdigit()) and tok.lower() not in _STOPWORDS
    }


# ── Task association ─────────────────────────────────────────────────
# A concept keeps an aggregated ``task_profile`` (normalized task label →
# activation count) instead of relying on an ever-growing event log.  The
# log is kept as a bounded *recent activity* window; the profile is the
# authoritative record of which tasks a concept has served under.
MAX_REINFORCEMENT_LOG: int = 64
MAX_TASK_PROFILE_ENTRIES: int = 64

# ── Recurrence ───────────────────────────────────────────────────────
# Activations are grouped into windows of RECURRENCE_WINDOW ticks (a
# "cognitive day").  ``recurrence_count`` counts the *distinct* windows in
# which a concept was activated, so thirty mentions in one burst count
# once while one mention a window for thirty windows counts thirty times.
# Spaced recurrence is the evidence that a concept is durable rather than
# incidental; the lifecycle uses it as an alternative promotion path and
# a revived FADING concept only returns to DEVELOPING when it has recurred
# at least REVIVAL_RECURRENCE times.
RECURRENCE_WINDOW: int = 24
REVIVAL_RECURRENCE: int = 3

# ── Salience: freshness ∨ evidence-backed persistence ────────────────
# ``salience()`` keeps a well-evidenced concept in view while it is
# dormant: the persistence floor is SALIENCE_EVIDENCE_SHARE × evidence(),
# forgotten on the era scale SALIENCE_ERA_HL (ticks) — the same era the
# decay engine uses for the confidence floor, so both halves of the
# belief forget deep history at one rate.
#
# 0.7 was chosen by sweep (docs/world0-cognitive-dynamics-analysis.md
# §7.1, scripts/sweep_salience.py): the cognitive benchmark is unchanged
# for every share in [0, 1]; in a slot-limited projection a dependency
# confirmed fifty times then dormant keeps a top slot against six fresh
# one-off mentions for ~500 observations, reaches parity around 1 000
# and yields to fresh context after that.  A one-off (evidence ≈ 0.06)
# never rises above the freshness floor, so noise does not persist.
SALIENCE_EVIDENCE_SHARE: float = 0.7
SALIENCE_ERA_HL: float = 4380.0


def normalize_task_label(task: str) -> str:
    """Canonical form of a task label used as a ``task_profile`` key."""
    return re.sub(r"\s+", " ", (task or "").strip().lower())


def task_match_score(query: str, label: str) -> float:
    """Graded match between a task query and a recorded task label.

    Returns a value in ``[0, 1]``:

    - ``1.0`` for an exact (normalized) match;
    - otherwise the fraction of the query's signature tokens that appear
      in the label (``"ml"`` vs ``"ml training"`` → 1.0, ``"ml serving"``
      vs ``"ml training"`` → 0.5);
    - when either side has no signature tokens (very short labels,
      CJK text), a whole-string containment fallback.

    Word-level matching deliberately replaces raw substring matching,
    which let ``"ml"`` match ``"html parsing"``.
    """
    q = normalize_task_label(query)
    lbl = normalize_task_label(label)
    if not q or not lbl:
        return 0.0
    if q == lbl:
        return 1.0
    q_tokens = tokenize_signature(q)
    l_tokens = tokenize_signature(lbl)
    if q_tokens and l_tokens:
        return len(q_tokens & l_tokens) / len(q_tokens)
    return 1.0 if q in lbl else 0.0


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


class ConceptSourceRef(BaseModel):
    """A concept-card pointer to raw source material."""

    source_id: str
    source: str = ""
    task: str = ""
    excerpt: str = ""
    first_seen_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_seen_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    count: int = 1


class ConceptTokenRef(BaseModel):
    """A token/surface-form observation attached to a concept."""

    token: str
    source_id: str = ""
    source: str = ""
    task: str = ""
    excerpt: str = ""
    role: str = "observed"
    first_seen_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_seen_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    count: int = 1


class ConceptNode(BaseModel):
    """A concept is not a static card — it is a living cognitive unit.

    It has confidence, maturity, activation history, and origin. It grows,
    sharpens, merges, or fades as the Agent works.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    kind: str = ""
    sense: str = ""
    identity_key: str = ""
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
    # Cognitive-time coordinates (see ``schemas/clock.py``): the world tick
    # (observation count) at creation and at the last activation.  Decay
    # and freshness are functions of ticks elapsed, with the wall clock as
    # a slow secondary drift.
    created_tick: int = 0
    last_activated_tick: int = 0
    # Distinct RECURRENCE_WINDOW-sized tick windows with an activation, and
    # the last window counted.
    recurrence_count: int = 0
    last_recurrence_window: int = -1
    last_weakened: datetime | None = None
    # Instant (both coordinates) at which time decay was last applied.
    # Lets the decay engine decay only the *elapsed interval* instead of
    # the whole span since the last activation on every call (idempotent
    # decay).
    last_decayed_at: datetime | None = None
    last_decayed_tick: int | None = None
    origin: str = ""
    # Bounded recent-activity window (see MAX_REINFORCEMENT_LOG).
    reinforcement_log: list[ReinforcementEntry] = Field(default_factory=list)
    # Aggregated task association: normalized task label → activation
    # count.  Authoritative for task affinity; never truncated by the
    # log window.
    task_profile: dict[str, int] = Field(default_factory=dict)
    source_refs: list[ConceptSourceRef] = Field(default_factory=list)
    token_refs: list[ConceptTokenRef] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _backfill_task_profile(self) -> "ConceptNode":
        """Rebuild ``task_profile`` from the log for pre-profile records."""
        if not self.task_profile and self.reinforcement_log:
            profile: dict[str, int] = {}
            for entry in self.reinforcement_log:
                label = normalize_task_label(entry.task)
                if label:
                    profile[label] = profile.get(label, 0) + 1
            self.task_profile = profile
        return self

    def normalized_name(self) -> str:
        return self.name.strip().lower()

    def ensure_identity_key(self) -> str:
        """Return a stable semantic identity key, creating one if absent."""
        if not self.identity_key:
            self.identity_key = build_concept_identity_key(
                name=self.name,
                kind=self.kind,
                sense=self.sense or self.description,
                domain=self.domain,
            )
        return self.identity_key

    def representation_feature(self) -> str:
        """Return the semantic feature word used in human-facing IDs."""
        for value in (self.sense, self.kind, self.domain):
            normalized = normalize_identity_part(value)
            if normalized and normalized not in {"core", "supporting", "background"}:
                return normalized
        desc_tokens = [
            token
            for token in tokenize_signature(self.description)
            if token not in tokenize_signature(self.name)
        ]
        if desc_tokens:
            return sorted(desc_tokens)[0]
        return "concept"

    def representation(self) -> str:
        """Human-facing concept representation: ``token.feature.uid``.

        This is a compact display/reference form.  The persistent concept
        UID remains ``id`` and graph relations continue to store ids.
        """
        return ".".join([
            representation_part(self.name, fallback="concept"),
            representation_part(self.representation_feature(), fallback="concept"),
            representation_part(self.id, fallback="uid"),
        ])

    def all_names(self) -> list[str]:
        return [self.normalized_name()] + [a.strip().lower() for a in self.aliases]

    def activate(
        self, source: str = "", task: str = "", *, tick: int | None = None
    ) -> None:
        """Record an activation event (confirmation evidence).

        ``tick`` is the world's cognitive time (observation count) at which
        the activation happened; engines always pass it.  Without it only
        the wall-clock coordinate moves.
        """
        now = datetime.now(timezone.utc)
        self.activation_count += 1
        self.last_activated = now
        if tick is not None:
            self.last_activated_tick = int(tick)
            window = int(tick) // RECURRENCE_WINDOW
            if window != self.last_recurrence_window:
                self.recurrence_count += 1
                self.last_recurrence_window = window
        self.reinforcement_log.append(
            ReinforcementEntry(timestamp=now, source=source, task=task)
        )
        if len(self.reinforcement_log) > MAX_REINFORCEMENT_LOG:
            del self.reinforcement_log[
                : len(self.reinforcement_log) - MAX_REINFORCEMENT_LOG
            ]
        self.record_task(task)
        # Each activation reinforces confidence (diminishing returns)
        # Tuned so that ~15 activations can reach 0.6 (established threshold)
        boost = 0.06 * (1.0 / (1.0 + self.activation_count * 0.08))
        self.confidence = min(1.0, self.confidence + boost)

        # A fading concept revives — to DEVELOPING when it has a history of
        # spaced recurrence, otherwise back to EMBRYONIC so a one-off that
        # faded does not skip the maturity ladder on re-mention.
        if self.maturity == Maturity.FADING:
            self.maturity = (
                Maturity.DEVELOPING
                if self.recurrence_count >= REVIVAL_RECURRENCE
                else Maturity.EMBRYONIC
            )

    def record_task(self, task: str, count: int = 1) -> None:
        """Count one (or ``count``) activation(s) under ``task``."""
        label = normalize_task_label(task)
        if not label or count <= 0:
            return
        self.task_profile[label] = self.task_profile.get(label, 0) + count
        if len(self.task_profile) > MAX_TASK_PROFILE_ENTRIES:
            # Keep the most frequent labels; ties broken lexically so the
            # trimmed profile is identical across processes.
            kept = sorted(
                self.task_profile.items(), key=lambda kv: (-kv[1], kv[0])
            )[:MAX_TASK_PROFILE_ENTRIES]
            self.task_profile = dict(kept)

    def task_affinity(self, task: str) -> float:
        """Graded association between this concept and ``task`` in [0, 1].

        ``1.0`` when the concept has been activated under exactly this
        task (or under a task containing every word of it), a fraction
        for partial word overlap, ``0.0`` when unrelated.  O(#distinct
        tasks) — independent of how often the concept was activated.
        """
        query = normalize_task_label(task)
        if not query or not self.task_profile:
            return 0.0
        if query in self.task_profile:
            return 1.0
        best = 0.0
        for label in self.task_profile:
            score = task_match_score(query, label)
            if score > best:
                best = score
                if best >= 1.0:
                    break
        return best

    def decay_reference_time(self) -> datetime:
        """Wall-clock instant from which the next decay interval is measured."""
        if self.last_decayed_at and self.last_decayed_at > self.last_activated:
            return self.last_decayed_at
        return self.last_activated

    def decay_reference_tick(self) -> int:
        """Tick from which the next decay interval is measured."""
        if (
            self.last_decayed_tick is not None
            and self.last_decayed_tick > self.last_activated_tick
        ):
            return self.last_decayed_tick
        return self.last_activated_tick

    def elapsed_since_activation(
        self, now_tick: int | None = None, now: datetime | None = None
    ) -> float:
        """Cognitive time since the last activation, in ticks.

        Without ``now_tick`` only the wall-clock drift term contributes.
        """
        return cognitive_elapsed(
            self.last_activated_tick if now_tick is None else now_tick,
            self.last_activated_tick,
            now or wall_now(),
            self.last_activated,
        )

    def decay_elapsed(self, now_tick: int, now: datetime | None = None) -> float:
        """Cognitive time since decay was last applied (or since activation)."""
        return cognitive_elapsed(
            now_tick,
            self.decay_reference_tick(),
            now or wall_now(),
            self.decay_reference_time(),
        )

    def record_source_ref(
        self,
        *,
        source_id: str,
        source: str = "",
        task: str = "",
        excerpt: str = "",
    ) -> None:
        """Attach or reinforce a raw source pointer for this concept card."""
        if not source_id:
            return
        now = datetime.now(timezone.utc)
        for ref in self.source_refs:
            if ref.source_id == source_id:
                ref.last_seen_at = now
                ref.count += 1
                if source and not ref.source:
                    ref.source = source
                if task and not ref.task:
                    ref.task = task
                if excerpt and not ref.excerpt:
                    ref.excerpt = excerpt
                return
        self.source_refs.append(
            ConceptSourceRef(
                source_id=source_id,
                source=source,
                task=task,
                excerpt=excerpt,
                first_seen_at=now,
                last_seen_at=now,
            )
        )

    def record_token_ref(
        self,
        *,
        token: str,
        source_id: str = "",
        source: str = "",
        task: str = "",
        excerpt: str = "",
        role: str = "observed",
    ) -> None:
        """Attach or reinforce a token form and its source provenance."""
        clean = token.strip()
        if not clean:
            return
        normalized = normalize_identity_part(clean)
        now = datetime.now(timezone.utc)
        for ref in self.token_refs:
            same_token = normalize_identity_part(ref.token) == normalized
            same_source = ref.source_id == source_id
            same_task = ref.task == task
            if same_token and same_source and same_task:
                ref.last_seen_at = now
                ref.count += 1
                if source and not ref.source:
                    ref.source = source
                if excerpt and not ref.excerpt:
                    ref.excerpt = excerpt
                if role and ref.role == "observed":
                    ref.role = role
                return
        self.token_refs.append(
            ConceptTokenRef(
                token=clean,
                source_id=source_id,
                source=source,
                task=task,
                excerpt=excerpt,
                role=role or "observed",
                first_seen_at=now,
                last_seen_at=now,
            )
        )

    def merged_token_refs(self) -> list[ConceptTokenRef]:
        """Token refs whose token differs from the canonical name."""
        canonical = self.normalized_name()
        return [
            ref
            for ref in self.token_refs
            if normalize_identity_part(ref.token) != canonical
        ]

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

    def evidence(self, saturation_k: float = 10.0) -> float:
        """How well confirmed this concept is, in [0, 1) — independent of time.

        Beta posterior mean of confirmations vs disconfirmations, scaled by
        a saturating count term ``n / (n + k)`` so one lucky observation
        cannot claim near-certainty.  Unlike ``confidence`` this never
        decays: it is the "is this real?" half of the belief.
        """
        n = self.activation_count
        if n <= 0:
            return 0.0
        return self.evidence_balance() * (n / (n + saturation_k))

    def salience(
        self,
        half_life: float = 168.0,
        *,
        now_tick: int | None = None,
        now: datetime | None = None,
    ) -> float:
        """How *present* this concept is in the world right now, in [0.1, 1].

        The "is this relevant now?" half of the belief.  A concept is
        salient either because it was **recently active** (freshness in
        cognitive time, ``temporal_relevance``) or because it is **well
        established** (structural persistence: ``evidence()`` scaled by
        ``SALIENCE_EVIDENCE_SHARE`` and forgotten on the era scale
        ``SALIENCE_ERA_HL``).  The two are combined with ``max`` — the
        stronger reason to keep the concept in view wins.

        Without the persistence term, time would be charged twice against
        a dormant concept during activation (once through the decayed
        ``confidence``, once through freshness) and a dependency confirmed
        fifty times fell below the activation cut after ~1 000 unrelated
        observations while a fresh one-off mention stayed.  A one-off
        concept has evidence ≈ 0.06, so its salience still collapses to
        the freshness floor: noise does not persist.
        """
        fresh = self.temporal_relevance(half_life, now_tick=now_tick, now=now)
        persistence = SALIENCE_EVIDENCE_SHARE * self.evidence()
        if persistence <= fresh:
            return fresh
        elapsed = self.elapsed_since_activation(now_tick, now)
        if elapsed > 0 and SALIENCE_ERA_HL > 0:
            persistence *= math.pow(0.5, elapsed / SALIENCE_ERA_HL)
        return max(fresh, persistence)

    def hours_since_activation(self, now: datetime | None = None) -> float:
        reference = now or datetime.now(timezone.utc)
        delta = reference - self.last_activated
        return delta.total_seconds() / 3600.0

    def temporal_relevance(
        self,
        half_life: float = 168.0,
        *,
        now_tick: int | None = None,
        now: datetime | None = None,
    ) -> float:
        """Freshness score in [0, 1] as a function of cognitive time.

        Returns 1.0 for a just-activated concept and decays exponentially
        with a configurable half-life measured in ticks (observations).
        A floor of 0.1 prevents ancient but structurally important
        concepts from being completely invisible.

        Args:
            half_life: Ticks after which relevance halves (default 168).
            now_tick: The world's current tick.  Engines evaluating many
                concepts in one pass share one ``now_tick``/``now`` so the
                result cannot depend on iteration order.
            now: Wall-clock reference for the drift term.
        """
        if half_life <= 0:
            return 1.0
        elapsed = self.elapsed_since_activation(now_tick, now)
        if elapsed <= 0:
            return 1.0
        raw = math.pow(0.5, elapsed / half_life)
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
