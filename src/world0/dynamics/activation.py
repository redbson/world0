"""Spreading activation — propagation through the relation network.

Activation factors:
  - concept confidence and maturity
  - relation weight, confidence, and type (type factor is resolved
    *through* the active Perspective, so different frames can produce
    different projections over the same world)
  - task affinity: relations/concepts associated with the current task
    propagate more strongly (graded by word-level task match)
  - domain affinity: concepts whose dominant domain is "in focus" for
    the Perspective receive an extra boost
  - temporal relevance: recently active concepts and relations propagate
    more strongly than stale ones
  - depth decay with configurable falloff
  - propagation floor to prevent low-confidence nodes from blocking spread

Aggregation:
  A concept reached through several paths accumulates their evidence with
  a bounded noisy-OR (``S · (1 − Π(1 − aᵢ / S))``, ``S`` = strongest seed
  score) instead of keeping only the strongest single path.  Convergence —
  the concept that two seeds *both* point at — therefore outscores a
  concept reachable from one seed alone, which is precisely the
  "conceptual intersection" a task projection should surface.  The
  ceiling ``S`` guarantees that no propagated concept outranks the
  strongest seed.

Layering:
  Propagation is breadth-first by layer.  A concept settles at the depth
  where it is first reached and is expanded exactly once; excitation never
  flows back into a shallower layer (seeds are frozen), so cycles cannot
  inflate scores and a recorded activation counts once per concept.

Floor:
  Weak but structurally connected signals are lifted into a narrow band
  just below ``PROPAGATION_MIN_RATIO × S`` rather than clamped to one
  constant, so distant concepts stay in the candidate pool *and* keep
  their relative order (closer / stronger still ranks first).

Inhibition:
  Negative-axis edges spread *negative* activation that accumulates on a
  separate inhibition channel.  The final score returned for each
  concept is ``max(0, activation - inhibition)``, so an aggressively
  repelled neighbor can disappear from projections even if it is
  also weakly activated through other paths.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from world0.dynamics.coefficients import (
    CONCEPT_TEMPORAL_HL,
    RELATION_TEMPORAL_HL,
    RELATION_TYPE_FACTOR,
)
from world0.schemas.clock import CognitiveClock
from world0.schemas.context import Perspective
from world0.schemas.relation import RelationType

if TYPE_CHECKING:
    from world0.core import ConceptStore, RelationStore

# Re-exported for backwards compatibility.  The constants live in
# ``dynamics.coefficients`` so multiple engines can share them without
# importing each other's modules.
__all__ = [
    "ActivationEngine",
    "RELATION_TYPE_FACTOR",
    "CONCEPT_TEMPORAL_HL",
    "RELATION_TEMPORAL_HL",
    "CONTRASTS_INHIBITION_FACTOR",
    "TASK_AFFINITY_BOOST",
    "PROPAGATION_FLOOR",
    "PROPAGATION_MIN_RATIO",
    "PROPAGATION_FLOOR_RANK_SPREAD",
]

# ── Inhibition coefficient for negative-axis links ────────────────────
# Negative is treated as an *inhibitory* relation: activating the
# source produces negative activation on the target (which subtracts
# from any positive spread on the same target).  The scalar below is
# multiplied by the same edge/depth/task/temporal factors used for
# excitation — repulsion strength tracks the evidence behind it.
CONTRASTS_INHIBITION_FACTOR: float = 0.6

# ── Task affinity boost ──────────────────────────────────────────────
# When the current task matches a concept's or relation's history,
# propagation is multiplied by up to this factor.  Partial word-level
# matches receive a proportional share of the boost.
TASK_AFFINITY_BOOST: float = 1.5

# ── Propagation floor ────────────────────────────────────────────────
# Low-confidence nodes still allow propagation to pass through at
# this minimum readiness level, preventing "dead node" blockage.
PROPAGATION_FLOOR: float = 0.3

# ── Propagation minimum ratio ────────────────────────────────────────
# Signals weaker than this fraction of the strongest *seed* score are
# lifted into a band just below it, preventing the multiplicative chain
# from zeroing out signal too early.  This widens the cognitive horizon
# from ~1 hop to 3-4 hops.
PROPAGATION_MIN_RATIO: float = 0.03

# Acceptance cut as a fraction of the strongest seed score.  Applied as
# ``min(min_activation, RELATIVE_MIN_ACTIVATION × seed_max)`` it only ever
# loosens the absolute cut, for seeds too weak to clear it otherwise.
RELATIVE_MIN_ACTIVATION: float = 0.02

# Width of the floor band, as a fraction of the floor.  A lifted signal
# lands in ``[(1 − SPREAD) · floor, floor)`` at a position proportional to
# its raw strength, so lifted concepts remain strictly ordered by the
# evidence behind them instead of collapsing into one tied score.
PROPAGATION_FLOOR_RANK_SPREAD: float = 0.1


def _accumulate(current: float, contribution: float, ceiling: float) -> float:
    """Bounded noisy-OR of two activation contributions.

    Both values are expressed as fractions of ``ceiling`` (the strongest
    seed score) so the result can approach, but never exceed, the
    ceiling: ``S · (1 − (1 − a/S)(1 − b/S))``.
    """
    if ceiling <= 0.0:
        return max(current, contribution)
    a = min(1.0, current / ceiling)
    b = min(1.0, contribution / ceiling)
    return ceiling * (a + b - a * b)


class ActivationEngine:
    """Spreads activation from seed concepts through the relation network.

    Implements the ``ActivationProvider`` Protocol from ``world0.core``.
    Depends only on ``ConceptStore`` / ``RelationStore`` Protocols.
    """

    def __init__(
        self,
        concepts: "ConceptStore",
        relations: "RelationStore",
        clock: CognitiveClock | None = None,
    ) -> None:
        self._concepts = concepts
        self._relations = relations
        self._clock = clock or CognitiveClock()

    def activate(
        self,
        seed_ids: list[str],
        *,
        max_depth: int = 2,
        decay: float = 0.6,
        min_activation: float = 0.01,
        source: str = "",
        task: str = "",
        record: bool = True,
        perspective: Perspective | None = None,
    ) -> dict[str, float]:
        """Spread activation from seeds with excitation + inhibition.

        Propagation strength (excitatory edges) =
            source_score
            * relation.weight * perspective.weight_for(relation.type)
            * perspective.weight_for_direction(forward | backward)
            * max(neighbor.confidence, PROPAGATION_FLOOR)
            * depth_decay
            * task_affinity
            * domain_affinity
            * relation.temporal_relevance
            * neighbor.temporal_relevance

        Contributions arriving at the same concept in the same layer are
        combined with a bounded noisy-OR (see module docstring).
        Negative-axis edges use the same multiplicative chain but feed an
        independent inhibition channel multiplied by
        CONTRASTS_INHIBITION_FACTOR.  The returned score for each
        concept is ``max(0, activation - inhibition)``.

        Args:
            record: If True, touched concepts get their activation_count
                incremented and last_activated updated (once per concept).
                Set to False for read-only operations like projection.
            perspective: Task/role view.  Relation type weights, active
                domains and the task label are all sourced from it.
                Plain ``task`` is honored for backward compatibility
                when no perspective is passed.

        Returns concept_id → *net* activation score mapping.
        """
        # Unify legacy (task: str) and new (Perspective) arguments.
        if perspective is None:
            perspective = Perspective(task=task)
        task_lower = (perspective.task or task).strip().lower()
        # One reference instant (both cognitive-time coordinates) for the
        # whole pass: freshness must not depend on iteration order.
        now_tick = self._clock.tick
        now = datetime.now(timezone.utc)

        activations: dict[str, float] = {}
        depth_of: dict[str, int] = {}
        inhibitions: dict[str, float] = {}

        # Seed concepts activate at their own confidence level
        seed_score_max = 0.0
        frontier: list[str] = []
        for cid in dict.fromkeys(seed_ids):
            node = self._concepts.get(cid)
            if not node:
                continue
            score = node.confidence
            # Boost seeds that have task affinity
            if task_lower:
                score = min(
                    1.0, score * self._task_boost(node.task_affinity(task_lower))
                )
            # Domain affinity stacks on top — a concept whose dominant
            # domain is "in focus" for the perspective is boosted too.
            if self._concept_in_perspective_domain(node, perspective):
                score = min(
                    1.0, score * perspective.domain_affinity_boost
                )
            activations[cid] = score
            depth_of[cid] = 0
            frontier.append(cid)
            if score > seed_score_max:
                seed_score_max = score
            if record:
                node.activate(source=source, task=task, tick=now_tick)

        # Propagation floor: minimum signal that can still pass through
        prop_floor = seed_score_max * PROPAGATION_MIN_RATIO
        # Acceptance cut: the absolute ``min_activation`` loosened to a
        # fraction of the strongest seed, so weak (e.g. embryonic) seeds keep
        # the same 3–4 hop horizon as confident ones.
        cut = min(min_activation, RELATIVE_MIN_ACTIVATION * seed_score_max)

        # Layered BFS propagation with decay
        for depth in range(max_depth):
            depth_factor = decay ** (depth + 1)
            layer: dict[str, float] = {}

            for cid in frontier:
                source_score = activations.get(cid, 0.0)
                if source_score < cut:
                    continue

                for rel in self._relations.for_concept(cid):
                    neighbor_id = rel.other_end(cid)
                    if neighbor_id is None:
                        continue

                    is_negative = rel.relation_type == RelationType.NEGATIVE
                    # Excitation only flows outward: a concept settled at
                    # this or a shallower layer is frozen, so cycles never
                    # feed activation back into their own source.
                    if (
                        not is_negative
                        and neighbor_id in depth_of
                        and depth_of[neighbor_id] <= depth
                    ):
                        continue

                    neighbor = self._concepts.get(neighbor_id)
                    if neighbor is None:
                        continue

                    default_type_factor = RELATION_TYPE_FACTOR.get(
                        rel.relation_type, 0.5
                    )
                    type_factor = perspective.weight_for(
                        rel.relation_type.value, default_type_factor
                    )
                    direction = "forward" if rel.source_id == cid else "backward"
                    edge_strength = (
                        rel.weight
                        * type_factor
                        * perspective.weight_for_direction(direction)
                    )

                    neighbor_readiness = max(
                        neighbor.confidence, PROPAGATION_FLOOR
                    )

                    task_boost = 1.0
                    if task_lower:
                        affinity = max(
                            rel.task_affinity(task_lower),
                            neighbor.task_affinity(task_lower),
                        )
                        task_boost = self._task_boost(affinity)

                    # Domain affinity boost for perspective-focused domains
                    domain_boost = 1.0
                    if self._concept_in_perspective_domain(
                        neighbor, perspective
                    ):
                        domain_boost = perspective.domain_affinity_boost

                    rel_freshness = rel.temporal_relevance(
                        RELATION_TEMPORAL_HL, now_tick=now_tick, now=now
                    )
                    neighbor_freshness = neighbor.temporal_relevance(
                        CONCEPT_TEMPORAL_HL, now_tick=now_tick, now=now
                    )

                    raw = (
                        source_score
                        * edge_strength
                        * neighbor_readiness
                        * depth_factor
                        * task_boost
                        * domain_boost
                        * rel_freshness
                        * neighbor_freshness
                    )

                    if is_negative:
                        # Inhibitory channel: negative-axis links spread negative
                        # activation instead of weak excitation.
                        inhibition = raw * CONTRASTS_INHIBITION_FACTOR
                        if inhibition < min_activation:
                            continue
                        inhibitions[neighbor_id] = _accumulate(
                            inhibitions.get(neighbor_id, 0.0),
                            inhibition,
                            seed_score_max,
                        )
                        continue

                    # Rank-preserving floor — ensures distant but
                    # structurally connected concepts still receive enough
                    # signal to participate in projections without
                    # collapsing into a single tied score.
                    propagated = self._apply_floor(raw, prop_floor)
                    if propagated < cut:
                        continue

                    layer[neighbor_id] = _accumulate(
                        layer.get(neighbor_id, 0.0), propagated, seed_score_max
                    )

            if not layer:
                break

            for neighbor_id, score in layer.items():
                activations[neighbor_id] = score
                depth_of[neighbor_id] = depth + 1
                if record:
                    neighbor = self._concepts.get(neighbor_id)
                    if neighbor is not None:
                        neighbor.activate(
                            source=source, task=task, tick=now_tick
                        )

            frontier = list(layer.keys())

        # Subtract inhibition from excitation; drop concepts driven to
        # zero or below so they vanish from the projection entirely.
        # Iterate ``activations`` in insertion order (layer order) so
        # projection selection is deterministic across process runs.
        net: dict[str, float] = {}
        for cid, excitation in activations.items():
            score = excitation - inhibitions.get(cid, 0.0)
            if score > cut:
                net[cid] = score
        return net

    @staticmethod
    def _apply_floor(raw: float, prop_floor: float) -> float:
        """Lift a weak signal into the rank-preserving floor band."""
        if raw <= 0.0 or prop_floor <= 0.0 or raw >= prop_floor:
            return raw
        position = raw / prop_floor  # in (0, 1)
        return prop_floor * (
            (1.0 - PROPAGATION_FLOOR_RANK_SPREAD)
            + PROPAGATION_FLOOR_RANK_SPREAD * position
        )

    @staticmethod
    def _task_boost(affinity: float) -> float:
        """Propagation multiplier for a graded task affinity in [0, 1]."""
        if affinity <= 0.0:
            return 1.0
        return 1.0 + (TASK_AFFINITY_BOOST - 1.0) * min(1.0, affinity)

    @staticmethod
    def _concept_in_perspective_domain(
        node, perspective: Perspective
    ) -> bool:
        if not perspective.active_domains:
            return False
        # Prefer the dominant domain in the concept's profile, fall
        # back to its static ``domain`` field.
        if node.domain_profile:
            top_domain, _ = max(
                node.domain_profile.items(), key=lambda item: item[1]
            )
            if perspective.domain_match(top_domain):
                return True
        return perspective.domain_match(node.domain)

    @staticmethod
    def _concept_has_task(node, task_lower: str) -> bool:
        """Check if a concept has been activated under a matching task."""
        return node.task_affinity(task_lower) > 0.0
