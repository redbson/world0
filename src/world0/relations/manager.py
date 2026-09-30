"""Relation lifecycle management — discover, reinforce, decay, refine."""

from __future__ import annotations

from collections.abc import Callable

from world0.dynamics.decay import relation_dead, settle_relation
from world0.schemas.clock import CognitiveClock, wall_now
from world0.schemas.relation import (
    RelationEdge,
    RelationType,
    semantic_relation_spec,
)
from world0.store.base import Store


class RelationManager:
    """Manages the lifecycle of relations in the cognitive world.

    Relations are discovered through the Agent's work. They start weak
    and strengthen through repeated observation.
    """

    def __init__(self, store: Store, clock: CognitiveClock | None = None) -> None:
        self._store = store
        # Cognitive clock shared with the owning World; stamps discovery
        # and reinforcement with the current observation tick.
        self._clock = clock or CognitiveClock()
        self._relations: dict[str, RelationEdge] = {}
        self._by_concept: dict[str, list[str]] = {}  # concept_id → [relation_ids]
        self._dirty: set[str] = set()  # relation ids with unsaved changes
        # Connection-event hook wired by the owning ``World``.
        self._on_connection: Callable[..., None] | None = None

    def connect_lifecycle(
        self, *, on_connection: Callable[..., None] | None = None
    ) -> None:
        """Wire the event-time lifecycle.

        ``on_connection(source_id, target_id)`` runs after a relation is
        created or reinforced, so a concept's promotion to CORE is judged
        at the moment its connections change instead of whenever a reflect
        happens to count them (``LifecycleEngine.on_connection``).
        """
        self._on_connection = on_connection

    def load(self) -> None:
        """Load all relations from persistent store."""
        self._relations.clear()
        self._by_concept.clear()
        for edge in self._store.load_all_relations():
            edge.ensure_probability()
            # Stores written before belief was separated from the inhibition
            # gain: rebase legacy explicit negative claims once.
            if edge.adopt_claim_prior():
                self._dirty.add(edge.id)
            self._relations[edge.id] = edge
            self._index(edge)

    def save_all(self) -> None:
        """Persist all relations to store (batch)."""
        self._store.save_relations_batch(list(self._relations.values()))
        self._dirty.clear()

    def flush(self) -> None:
        """Persist only dirty (modified) relations to store."""
        if not self._dirty:
            return
        dirty_rels = [
            self._relations[rid] for rid in self._dirty if rid in self._relations
        ]
        self._store.save_relations_batch(dirty_rels)
        self._dirty.clear()

    def mark_dirty(self, relation_id: str) -> None:
        """Mark a relation as having unsaved changes."""
        self._dirty.add(relation_id)

    # ── lazy reaping ──────────────────────────────────────────────────

    def reap_dead_between(self, id_a: str, id_b: str) -> int:
        """Remove relations between two concepts that are already dead.

        A relation whose settled weight is below the prune line is gone,
        whether or not a reflect has physically deleted it: a restatement
        or a co-occurrence must meet the same state either way (a fresh
        edge), not revive the old one just because no reflect ran
        (docs/paper, Theorem 3.7).  Deadness is judged at the last tick a
        reflect could have run, the tick before the current observation.
        Returns the number removed.
        """
        tick = self._clock.tick - 1
        now = wall_now()
        dead = [e for e in self.find_any_between(id_a, id_b) if relation_dead(e, tick, now)]
        for edge in dead:
            self.remove(edge.id)
        return len(dead)

    # ── discovery ─────────────────────────────────────────────────────

    def discover(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType = RelationType.PARALLEL,
        *,
        semantic_relation: str = "",
        provenance: str = "",
        is_explicit: bool = True,
        probability: float | None = None,
        prior_probability: float | None = None,
        prior_strength: float = 1.0,
        evidence_strength: float = 2.0,
    ) -> tuple[RelationEdge, bool]:
        """Find existing relation or discover a new one.

        Args:
            is_explicit: True for Agent-declared relations, False for
                Hebbian (auto-discovered) relations. Explicit relations
                can reach weight 1.0; Hebbian relations cap at 0.7.

        Returns (relation, is_new).
        """
        if source_id == target_id:
            raise ValueError("Cannot create a self-relation")

        semantic_spec = semantic_relation_spec(semantic_relation or relation_type.value)
        relation_type = semantic_spec.axis

        # Directed relations are matched in their stated orientation:
        # "B depends on A" is a different claim from "A depends on B" and
        # must not confirm it (docs §7.18).
        self.reap_dead_between(source_id, target_id)
        existing = self.find_between(
            source_id, target_id, relation_type, directed=True,
            semantic=semantic_spec.name if is_explicit and semantic_relation else None,
        )
        if existing and existing.is_retracted and not is_explicit:
            # Co-occurrence cannot restore a withdrawn claim; a withdrawn
            # co-occurrence edge is replaced by a fresh one.
            self.remove(existing.id)
            existing = None
        promoted = False
        if existing:
            if is_explicit:
                existing.record_claim(provenance)
                self._dirty.add(existing.id)
                if not existing.is_explicit:
                    # A co-occurrence edge the Agent now states as a typed
                    # claim becomes that claim (it was left a co-occurrence
                    # edge with a new label, so the claim never showed).
                    settle_relation(existing, self._clock.tick)
                    existing.is_explicit = True
                    existing.belief_prior = semantic_spec.claim_prior
                    existing.probability = max(existing.probability, semantic_spec.claim_prior)
                    existing.weight = max(existing.weight, semantic_spec.propagation_strength)
                    existing.confidence = max(existing.confidence, semantic_spec.structural_strength)
                    existing.semantic_relation = semantic_spec.name
                    existing.structural_strength = semantic_spec.structural_strength
                    existing.propagation_strength = semantic_spec.propagation_strength
                    promoted = True
            if existing.is_retracted and is_explicit:
                # Restated: the claim holds again.  Its weight relaxed toward
                # zero while withdrawn; settle that first.
                settle_relation(existing, self._clock.tick)
                existing.retracted_tick = None
                self._dirty.add(existing.id)
            if probability is not None or prior_probability is not None:
                # Owed decay first, then the restatement (Theorem 3.2).
                settle_relation(existing, self._clock.tick)
                existing.update_probability(
                    evidence_probability=probability,
                    prior_probability=prior_probability,
                    prior_strength=prior_strength,
                    evidence_strength=evidence_strength,
                    provenance=provenance,
                    tick=self._clock.tick,
                )
                self._dirty.add(existing.id)
                if self._on_connection is not None:
                    self._on_connection(existing.source_id, existing.target_id)
            # A promoted edge is a new claim: the caller treats it as stated
            # once, not as a restatement.
            return existing, promoted

        # Explicit relations start stronger than Hebbian (auto-discovered)
        init_weight = semantic_spec.propagation_strength if is_explicit else 0.15
        init_confidence = semantic_spec.structural_strength if is_explicit else 0.15
        # Belief that the statement is correct is not the gain of the channel
        # it travels on: ``init_weight`` (operational strength; the
        # inhibition gain on the negative axis) is unchanged, the default
        # belief is the label's claim prior.
        init_probability = self._initial_probability(
            default=semantic_spec.claim_prior if is_explicit else init_weight,
            probability=probability,
            prior_probability=prior_probability,
            prior_strength=prior_strength,
            evidence_strength=evidence_strength,
        )
        has_probability_input = probability is not None or prior_probability is not None

        edge = RelationEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            semantic_relation=semantic_spec.name,
            structural_strength=semantic_spec.structural_strength,
            propagation_strength=semantic_spec.propagation_strength,
            probability=init_probability,
            belief_prior=init_probability if is_explicit else None,
            probability_observation_count=1 if probability is not None else 0,
            # The inhibition gain of a negative claim is a property of its
            # label, not of how sure the extractor was (belief lives in
            # ``probability``); other axes seed the weight from the belief.
            weight=(
                init_probability
                if has_probability_input and relation_type != RelationType.NEGATIVE
                else init_weight
            ),
            confidence=init_probability if has_probability_input else init_confidence,
            provenance=provenance,
            task_history=[provenance] if provenance else [],
            is_explicit=is_explicit,
            reinforcement_count=1 if probability is not None and probability >= 0.5 else 0,
            disconfirmation_count=1 if probability is not None and probability < 0.5 else 0,
            discovered_tick=self._clock.tick,
            last_reinforced_tick=self._clock.tick,
        )
        if is_explicit:
            edge.record_claim(provenance)
        self._relations[edge.id] = edge
        self._index(edge)
        self._dirty.add(edge.id)
        if self._on_connection is not None:
            self._on_connection(source_id, target_id)
        return edge, True

    @staticmethod
    def _initial_probability(
        *,
        default: float,
        probability: float | None,
        prior_probability: float | None,
        prior_strength: float,
        evidence_strength: float,
    ) -> float:
        total = 0.0
        strength = 0.0
        if prior_probability is not None and prior_strength > 0:
            total += min(1.0, max(0.0, prior_probability)) * prior_strength
            strength += prior_strength
        if probability is not None and evidence_strength > 0:
            total += min(1.0, max(0.0, probability)) * evidence_strength
            strength += evidence_strength
        if strength <= 0:
            return default
        return min(1.0, max(0.0, total / strength))

    # ── lookup ────────────────────────────────────────────────────────

    def get(self, relation_id: str) -> RelationEdge | None:
        return self._relations.get(relation_id)

    def all(self) -> list[RelationEdge]:
        return list(self._relations.values())

    def for_concept(self, concept_id: str) -> list[RelationEdge]:
        rids = self._by_concept.get(concept_id, [])
        return [self._relations[rid] for rid in rids if rid in self._relations]

    def neighbors(self, concept_id: str) -> list[str]:
        result: list[str] = []
        for rel in self.for_concept(concept_id):
            other = rel.other_end(concept_id)
            if other:
                result.append(other)
        return result

    def find_between(
        self,
        id_a: str,
        id_b: str,
        relation_type: RelationType | None = None,
        *,
        directed: bool = False,
        semantic: str | None = None,
    ) -> RelationEdge | None:
        """Find a specific relation between two concepts.

        ``directed=True`` matches a directed edge only in the
        ``id_a → id_b`` orientation (see ``RelationEdge.connects``).
        With ``semantic`` the claim is identified by its label too: "A
        contains B" and "A depends on B" are different claims, so only an
        edge with that label matches, or else a co-occurrence edge on the
        axis (which asserts no label and becomes the claim when stated).
        """
        fallback = None
        for rel in self.for_concept(id_a):
            if not rel.connects(id_a, id_b, directed=directed):
                continue
            if relation_type is not None and rel.relation_type != relation_type:
                continue
            if semantic is None or rel.semantic_relation == semantic:
                return rel
            if not rel.is_explicit and fallback is None:
                fallback = rel
        return fallback

    def find_any_between(self, id_a: str, id_b: str) -> list[RelationEdge]:
        return [r for r in self.for_concept(id_a) if r.involves(id_b)]

    # ── reinforcement ─────────────────────────────────────────────────

    def reinforce(self, relation_id: str, provenance: str = "") -> RelationEdge | None:
        edge = self._relations.get(relation_id)
        if not edge:
            return None
        # Settle the decay owed since the last reinforcement first
        # (docs/paper, Theorem 3.2).
        settle_relation(edge, self._clock.tick)
        edge.reinforce(provenance=provenance, tick=self._clock.tick)
        self._dirty.add(edge.id)
        if self._on_connection is not None:
            self._on_connection(edge.source_id, edge.target_id)
        return edge

    def weaken(self, relation_id: str, provenance: str = "") -> RelationEdge | None:
        edge = self._relations.get(relation_id)
        if not edge:
            return None
        # Settle first: the penalty must not be decayed retroactively over
        # an interval that started before it (Theorem 3.2).
        settle_relation(edge, self._clock.tick)
        edge.weaken(provenance=provenance)
        self._dirty.add(edge.id)
        return edge

    def retract(self, relation_id: str) -> RelationEdge | None:
        """Withdraw a claim: it no longer holds (``RelationEdge.retracted_tick``).

        Unlike ``weaken`` this is not evidence the claim was wrong, so its
        belief is untouched; the edge simply leaves the live world.
        """
        edge = self._relations.get(relation_id)
        if not edge:
            return None
        settle_relation(edge, self._clock.tick)
        edge.retracted_tick = self._clock.tick
        self._dirty.add(edge.id)
        return edge

    def refine_type(self, relation_id: str, new_type: RelationType) -> None:
        """Agent refines a RELATED_TO relation to a more specific type."""
        edge = self._relations.get(relation_id)
        if edge:
            edge.relation_type = new_type
            self._dirty.add(edge.id)

    def adjust_strength(
        self,
        relation_id: str,
        *,
        weight_delta: float = 0.0,
        confidence_delta: float = 0.0,
    ) -> RelationEdge | None:
        """Apply bounded strength adjustments to a relation."""
        edge = self._relations.get(relation_id)
        if not edge:
            return None
        settle_relation(edge, self._clock.tick)
        edge.weight = min(1.0, max(0.01, edge.weight + weight_delta))
        edge.confidence = min(1.0, max(0.01, edge.confidence + confidence_delta))
        # Feedback about the relation's usefulness moves the semantic belief
        # by the same delta; it is never overwritten with the (differently
        # scaled) confidence value.
        edge.probability = min(1.0, max(0.01, edge.probability + confidence_delta))
        self._dirty.add(edge.id)
        if (weight_delta > 0 or confidence_delta > 0) and self._on_connection is not None:
            self._on_connection(edge.source_id, edge.target_id)
        return edge

    # ── removal ───────────────────────────────────────────────────────

    def remove(self, relation_id: str) -> bool:
        edge = self._relations.pop(relation_id, None)
        if edge:
            self._unindex(edge)
            self._store.delete_relation(relation_id)
            return True
        return False

    def remove_for_concept(self, concept_id: str) -> int:
        rels = self.for_concept(concept_id)
        for r in rels:
            self._relations.pop(r.id, None)
            self._unindex(r)
            self._store.delete_relation(r.id)
        return len(rels)

    def migrate_concept(self, old_id: str, new_id: str) -> int:
        """Rewrite all relations touching ``old_id`` to use ``new_id``.

        If the rewrite produces a duplicate relation (same direction
        and type between the same pair), the older edge absorbs the
        duplicate: weights sum (capped at 1.0), counters add, the
        latest ``last_reinforced`` timestamp wins.  Self-loops that
        result from the migration are dropped.

        Returns the number of edges that were migrated or absorbed.
        """
        if old_id == new_id:
            return 0

        affected = 0
        # Copy list — we mutate during iteration
        for rel in self.for_concept(old_id):
            old_src, old_tgt = rel.source_id, rel.target_id
            new_src = new_id if old_src == old_id else old_src
            new_tgt = new_id if old_tgt == old_id else old_tgt

            # Drop self-loops created by the migration
            if new_src == new_tgt:
                self._relations.pop(rel.id, None)
                self._unindex(rel)
                self._store.delete_relation(rel.id)
                affected += 1
                continue

            duplicate = self._find_directed(
                new_src, new_tgt, rel.relation_type, rel.semantic_relation
            )
            if duplicate is not None and duplicate.id != rel.id:
                # Absorb into the existing edge
                duplicate.weight = min(1.0, duplicate.weight + rel.weight)
                duplicate.confidence = min(
                    1.0, duplicate.confidence + rel.confidence
                )
                duplicate.probability = min(
                    1.0, max(duplicate.probability, rel.probability)
                )
                duplicate.probability_observation_count += (
                    rel.probability_observation_count
                )
                duplicate.reinforcement_count += rel.reinforcement_count
                duplicate.disconfirmation_count += rel.disconfirmation_count
                if rel.last_reinforced > duplicate.last_reinforced:
                    duplicate.last_reinforced = rel.last_reinforced
                duplicate.last_reinforced_tick = max(
                    duplicate.last_reinforced_tick, rel.last_reinforced_tick
                )
                duplicate.discovered_tick = min(
                    duplicate.discovered_tick, rel.discovered_tick
                )
                for t in rel.task_history:
                    if t not in duplicate.task_history:
                        duplicate.task_history.append(t)
                for t in rel.claim_tasks:
                    duplicate.record_claim(t)
                # The merged claim is withdrawn only if both were: a live
                # statement under either name keeps it in the world.
                if duplicate.is_retracted and rel.is_retracted:
                    duplicate.retracted_tick = max(duplicate.retracted_tick, rel.retracted_tick)
                else:
                    duplicate.retracted_tick = None
                self._dirty.add(duplicate.id)

                self._relations.pop(rel.id, None)
                self._unindex(rel)
                self._store.delete_relation(rel.id)
            else:
                self._unindex(rel)
                rel.source_id = new_src
                rel.target_id = new_tgt
                self._index(rel)
                self._dirty.add(rel.id)
            affected += 1

        return affected

    def _find_directed(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        semantic: str = "",
    ) -> RelationEdge | None:
        """The edge a migrated edge merges into: same direction, axis and label."""
        for rel in self.for_concept(source_id):
            if (
                rel.source_id == source_id
                and rel.target_id == target_id
                and rel.relation_type == relation_type
                and rel.semantic_relation == semantic
            ):
                return rel
        return None

    # ── internals ─────────────────────────────────────────────────────

    def _index(self, edge: RelationEdge) -> None:
        self._by_concept.setdefault(edge.source_id, []).append(edge.id)
        self._by_concept.setdefault(edge.target_id, []).append(edge.id)

    def _unindex(self, edge: RelationEdge) -> None:
        for cid in (edge.source_id, edge.target_id):
            ids = self._by_concept.get(cid, [])
            if edge.id in ids:
                ids.remove(edge.id)

    def __len__(self) -> int:
        return len(self._relations)
