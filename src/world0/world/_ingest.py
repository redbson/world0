"""Ingest pipeline — observation → concept/relation updates.

The pipeline is a pure orchestrator: it owns no state of its own and
holds only Protocol references to the subsystems it drives.  Each step
is a small private method so individual passes can be unit-tested in
isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from world0.dynamics.decay import concept_expired
from world0.schemas.clock import CognitiveClock, wall_now
from world0.schemas.relation import (
    RelationType,
    normalize_semantic_relation,
    orient_relation,
    semantic_relation_spec,
)
from world0.schemas.types import ConceptCandidate, IngestResult, Observation, PredictionError

if TYPE_CHECKING:
    from world0.core import (
        ColorField,
        ConceptStore,
        HebbianLearner,
        RelationStore,
    )


class IngestPipeline:
    """Six-step ingest: concepts → relations → hebbian → descriptions →
    disconfirmation → color seeding.

    The pipeline never touches persistence directly — the caller (the
    ``World`` facade) is responsible for flushing dirty state once the
    pipeline returns.  This keeps the pipeline trivially testable
    against in-memory ConceptStore / RelationStore mocks.
    """

    def __init__(
        self,
        *,
        concepts: ConceptStore,
        relations: RelationStore,
        hebbian: HebbianLearner,
        color: ColorField,
        clock: CognitiveClock | None = None,
    ) -> None:
        self._concepts = concepts
        self._relations = relations
        self._hebbian = hebbian
        self._color = color
        # Optional: without a clock (mock stores) expiry is not judged.
        self._clock = clock

    def _reap_if_expired(self, node):
        """Delete ``node`` (with its relations) if it is already dead.

        A concept past recovery — FADING, below the prune line and idle for
        ``PRUNE_MIN_IDLE_TICKS`` — is gone whether or not a reflect has
        physically deleted it; a mention must meet the same state either
        way (a fresh node), or existence would depend on how often reflect
        ran (docs/paper, Theorem 3.7).  Judged at the last tick a reflect
        could have run, the one before this observation.  Returns True
        when the node was removed.
        """
        if self._clock is None or node is None:
            return False
        if not concept_expired(node, self._clock.tick - 1, wall_now()):
            return False
        self._relations.remove_for_concept(node.id)
        self._concepts.remove(node.id)
        return True

    def _live_resolve(self, ref: str):
        """``resolve`` that treats an already-expired concept as absent."""
        node = self._concepts.resolve(ref)
        if node is not None and self._reap_if_expired(node):
            return None
        return node

    def run(self, observation: Observation) -> IngestResult:
        result = IngestResult()
        resolved_ids: list[str] = []
        local_refs: dict[str, str] = {}

        self._step_concepts(observation, result, resolved_ids, local_refs)
        self._step_relations(observation, result, local_refs)
        # Prediction before learning: the observation is scored against
        # what co-occurrence predicted, then learned (docs/mc/04).
        result.prediction = self._prediction_error(resolved_ids)
        self._step_hebbian(observation, resolved_ids, result)
        self._step_descriptions(observation, local_refs)
        self._step_disconfirmation(observation, result, local_refs)
        self._step_retractions(observation, result, local_refs)
        self._step_color(observation, resolved_ids)

        return result

    # ── individual passes ────────────────────────────────────────────

    def _step_concepts(
        self,
        observation: Observation,
        result: IngestResult,
        resolved_ids: list[str],
        local_refs: dict[str, str],
    ) -> None:
        candidates = observation.concept_candidates or [
            ConceptCandidate(
                uid=name,
                name=name,
                description=observation.descriptions.get(name, ""),
                domain=observation.domain,
            )
            for name in observation.concepts
        ]
        for candidate in candidates:
            name = candidate.name
            create_args = dict(
                origin=observation.source,
                task=observation.task,
                description=candidate.description
                or observation.descriptions.get(name, ""),
                kind=candidate.kind,
                sense=candidate.sense,
                domain=candidate.domain or observation.domain,
                aliases=candidate.aliases,
            )
            node, is_new = self._concepts.get_or_create(name, **create_args)
            if not is_new and self._reap_if_expired(node):
                node, is_new = self._concepts.get_or_create(name, **create_args)
            for alias in candidate.aliases:
                self._concepts.add_alias(node.id, alias)
            # Always reinforce — creation is also an activation event.
            self._concepts.reinforce(
                node.id, source=observation.source, task=observation.task
            )
            if candidate.uid:
                local_refs[candidate.uid] = node.id
            local_refs.setdefault(name, node.id)
            source_ref_id = observation.source_id or observation.source
            concept_meta = observation.extraction_metadata.get("concepts", {})
            meta = concept_meta.get(name, {}) if isinstance(concept_meta, dict) else {}
            evidence = candidate.evidence
            if not evidence and isinstance(meta, dict):
                evidence = meta.get("evidence", "")
            if source_ref_id:
                node.record_source_ref(
                    source_id=source_ref_id,
                    source=observation.source,
                    task=observation.task,
                    excerpt=str(evidence or ""),
                )
                self._concepts.mark_dirty(node.id)
            self._record_token_refs(
                node=node,
                candidate=candidate,
                source_id=source_ref_id,
                source=observation.source,
                task=observation.task,
                excerpt=str(evidence or ""),
            )
            (result.new_concepts if is_new else result.reinforced_concepts).append(
                node.name
            )
            resolved_ids.append(node.id)

    def _record_token_refs(
        self,
        *,
        node,
        candidate: ConceptCandidate,
        source_id: str,
        source: str,
        task: str,
        excerpt: str,
    ) -> None:
        role = (
            "canonical"
            if candidate.name.strip().lower() == node.name.strip().lower()
            else "synonym"
        )
        node.record_token_ref(
            token=candidate.name,
            source_id=source_id,
            source=source,
            task=task,
            excerpt=excerpt,
            role=role,
        )
        for alias in candidate.aliases:
            node.record_token_ref(
                token=alias,
                source_id=source_id,
                source=source,
                task=task,
                excerpt=excerpt,
                role="alias",
            )
        self._concepts.mark_dirty(node.id)

    def _step_relations(
        self,
        observation: Observation,
        result: IngestResult,
        local_refs: dict[str, str],
    ) -> None:
        relation_meta = self._relation_metadata_by_key(observation)
        relation_priors = self._relation_priors_by_key(observation)
        for src_name, tgt_name, relation_name in observation.relations:
            src = self._resolve_observation_ref(src_name, local_refs)
            tgt = self._resolve_observation_ref(tgt_name, local_refs)
            if not src or not tgt:
                continue
            # Skip self-loops.  Endpoints can collapse to one concept either
            # because the model emitted the same name twice or because two
            # different surface forms resolved to the same node — both would
            # make RelationManager.discover() raise on a self-relation.
            if src.id == tgt.id:
                continue

            src, tgt, semantic_relation = orient_relation(src, tgt, relation_name)
            rel_type = semantic_relation_spec(semantic_relation).axis
            key = (src_name, tgt_name, semantic_relation)
            meta = relation_meta.get(key, {})
            prior = relation_priors.get(key)

            edge, is_new = self._relations.discover(
                src.id,
                tgt.id,
                rel_type,
                semantic_relation=semantic_relation,
                provenance=observation.task,
                prior_probability=prior.probability if prior else None,
                prior_strength=prior.strength if prior else 1.0,
            )
            label = f"{src.name} → {semantic_relation} → {tgt.name}"
            if is_new:
                result.new_relations.append(label)
            else:
                if prior is None:
                    # Operational reinforcement plus semantic confirmation:
                    # an Agent re-stating a typed relation is evidence that
                    # the relation is correct, not just that its endpoints
                    # co-occur (Hebbian learning only does the former).
                    self._relations.reinforce(edge.id, provenance=observation.task)
                    edge.confirm()
                    self._relations.mark_dirty(edge.id)
                result.reinforced_relations.append(label)
            self._weaken_opposing(
                edge, src, tgt, rel_type, semantic_relation, observation, result
            )

    def _stated_claim(self, src_id: str, tgt_id: str, rel_type, semantic_relation: str):
        """The claim a withdrawal or contradiction names.

        The claim with that label; failing that, the only live claim on that
        ordered pair and axis (an extractor may paraphrase the label it
        withdraws: "X no longer depends on Y" for a stored "enables").  A
        co-occurrence edge is never a claim to withdraw or contradict.
        """
        exact = self._relations.find_between(
            src_id, tgt_id, rel_type, directed=True,
            semantic=semantic_relation, cooccurrence_fallback=False,
        )
        if exact is not None:
            return exact
        live = [
            r for r in self._relations.find_any_between(src_id, tgt_id)
            if r.is_explicit and not r.is_retracted and r.relation_type == rel_type
            and r.connects(src_id, tgt_id, directed=True)
        ]
        return live[0] if len(live) == 1 else None

    def _weaken_opposing(
        self, edge, src, tgt, rel_type, semantic_relation, observation, result
    ) -> None:
        """An explicit claim is evidence against the opposite claim.

        Stating "A conflicts with B" disconfirms an explicit "A enables B"
        (and vice versa) the same way ``contradicted_relations`` would,
        so contradictory beliefs about a pair compete instead of both
        staying confident (docs §7.18).  Co-occurrence edges and
        ``generic_relation`` claims are never weakened this way.
        """
        for other in self._relations.find_any_between(src.id, tgt.id):
            if other.id == edge.id or not other.is_explicit or other.is_retracted:
                continue
            if not other.opposes(rel_type, semantic_relation):
                continue
            self._relations.weaken(other.id, provenance=observation.task)
            other_src = src if other.source_id == src.id else tgt
            other_tgt = tgt if other_src is src else src
            result.weakened_relations.append(
                f"{other_src.name} → {other.semantic_relation} → {other_tgt.name}"
            )

    @staticmethod
    def _relation_metadata_by_key(
        observation: Observation,
    ) -> dict[tuple[str, str, str], dict]:
        raw = observation.extraction_metadata.get("relations", {})
        if not isinstance(raw, list):
            return {}
        result: dict[tuple[str, str, str], dict] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            src = str(item.get("source", "")).strip()
            tgt = str(item.get("target", "")).strip()
            rel_type = normalize_semantic_relation(item.get("type", "generic_relation"))
            if src and tgt:
                result[(src, tgt, rel_type)] = item
        return result

    @staticmethod
    def _relation_priors_by_key(observation: Observation):
        result = {}
        for prior in observation.relation_priors:
            rel_type = normalize_semantic_relation(prior.relation_type)
            result[(prior.source, prior.target, rel_type)] = prior
        return result

    def _prediction_error(self, resolved_ids: list[str]) -> PredictionError:
        """Prediction error of this observation, with concept names."""
        raw = self._hebbian.prediction_error(resolved_ids)

        def name(cid: str) -> str:
            node = self._concepts.get(cid)
            return node.name if node else cid

        return raw.model_copy(update={
            "missing": [(name(a), name(b), p) for a, b, p in raw.missing],
            "novel_pairs": [(name(a), name(b)) for a, b in raw.novel_pairs],
        })

    def _step_hebbian(
        self,
        observation: Observation,
        resolved_ids: list[str],
        result: IngestResult,
    ) -> None:
        if len(resolved_ids) <= 1:
            return
        new_hebbian = self._hebbian.learn(
            resolved_ids, provenance=observation.task
        )
        for rid in new_hebbian:
            edge = self._relations.get(rid)
            if not edge:
                continue
            src = self._concepts.get(edge.source_id)
            tgt = self._concepts.get(edge.target_id)
            if src and tgt:
                result.hebbian_relations.append(f"{src.name} ↔ {tgt.name}")

    def _step_descriptions(
        self, observation: Observation, local_refs: dict[str, str]
    ) -> None:
        for candidate in observation.concept_candidates:
            if not candidate.description:
                continue
            node = (
                self._concepts.get(local_refs[candidate.uid])
                if candidate.uid and candidate.uid in local_refs
                else None
            )
            if node is None and candidate.name in local_refs:
                node = self._concepts.get(local_refs[candidate.name])
            if node:
                self._concepts.update_description(node.id, candidate.description)
        for name, desc in observation.descriptions.items():
            node = self._live_resolve(name)
            if node:
                self._concepts.update_description(node.id, desc)

    def _step_disconfirmation(
        self,
        observation: Observation,
        result: IngestResult,
        local_refs: dict[str, str],
    ) -> None:
        for name in observation.weakened:
            node = self._resolve_observation_ref(name, local_refs)
            if node:
                self._concepts.weaken(
                    node.id,
                    source=observation.source,
                    task=observation.task,
                )
                result.weakened_concepts.append(node.name)

        for src_name, tgt_name, relation_name in observation.contradicted_relations:
            src = self._resolve_observation_ref(src_name, local_refs)
            tgt = self._resolve_observation_ref(tgt_name, local_refs)
            if not src or not tgt:
                continue
            src, tgt, semantic_relation = orient_relation(src, tgt, relation_name)
            rel_type = semantic_relation_spec(semantic_relation).axis
            existing = self._stated_claim(src.id, tgt.id, rel_type, semantic_relation)
            if existing is None:
                # Contradiction without an existing edge weakens both
                # endpoint concepts instead — there is nothing else to
                # attach disconfirmation to.  Report the applied
                # disconfirmation so callers can observe it, mirroring the
                # ``observation.weakened`` path above.
                self._concepts.weaken(
                    src.id, source=observation.source, task=observation.task
                )
                self._concepts.weaken(
                    tgt.id, source=observation.source, task=observation.task
                )
                for node in (src, tgt):
                    if node.name not in result.weakened_concepts:
                        result.weakened_concepts.append(node.name)
                continue
            self._relations.weaken(existing.id, provenance=observation.task)
            result.weakened_relations.append(
                f"{src.name} → {semantic_relation} → {tgt.name}"
            )

    def _step_retractions(
        self,
        observation: Observation,
        result: IngestResult,
        local_refs: dict[str, str],
    ) -> None:
        """Withdraw claims that no longer hold (after this observation's own
        statements, so "A now depends on C; A no longer depends on B" in one
        observation leaves A→C live and A→B withdrawn)."""
        for src_name, tgt_name, relation_name in observation.retracted_relations:
            src = self._resolve_observation_ref(src_name, local_refs)
            tgt = self._resolve_observation_ref(tgt_name, local_refs)
            if not src or not tgt or src.id == tgt.id:
                continue
            src, tgt, semantic_relation = orient_relation(src, tgt, relation_name)
            rel_type = semantic_relation_spec(semantic_relation).axis
            # A claim that is already dead is gone, whether or not a reflect
            # removed it: withdrawing it changes nothing (Theorem 3.7).
            self._relations.reap_dead_between(src.id, tgt.id)
            existing = self._stated_claim(src.id, tgt.id, rel_type, semantic_relation)
            if existing is None or existing.is_retracted:
                continue
            self._relations.retract(existing.id)
            result.retracted_relations.append(
                f"{src.name} → {existing.semantic_relation} → {tgt.name}"
            )

    def _step_color(
        self, observation: Observation, resolved_ids: list[str]
    ) -> None:
        self._color.seed_and_diffuse(
            resolved_ids,
            domain_label=observation.domain or observation.task,
        )

    def _resolve_observation_ref(
        self, ref: str, local_refs: dict[str, str]
    ):
        concept_id = local_refs.get(ref)
        if concept_id:
            return self._concepts.get(concept_id)
        return self._live_resolve(ref)
