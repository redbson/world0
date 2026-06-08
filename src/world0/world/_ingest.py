"""Ingest pipeline — observation → concept/relation updates.

The pipeline is a pure orchestrator: it owns no state of its own and
holds only Protocol references to the subsystems it drives.  Each step
is a small private method so individual passes can be unit-tested in
isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from world0.schemas.relation import (
    RelationType,
    normalize_semantic_relation,
    semantic_relation_spec,
)
from world0.schemas.types import ConceptCandidate, IngestResult, Observation

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
    ) -> None:
        self._concepts = concepts
        self._relations = relations
        self._hebbian = hebbian
        self._color = color

    def run(self, observation: Observation) -> IngestResult:
        result = IngestResult()
        resolved_ids: list[str] = []
        local_refs: dict[str, str] = {}

        self._step_concepts(observation, result, resolved_ids, local_refs)
        self._step_relations(observation, result, local_refs)
        self._step_hebbian(observation, resolved_ids, result)
        self._step_descriptions(observation, local_refs)
        self._step_disconfirmation(observation, result, local_refs)
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
            node, is_new = self._concepts.get_or_create(
                name,
                origin=observation.source,
                task=observation.task,
                description=candidate.description
                or observation.descriptions.get(name, ""),
                kind=candidate.kind,
                sense=candidate.sense,
                domain=candidate.domain or observation.domain,
                aliases=candidate.aliases,
            )
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

            semantic_relation = normalize_semantic_relation(relation_name)
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
                    self._relations.reinforce(edge.id, provenance=observation.task)
                result.reinforced_relations.append(label)

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
            node = self._concepts.resolve(name)
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
            semantic_relation = normalize_semantic_relation(relation_name)
            rel_type = semantic_relation_spec(semantic_relation).axis
            existing = self._relations.find_between(src.id, tgt.id, rel_type)
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
        return self._concepts.resolve(ref)
