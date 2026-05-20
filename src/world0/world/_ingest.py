"""Ingest pipeline — observation → concept/relation updates.

The pipeline is a pure orchestrator: it owns no state of its own and
holds only Protocol references to the subsystems it drives.  Each step
is a small private method so individual passes can be unit-tested in
isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from world0.schemas.relation import RelationType
from world0.schemas.types import IngestResult, Observation

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

        self._step_concepts(observation, result, resolved_ids)
        self._step_relations(observation, result)
        self._step_hebbian(observation, resolved_ids, result)
        self._step_descriptions(observation)
        self._step_disconfirmation(observation, result)
        self._step_color(observation, resolved_ids)

        return result

    # ── individual passes ────────────────────────────────────────────

    def _step_concepts(
        self,
        observation: Observation,
        result: IngestResult,
        resolved_ids: list[str],
    ) -> None:
        for name in observation.concepts:
            node, is_new = self._concepts.get_or_create(
                name,
                origin=observation.source,
                task=observation.task,
                description=observation.descriptions.get(name, ""),
                domain=observation.domain,
            )
            # Always reinforce — creation is also an activation event.
            self._concepts.reinforce(
                node.id, source=observation.source, task=observation.task
            )
            (result.new_concepts if is_new else result.reinforced_concepts).append(
                node.name
            )
            resolved_ids.append(node.id)

    def _step_relations(
        self, observation: Observation, result: IngestResult
    ) -> None:
        for src_name, tgt_name, rel_type_str in observation.relations:
            src = self._concepts.resolve(src_name)
            tgt = self._concepts.resolve(tgt_name)
            if not src or not tgt:
                continue

            try:
                rel_type = RelationType(rel_type_str)
            except ValueError:
                rel_type = RelationType.RELATED_TO

            edge, is_new = self._relations.discover(
                src.id, tgt.id, rel_type, provenance=observation.task
            )
            label = f"{src.name} → {rel_type.value} → {tgt.name}"
            if is_new:
                result.new_relations.append(label)
            else:
                self._relations.reinforce(edge.id, provenance=observation.task)
                result.reinforced_relations.append(label)

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

    def _step_descriptions(self, observation: Observation) -> None:
        for name, desc in observation.descriptions.items():
            node = self._concepts.resolve(name)
            if node:
                self._concepts.update_description(node.id, desc)

    def _step_disconfirmation(
        self, observation: Observation, result: IngestResult
    ) -> None:
        for name in observation.weakened:
            node = self._concepts.resolve(name)
            if node:
                self._concepts.weaken(
                    node.id,
                    source=observation.source,
                    task=observation.task,
                )
                result.weakened_concepts.append(node.name)

        for src_name, tgt_name, rel_type_str in observation.contradicted_relations:
            src = self._concepts.resolve(src_name)
            tgt = self._concepts.resolve(tgt_name)
            if not src or not tgt:
                continue
            try:
                rel_type = RelationType(rel_type_str)
            except ValueError:
                rel_type = RelationType.RELATED_TO
            existing = self._relations.find_between(src.id, tgt.id, rel_type)
            if existing is None:
                # Contradiction without an existing edge weakens both
                # endpoint concepts instead — there is nothing else to
                # attach disconfirmation to.
                self._concepts.weaken(
                    src.id, source=observation.source, task=observation.task
                )
                self._concepts.weaken(
                    tgt.id, source=observation.source, task=observation.task
                )
                continue
            self._relations.weaken(existing.id, provenance=observation.task)
            result.weakened_relations.append(
                f"{src.name} → {rel_type.value} → {tgt.name}"
            )

    def _step_color(
        self, observation: Observation, resolved_ids: list[str]
    ) -> None:
        self._color.seed_and_diffuse(
            resolved_ids,
            domain_label=observation.domain or observation.task,
        )
