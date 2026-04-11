"""Projection engine — generates LLM-prompt-ready cognitive views.

Uses MMR (Maximal Marginal Relevance) selection to balance activation
score against diversity, preventing hub concepts from monopolizing
the projection.
"""

from __future__ import annotations

from world0.concepts.manager import ConceptManager
from world0.relations.manager import RelationManager
from world0.schemas.types import Projection

# MMR diversity weight. 0 = pure score ranking, 1 = pure diversity.
# 0.3 gives a mild diversity nudge while keeping relevance primary.
MMR_LAMBDA: float = 0.3


class ProjectionEngine:
    """Generates a Projection from activation scores.

    The projection is the operational output — it is what gets injected
    into the Agent's prompt to shape its reasoning.
    """

    def __init__(
        self, concepts: ConceptManager, relations: RelationManager
    ) -> None:
        self._concepts = concepts
        self._relations = relations

    def project(
        self,
        activations: dict[str, float],
        *,
        max_concepts: int = 15,
        min_activation: float = 0.01,
        task: str = "",
    ) -> Projection:
        """Build a cognitive projection from activation scores.

        1. Filter by minimum activation
        2. MMR greedy selection: balance score vs diversity
        3. Include relations between selected concepts
        4. Return LLM-prompt-ready Projection
        """
        # Filter candidates
        candidates = {
            cid: score
            for cid, score in activations.items()
            if score >= min_activation
        }

        if not candidates:
            return Projection(task=task)

        # Normalize scores to [0, 1] for MMR
        max_score = max(candidates.values()) if candidates else 1.0
        if max_score == 0:
            max_score = 1.0

        # Build neighbor sets for similarity computation
        neighbor_sets: dict[str, set[str]] = {}
        for cid in candidates:
            neighbor_sets[cid] = set(self._relations.neighbors(cid))

        # MMR greedy selection
        selected: list[str] = []
        remaining = set(candidates.keys())

        while remaining and len(selected) < max_concepts:
            best_id = None
            best_mmr = -float("inf")

            for cid in remaining:
                relevance = candidates[cid] / max_score

                # Redundancy: max Jaccard similarity to any already-selected
                if selected:
                    neighbors_c = neighbor_sets.get(cid, set())
                    max_sim = 0.0
                    for sid in selected:
                        neighbors_s = neighbor_sets.get(sid, set())
                        union = neighbors_c | neighbors_s
                        if union:
                            sim = len(neighbors_c & neighbors_s) / len(union)
                        else:
                            sim = 0.0
                        if sim > max_sim:
                            max_sim = sim
                    redundancy = max_sim
                else:
                    redundancy = 0.0

                mmr = (1 - MMR_LAMBDA) * relevance - MMR_LAMBDA * redundancy

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_id = cid

            if best_id is None:
                break

            selected.append(best_id)
            remaining.discard(best_id)

        selected_ids = set(selected)
        selected_scores = {cid: candidates[cid] for cid in selected_ids}

        # Resolve concepts
        concepts = []
        for cid in selected_ids:
            node = self._concepts.get(cid)
            if node:
                concepts.append(node)

        # Gather relations between selected concepts
        relations = []
        seen: set[str] = set()
        for cid in selected_ids:
            for rel in self._relations.for_concept(cid):
                if rel.id in seen:
                    continue
                if rel.source_id in selected_ids and rel.target_id in selected_ids:
                    relations.append(rel)
                    seen.add(rel.id)

        return Projection(
            concepts=concepts,
            relations=relations,
            activation_scores=selected_scores,
            task=task,
        )
