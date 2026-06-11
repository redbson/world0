"""Similarity linking — explicit near-duplicate edges between concepts.

HippoRAG showed that a second class of automatically generated
similarity ("synonymy") edges, layered on top of declared/typed edges,
measurably improves retrieval by letting activation bridge
near-duplicate concepts that were never explicitly related.  A-MEM and
Mem0-g refine the recipe into two stages: cheap candidate generation,
then an LLM judgment of which candidates genuinely connect.

World 0 follows the two-stage shape:

1. **Recall** — signature-token matching (the same Jaccard scoring used
   for concept consolidation) surfaces candidates that share surface
   vocabulary with the new concept.  Cheap, local, embedding-free.
2. **Precision** — when an LLM provider is configured, the
   ``similarity.judge.system`` prompt decides which candidates denote
   genuinely the same concept, and picks the semantic relation
   (``equivalence`` / ``approximate_equivalence`` /
   ``similarity_kernel``).  This judges *meaning*, so it links
   "vector database" ≈ "vector store" (zero stem overlap) and rejects
   "chain_2" vs "chain_3" (perfect token overlap, different concepts).

Without an LLM the linker falls back to a strict lexical rule: high
Jaccard threshold plus a minimum-signature-size guard.  Either way the
result is a typed, inspectable PARALLEL edge — the similarity is
*materialized* in the concept-world instead of living in a hidden
vector index (Rule 1: explicit structure over latent similarity).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from world0.llm.parsing import extract_json
from world0.prompts import PromptRegistry
from world0.schemas.relation import RelationType

if TYPE_CHECKING:
    from world0.core import ConceptStoreReader, LLMProvider, RelationStore
    from world0.schemas.concept import ConceptNode

# ── Lexical (no-LLM) mode ────────────────────────────────────────────

# Minimum signature similarity at which the lexical fallback creates an
# edge.  Deliberately strict — a wrong similarity edge is worse than a
# missing one, since activation will happily travel across it.
SIMILARITY_EDGE_THRESHOLD: float = 0.5

# A probe needs at least this many signature tokens before lexical
# similarity is meaningful.  Single-token signatures ("chain_2"
# tokenizes to just {"chain"}) match every sibling sharing that one
# word — no compositional evidence of near-duplication exists at size
# one.  The LLM judge is exempt: it sees the full labels and decides
# on meaning, not token overlap.
MIN_SIGNATURE_TOKENS: int = 2

# ── LLM-judged mode ──────────────────────────────────────────────────

# Recall threshold for candidate generation when an LLM judge will
# filter for precision.  Low on purpose: "vector database" vs
# "vector store" only overlaps on one token (Jaccard ≈ 0.33) yet is
# exactly the pair the judge should see.
LLM_CANDIDATE_MIN_SIMILARITY: float = 0.2

# Semantic relations the judge may choose from — all PARALLEL-axis
# similarity flavors.  Anything else the model invents collapses to
# ``similarity_kernel``.
ALLOWED_JUDGE_RELATIONS: frozenset[str] = frozenset(
    {"equivalence", "approximate_equivalence", "similarity_kernel"}
)

# ── Shared ───────────────────────────────────────────────────────────

# Maximum similarity edges created per concept per link() call — keeps
# a generically-named concept from fanning out across the whole world.
MAX_SIMILARITY_LINKS: int = 3

# Auto-generated similarity edges may not exceed this weight no matter
# how confident the source signal is — same ceiling as Hebbian edges,
# so implicit evidence never outranks an Agent-declared relation.
SIMILARITY_WEIGHT_CAP: float = 0.7


class SimilarityLinker:
    """Creates explicit similarity edges between near-duplicate concepts.

    Implements the ``SimilarityLinkerP`` Protocol from ``world0.core``.
    Depends only on the ``ConceptStoreReader`` / ``RelationStore``
    Protocols, plus an optional ``LLMProvider`` for the precision stage.
    """

    def __init__(
        self,
        concepts: "ConceptStoreReader",
        relations: "RelationStore",
        *,
        llm: "LLMProvider | None" = None,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self._concepts = concepts
        self._relations = relations
        self._llm = llm
        self._prompts = prompt_registry or PromptRegistry()

    def link(
        self,
        concept_ids: list[str],
        *,
        provenance: str = "",
    ) -> list[str]:
        """Link each concept to its strongest unlinked near-duplicates.

        With an LLM configured, candidates are recalled lexically and
        judged semantically; otherwise the strict lexical rule applies.
        Returns the ids of relations that were created.
        """
        created: list[str] = []
        for cid in concept_ids:
            node = self._concepts.get(cid)
            if node is None:
                continue
            if self._llm is not None:
                created.extend(self._link_judged(node, provenance))
            else:
                created.extend(self._link_lexical(node, provenance))
        return created

    # ── LLM-judged path ──────────────────────────────────────────────

    def _link_judged(self, node: "ConceptNode", provenance: str) -> list[str]:
        candidates = self._fresh_candidates(
            node, min_similarity=LLM_CANDIDATE_MIN_SIMILARITY
        )
        if not candidates:
            return []
        try:
            verdicts = self._judge(node, [c for c, _ in candidates])
        except Exception:
            # Provider hiccups must not break ingest — fall back to the
            # lexical rule over the candidates we already have.
            return self._apply_lexical_rule(node, candidates, provenance)
        created: list[str] = []
        for index, relation, confidence in verdicts:
            if len(created) >= MAX_SIMILARITY_LINKS:
                break
            if not 0 <= index < len(candidates):
                continue
            candidate, _ = candidates[index]
            edge_id = self._materialize(
                node,
                candidate,
                semantic_relation=relation,
                strength=confidence,
                provenance=provenance or "similarity:llm-judge",
            )
            if edge_id:
                created.append(edge_id)
        return created

    def _judge(
        self, node: "ConceptNode", candidates: list["ConceptNode"]
    ) -> list[tuple[int, str, float]]:
        """Ask the LLM which candidates are genuine near-duplicates.

        Returns ``(candidate_index, semantic_relation, confidence)``
        triples.  Raises on provider/parse failure so the caller can
        fall back.
        """
        system = self._prompts.render("similarity.judge.system")
        payload = {
            "new_concept": self._describe(node),
            "candidates": [
                {"index": i, **self._describe(c)}
                for i, c in enumerate(candidates)
            ],
        }
        raw = self._llm.complete_json(
            system, json.dumps(payload, ensure_ascii=False)
        )
        data = json.loads(extract_json(raw))
        verdicts: list[tuple[int, str, float]] = []
        for item in data.get("links", []):
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if not isinstance(index, int):
                continue
            relation = str(item.get("relation", "")).strip()
            if relation not in ALLOWED_JUDGE_RELATIONS:
                relation = "similarity_kernel"
            try:
                confidence = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = min(1.0, max(0.0, confidence))
            verdicts.append((index, relation, confidence))
        return verdicts

    @staticmethod
    def _describe(node: "ConceptNode") -> dict:
        return {
            "name": node.name,
            "aliases": list(node.aliases),
            "description": node.description,
            "domain": node.domain,
        }

    # ── Lexical path ─────────────────────────────────────────────────

    def _link_lexical(self, node: "ConceptNode", provenance: str) -> list[str]:
        if len(node.signature_tokens()) < MIN_SIGNATURE_TOKENS:
            return []
        candidates = self._fresh_candidates(
            node, min_similarity=SIMILARITY_EDGE_THRESHOLD
        )
        return self._apply_lexical_rule(node, candidates, provenance)

    def _apply_lexical_rule(
        self,
        node: "ConceptNode",
        candidates: list[tuple["ConceptNode", float]],
        provenance: str,
    ) -> list[str]:
        if len(node.signature_tokens()) < MIN_SIGNATURE_TOKENS:
            return []
        created: list[str] = []
        for candidate, similarity in candidates:
            if len(created) >= MAX_SIMILARITY_LINKS:
                break
            if similarity < SIMILARITY_EDGE_THRESHOLD:
                continue
            if len(candidate.signature_tokens()) < MIN_SIGNATURE_TOKENS:
                continue
            edge_id = self._materialize(
                node,
                candidate,
                semantic_relation="similarity_kernel",
                strength=similarity,
                provenance=provenance or "similarity:signature",
            )
            if edge_id:
                created.append(edge_id)
        return created

    # ── Shared mechanics ─────────────────────────────────────────────

    def _fresh_candidates(
        self, node: "ConceptNode", *, min_similarity: float
    ) -> list[tuple["ConceptNode", float]]:
        """Signature-similar concepts not already connected to ``node``."""
        probe = " ".join(
            part
            for part in (node.name, *node.aliases, node.description)
            if part
        )
        scored = self._concepts.find_similar(
            probe,
            domain=node.domain,
            min_similarity=min_similarity,
            limit=MAX_SIMILARITY_LINKS * 2,
        )
        return [
            (candidate, similarity)
            for candidate, similarity in scored
            if candidate.id != node.id
            and not self._relations.find_any_between(node.id, candidate.id)
        ]

    def _materialize(
        self,
        node: "ConceptNode",
        candidate: "ConceptNode",
        *,
        semantic_relation: str,
        strength: float,
        provenance: str,
    ) -> str | None:
        """Create the similarity edge; returns its id (None if it existed)."""
        edge, is_new = self._relations.discover(
            node.id,
            candidate.id,
            RelationType.PARALLEL,
            semantic_relation=semantic_relation,
            provenance=provenance,
            is_explicit=False,
        )
        if not is_new:
            return None
        # Encode the similarity signal as the edge weight so a strong
        # near-duplicate propagates more than a marginal one.
        target = min(strength, SIMILARITY_WEIGHT_CAP)
        self._relations.adjust_strength(
            edge.id,
            weight_delta=target - edge.weight,
            confidence_delta=target - edge.confidence,
        )
        return edge.id
