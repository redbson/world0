"""Community detection — discovers emergent concept clusters.

Label propagation over the weighted concept graph.  The "coupling"
between two concepts for detection purposes follows the spirit of doc
§16.1: it is not a single relation weight but a composite — relation
weight × relation-type factor × temporal relevance.  This matches the
cognitive idea that recent, strongly typed edges pull harder than
stale generic co-occurrences.

The algorithm is a deterministic synchronous label-propagation pass:

1. Each node starts with its own id as label.
2. Iterate: every node adopts the label of its most strongly coupled
   neighborhood.  Ties are broken by lexicographically smallest label,
   so the outcome is stable across process runs (PYTHONHASHSEED).
3. Stop when no node changes label, or after ``max_iters``.
4. Group nodes by label → candidate communities.  Labels whose group
   is below ``min_size`` are discarded as noise.

This is not a spectral method (doc §18.1); label propagation is a
well-known cheap approximation to modularity-maximising clustering and
is sufficient for the observation layer.  The ``detect`` output feeds
into ``CommunityManager`` which is responsible for persistence and
stability accumulation across reflect cycles.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from world0.dynamics.coefficients import (
    CONCEPT_TEMPORAL_HL,
    RELATION_TEMPORAL_HL,
    RELATION_TYPE_FACTOR,
)
from world0.schemas.community import Community, signature_id, community_color_for

if TYPE_CHECKING:
    from world0.core import ConceptStore, RelationStore

# Label-propagation iteration cap — small because the algorithm
# converges quickly on typical cognitive graphs (<200 concepts).
DETECTION_MAX_ITERS: int = 8

# Minimum nodes a cluster must have to count as a community.
# Doc §11.1: short-term or singleton co-occurrences must not birth color.
MIN_COMMUNITY_SIZE: int = 3

# Fraction of highest-internal-degree nodes classified as community core.
CORE_FRACTION: float = 0.35


class CommunityDetector:
    """Detects candidate communities in the current concept graph.

    Implements the ``CommunityDetectorP`` Protocol from ``world0.core``.
    """

    def __init__(
        self,
        concepts: "ConceptStore",
        relations: "RelationStore",
    ) -> None:
        self._concepts = concepts
        self._relations = relations

    def detect(
        self,
        *,
        max_iters: int = DETECTION_MAX_ITERS,
        min_size: int = MIN_COMMUNITY_SIZE,
    ) -> list[Community]:
        """Return candidate communities for the current graph snapshot.

        The caller (``CommunityManager``) is responsible for matching
        these against persisted communities — this method is stateless.
        """
        node_ids = sorted(n.id for n in self._concepts.all())
        if len(node_ids) < min_size:
            return []

        coupling = self._build_coupling()
        if not coupling:
            return []

        labels: dict[str, str] = {nid: nid for nid in node_ids}

        for _ in range(max_iters):
            changed = False
            for nid in node_ids:
                neighbors = coupling.get(nid)
                if not neighbors:
                    continue
                votes: dict[str, float] = defaultdict(float)
                for neighbor_id, weight in neighbors.items():
                    votes[labels[neighbor_id]] += weight
                # Highest weighted vote; tie-breaker: lexicographic
                # smallest label for determinism across process runs.
                best_label = max(votes.items(), key=lambda kv: (kv[1], -_rank(kv[0])))[0]
                if best_label != labels[nid]:
                    labels[nid] = best_label
                    changed = True
            if not changed:
                break

        return self._build_communities(labels, coupling, min_size)

    def _build_coupling(self) -> dict[str, dict[str, float]]:
        """Effective coupling K_ij(t) from the current relation layer.

        `weight × type-factor × relation-freshness × mean(endpoint
        freshness)` — any directional edge is folded symmetrically into
        the coupling graph because community structure is undirected.
        """
        coupling: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for edge in self._relations.all():
            type_factor = RELATION_TYPE_FACTOR.get(edge.relation_type, 0.5)
            if type_factor <= 0 or edge.weight <= 0:
                continue
            src_node = self._concepts.get(edge.source_id)
            tgt_node = self._concepts.get(edge.target_id)
            if src_node is None or tgt_node is None:
                continue
            rel_fresh = edge.temporal_relevance(RELATION_TEMPORAL_HL)
            endpoint_fresh = 0.5 * (
                src_node.temporal_relevance(CONCEPT_TEMPORAL_HL)
                + tgt_node.temporal_relevance(CONCEPT_TEMPORAL_HL)
            )
            k = edge.weight * type_factor * rel_fresh * endpoint_fresh
            if k <= 0:
                continue
            coupling[edge.source_id][edge.target_id] += k
            coupling[edge.target_id][edge.source_id] += k
        return coupling

    @staticmethod
    def _build_communities(
        labels: dict[str, str],
        coupling: dict[str, dict[str, float]],
        min_size: int,
    ) -> list[Community]:
        groups: dict[str, list[str]] = defaultdict(list)
        for nid, lbl in labels.items():
            groups[lbl].append(nid)

        communities: list[Community] = []
        for members in groups.values():
            if len(members) < min_size:
                continue
            member_ids = sorted(members)
            # Core = nodes with the highest internal coupling sum within
            # the community; doc §5.2.  Round up so every community has
            # at least one core member.
            member_set = set(member_ids)
            internal_degree: dict[str, float] = {}
            for nid in member_ids:
                internal_degree[nid] = sum(
                    w
                    for neighbor, w in coupling.get(nid, {}).items()
                    if neighbor in member_set
                )
            core_count = max(1, int(round(len(member_ids) * CORE_FRACTION)))
            core_ids = sorted(
                member_ids,
                key=lambda nid: (-internal_degree.get(nid, 0.0), nid),
            )[:core_count]

            community_id = signature_id(member_ids)
            communities.append(
                Community(
                    id=community_id,
                    member_ids=member_ids,
                    core_ids=core_ids,
                    color_hex=community_color_for(community_id),
                    stability=1,
                    seen_count=1,
                    last_detected_size=len(member_ids),
                )
            )

        # Deterministic order: largest first, then by id.
        communities.sort(key=lambda c: (-len(c.member_ids), c.id))
        return communities


def _rank(label: str) -> int:
    """Cheap lexicographic-inverse rank used as tie-breaker."""
    # Using ``int.from_bytes`` gives a stable per-process integer from
    # the label bytes without depending on Python's hash randomisation.
    return int.from_bytes(label.encode("utf-8")[:8].ljust(8, b"\x00"), "big")
