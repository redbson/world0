"""Metacognitive monitoring of a projection (``docs/mc``, HOT-2).

A projection hands the Agent a small set of concepts and relations; this
module adds what the world knows *about that knowledge*:

- **reliability** of each concept from its time-independent evidence
  (``ConceptNode.evidence()``): a concept seen once or twice is
  ``tentative``, one confirmed across many observations is
  ``well_evidenced``;
- **contested claims**: a concept pair carrying opposing explicit claims
  (a negative-axis claim next to a positive / parallel one — the same
  opposition ``RelationEdge.opposes`` uses when opposite claims weaken
  each other on ingest).  Both beliefs are reported; the pair is
  ``contested`` when neither leads by ``CONTEST_MARGIN``, ``leaning``
  otherwise.

Before this, a projection listed ``pytorch enables gpu cluster`` and
``pytorch conflict gpu cluster`` side by side without their beliefs, and
a concept mentioned once looked as solid as one confirmed fifty times.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from world0.schemas.concept import ConceptNode
from world0.schemas.relation import RelationEdge
from world0.schemas.types import ContestedClaim, EpistemicStatus

# Evidence below this: seen at most about twice (one mention ≈ 0.06,
# two ≈ 0.13, three ≈ 0.19).
TENTATIVE_EVIDENCE: float = 0.15

# Evidence at or above this: confirmed across roughly a dozen or more
# observations (ten mentions ≈ 0.46, twenty ≈ 0.64).
WELL_EVIDENCED: float = 0.5

# Opposing claims whose beliefs differ by less than this are contested.
CONTEST_MARGIN: float = 0.25


def reliability_level(node: ConceptNode) -> str:
    evidence = node.evidence()
    if evidence < TENTATIVE_EVIDENCE:
        return "tentative"
    if evidence >= WELL_EVIDENCED:
        return "well_evidenced"
    return "moderate"


def assess(
    concepts: Iterable[ConceptNode], relations: Iterable[RelationEdge]
) -> EpistemicStatus:
    """Epistemic status of a projection's concepts and relations."""
    status = EpistemicStatus(
        reliability={node.id: reliability_level(node) for node in concepts}
    )
    by_pair: dict[frozenset[str], list[RelationEdge]] = defaultdict(list)
    for rel in relations:
        if rel.is_explicit and rel.semantic_relation != "generic_relation":
            by_pair[frozenset((rel.source_id, rel.target_id))].append(rel)
    for pair in sorted(by_pair, key=lambda p: sorted(p)):
        claims = by_pair[pair]
        if not any(a.opposes(b.relation_type, b.semantic_relation) for a in claims for b in claims):
            continue
        ranked = sorted(claims, key=lambda r: (-r.probability, r.id))
        lead = ranked[0]
        opposing = [r for r in ranked[1:] if lead.opposes(r.relation_type, r.semantic_relation)]
        margin = lead.probability - opposing[0].probability
        status.contested.append(ContestedClaim(
            source_id=lead.source_id,
            target_id=lead.target_id,
            claims=[(r.id, r.semantic_relation, round(r.probability, 4)) for r in ranked],
            leading=lead.id,
            margin=round(margin, 4),
            status="contested" if margin < CONTEST_MARGIN else "leaning",
        ))
    return status
