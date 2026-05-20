"""Compose a WorldStatus snapshot from the concept/relation/community state."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from world0.schemas.types import WorldStatus

if TYPE_CHECKING:
    from world0.communities.manager import CommunityManager
    from world0.core import ConceptStoreReader, RelationStoreReader


def build_status(
    *,
    concepts: ConceptStoreReader,
    relations: RelationStoreReader,
    communities: CommunityManager,
    last_reflect_iso: str | None,
) -> WorldStatus:
    all_concepts = concepts.all()
    by_maturity: dict[str, int] = {}
    total_confidence = 0.0
    purity_total = 0.0
    purity_n = 0
    bridge_count = 0
    for c in all_concepts:
        by_maturity[c.maturity.value] = by_maturity.get(c.maturity.value, 0) + 1
        total_confidence += c.confidence
        if c.domain_profile:
            purity_total += c.color_purity()
            purity_n += 1
        if c.is_bridge():
            bridge_count += 1

    all_communities = communities.all()
    stable_communities = sum(
        1 for c in all_communities if c.is_color_source()
    )

    return WorldStatus(
        total_concepts=len(all_concepts),
        total_relations=len(relations),
        by_maturity=by_maturity,
        avg_confidence=(
            total_confidence / len(all_concepts) if all_concepts else 0.0
        ),
        last_reflect=(
            datetime.fromisoformat(last_reflect_iso)
            if last_reflect_iso
            else None
        ),
        total_communities=len(all_communities),
        stable_communities=stable_communities,
        bridge_concepts=bridge_count,
        avg_color_purity=(purity_total / purity_n) if purity_n else 1.0,
    )
