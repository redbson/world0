"""Reflect pipeline — periodic consolidation.

Five-stage process per ``docs/world0-color-field-dynamics.md``:

1. decay (concepts + relations)
2. community detection / reconciliation, color-source identification
3. color-field dynamics: per-component fade → community injection → settle
4. lifecycle promotions/demotions
5. prune deeply decayed items

Step 3a (per-component fade) intentionally runs *before* fresh
injection so old support-less components drain and do not fight new
color sources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from world0.schemas.types import ReflectResult

if TYPE_CHECKING:
    from world0.communities.manager import CommunityManager
    from world0.core import ColorField, DecayPolicy, LifecyclePolicy


class ReflectPipeline:
    """Drives one reflect cycle.

    Owns no state — the caller (the ``World`` facade) is responsible
    for persisting whatever changed afterwards (typically by flushing
    the concept/relation stores and saving the world state dict).
    """

    def __init__(
        self,
        *,
        decay: DecayPolicy,
        lifecycle: LifecyclePolicy,
        color: ColorField,
        communities: CommunityManager,
    ) -> None:
        self._decay = decay
        self._lifecycle = lifecycle
        self._color = color
        self._communities = communities

    def run(self) -> ReflectResult:
        result = ReflectResult()

        # 1. Decay
        result.decayed_concepts = self._decay.decay_concepts()
        result.decayed_relations = self._decay.decay_relations()

        # 2. Community detection + stability reconciliation
        community_update = self._communities.detect_and_update()
        result.new_communities = community_update.new
        result.pruned_communities = community_update.pruned
        result.color_sources = community_update.color_sources
        result.stable_communities = [
            c.id for c in self._communities.all() if c.is_color_source()
        ]

        # 3. Color-field dynamics (doc §4 five-stage process)
        # 3a. Per-component fade (reaction term) runs *before* fresh
        #     injection so old support-less components drain and do
        #     not fight new color sources.
        self._color.fade_step()
        # 3b. Inject color from communities that have earned the right
        #     to be color sources.
        self._color.seed_from_communities(self._communities.color_sources())
        # 3c. Slow diffusion pass — legacy task/domain settling, kept
        #     as a coarse source (doc §14 transition).
        self._color.settle()

        # 4. Lifecycle
        promoted, demoted = self._lifecycle.evaluate()
        result.promoted_concepts = promoted
        result.demoted_concepts = demoted

        # 5. Prune
        result.pruned_relations = self._decay.prune_relations()
        result.pruned_concepts = self._decay.prune_concepts()

        return result
