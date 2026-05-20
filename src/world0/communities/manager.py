"""Community lifecycle — persistence, stability, promotion to color source.

Detection is stateless (see ``CommunityDetector``).  This manager is
the place where short-lived coincidences are separated from persistent
structure:

- freshly detected cluster with high member overlap against an existing
  community → that community's stability counter is incremented;
- cluster with no close match → recorded as a new candidate at
  ``stability = 1``;
- previously persisted community that was *not* detected this cycle →
  stability decremented; removed once it reaches zero.

Only communities whose stability has crossed ``MIN_COLOR_SOURCE_STABILITY``
act as color sources.  This is the "structure before color" rule
operationalised: transient co-occurrences never get to drive the color
field.
"""

from __future__ import annotations

from world0.dynamics.community import CommunityDetector
from world0.schemas.community import Community

# Minimum Jaccard overlap between a freshly detected cluster and a
# persisted community for them to be treated as the same community.
# Chosen to be tolerant of small membership drift without collapsing
# genuinely different clusters.
COMMUNITY_MATCH_THRESHOLD: float = 0.45

# Stability counters.  A community only becomes a color source at
# `MIN_COLOR_SOURCE_STABILITY`; it survives `MAX_STABILITY` cycles
# without re-detection before being dropped.
MIN_COLOR_SOURCE_STABILITY: int = 2
MAX_STABILITY: int = 6


class CommunityManager:
    """Keeps the set of persisted communities in sync with the graph."""

    def __init__(
        self,
        detector: CommunityDetector,
        *,
        initial: list[Community] | None = None,
    ) -> None:
        self._detector = detector
        self._communities: dict[str, Community] = {}
        if initial:
            for c in initial:
                self._communities[c.id] = c

    # ── public surface ───────────────────────────────────────────────

    def all(self) -> list[Community]:
        return list(self._communities.values())

    def color_sources(self) -> list[Community]:
        """Communities currently eligible to seed color."""
        return [
            c
            for c in self._communities.values()
            if c.is_color_source(min_stability=MIN_COLOR_SOURCE_STABILITY)
        ]

    def snapshot(self) -> list[dict]:
        """JSON-serialisable snapshot used by ``JsonStore.state``."""
        return [c.model_dump(mode="json") for c in self._communities.values()]

    @classmethod
    def from_snapshot(
        cls,
        snapshot: list[dict] | None,
        detector: CommunityDetector,
    ) -> "CommunityManager":
        if not snapshot:
            return cls(detector)
        communities = [Community.model_validate(item) for item in snapshot]
        return cls(detector, initial=communities)

    # ── detect + reconcile ───────────────────────────────────────────

    def detect_and_update(self) -> "CommunityUpdateResult":
        """Run detection and fold the results into the persisted set.

        Returns a summary describing what was created, reinforced or
        pruned, so ``reflect()`` can expose it to callers.
        """
        detected = self._detector.detect()
        detected_by_id: dict[str, Community] = {c.id: c for c in detected}

        matched_existing: set[str] = set()
        matched_detected: set[str] = set()

        # Pass 1: direct id match (same exact member set).
        for cand_id, candidate in detected_by_id.items():
            if cand_id in self._communities:
                existing = self._communities[cand_id]
                existing.stability = min(MAX_STABILITY, existing.stability + 1)
                existing.touch(
                    members=candidate.member_ids,
                    core_ids=candidate.core_ids,
                )
                matched_existing.add(cand_id)
                matched_detected.add(cand_id)

        # Pass 2: fuzzy member-overlap match for the rest.
        for cand_id, candidate in detected_by_id.items():
            if cand_id in matched_detected:
                continue
            best: tuple[str, float] | None = None
            cand_members = candidate.member_set()
            for existing_id, existing in self._communities.items():
                if existing_id in matched_existing:
                    continue
                sim = existing.jaccard(cand_members)
                if best is None or sim > best[1]:
                    best = (existing_id, sim)
            if best is not None and best[1] >= COMMUNITY_MATCH_THRESHOLD:
                existing = self._communities[best[0]]
                existing.stability = min(MAX_STABILITY, existing.stability + 1)
                existing.touch(
                    members=candidate.member_ids,
                    core_ids=candidate.core_ids,
                )
                matched_existing.add(best[0])
                matched_detected.add(cand_id)

        # Pass 4: decay communities that did not re-appear this cycle.
        # Run *before* registering brand-new candidates so a fresh
        # community at stability=1 is not immediately decremented to 0
        # in its very first cycle.
        pruned: list[str] = []
        for existing_id in list(self._communities.keys()):
            if existing_id in matched_existing:
                continue
            existing = self._communities[existing_id]
            existing.stability -= 1
            if existing.stability <= 0:
                pruned.append(existing_id)
                del self._communities[existing_id]

        # Pass 3 (now last): register fresh candidates as new communities.
        new_ids: list[str] = []
        for cand_id, candidate in detected_by_id.items():
            if cand_id in matched_detected:
                continue
            self._communities[cand_id] = candidate
            new_ids.append(cand_id)

        color_source_ids = [c.id for c in self.color_sources()]

        return CommunityUpdateResult(
            detected=list(detected_by_id.keys()),
            new=new_ids,
            matched=list(matched_existing),
            pruned=pruned,
            color_sources=color_source_ids,
        )


class CommunityUpdateResult:
    """Summary of one reconcile pass (not a Pydantic model — transient)."""

    __slots__ = ("detected", "new", "matched", "pruned", "color_sources")

    def __init__(
        self,
        *,
        detected: list[str],
        new: list[str],
        matched: list[str],
        pruned: list[str],
        color_sources: list[str],
    ) -> None:
        self.detected = detected
        self.new = new
        self.matched = matched
        self.pruned = pruned
        self.color_sources = color_sources
