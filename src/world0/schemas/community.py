"""Community — a self-organized concept cluster that carries a color identity.

A Community is the operational unit of the color-field dynamics described
in ``docs/world0-color-field-dynamics.md``.  Communities are *not*
predefined domains: they emerge from relation coupling, become stable
over multiple reflect cycles, and only then birth a color source that
diffuses into their members.

Key properties:

- ``id``: deterministic, derived from the initial sorted member-id set.
- ``color_hex``: deterministic, derived from ``id`` via the same HLS
  scheme used for domain colors so that visualisations stay consistent.
- ``stability``: integer counter. Each reflect cycle that re-detects
  this same cluster increments it; cycles that miss it decrement it.
  A community only becomes a color *source* once stability crosses a
  threshold, which matches the "structure before color" principle.
- ``core_ids``: members whose internal degree ranks them as the
  structural core of the cluster (per §5.2 core/inner ring/boundary).
"""

from __future__ import annotations

import colorsys
import hashlib
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def community_color_for(community_id: str) -> str:
    """Deterministic color assignment for a community id."""
    digest = hashlib.sha1(community_id.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") % 360
    saturation = 0.55 + (digest[2] / 255.0) * 0.2
    lightness = 0.48 + (digest[3] / 255.0) * 0.12
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, lightness, saturation)
    return "#{:02x}{:02x}{:02x}".format(
        round(r * 255), round(g * 255), round(b * 255)
    )


def signature_id(member_ids: list[str]) -> str:
    """Deterministic id derived from a sorted member-id fingerprint.

    Using the sorted full-member hash means an identical member set
    always yields the same id across processes — critical for stability
    tracking to hold across restarts.  Overlap-based reconciliation in
    ``CommunityManager`` handles the case where members drift slightly.
    """
    joined = "|".join(sorted(member_ids))
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()
    return f"com_{digest[:12]}"


class Community(BaseModel):
    """A candidate or confirmed cognitive community.

    Candidates start at ``stability=1`` and must accumulate more
    detections to be promoted to a color source.
    """

    id: str
    member_ids: list[str] = Field(default_factory=list)
    core_ids: list[str] = Field(default_factory=list)
    color_hex: str = ""
    stability: int = 1
    seen_count: int = 1
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_seen: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_detected_size: int = 0

    def is_color_source(self, min_stability: int = 2, min_size: int = 3) -> bool:
        """Whether this community may inject color into its core members.

        Matches doc §4.3: color identity only after the community has
        persisted over enough cycles to be considered sub-stable.
        """
        return (
            self.stability >= min_stability
            and len(self.member_ids) >= min_size
        )

    def member_set(self) -> set[str]:
        return set(self.member_ids)

    def jaccard(self, other_members: set[str]) -> float:
        a = self.member_set()
        if not a or not other_members:
            return 0.0
        return len(a & other_members) / len(a | other_members)

    def touch(self, *, members: list[str], core_ids: list[str]) -> None:
        """Record a fresh detection of this community."""
        self.member_ids = members
        self.core_ids = core_ids
        self.last_detected_size = len(members)
        self.seen_count += 1
        self.last_seen = datetime.now(timezone.utc)
