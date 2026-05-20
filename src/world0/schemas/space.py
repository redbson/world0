"""Space — an isolated concept world.

A space is a complete, independent World 0 instance with its own
concepts, relations, sessions, and state.  Spaces do not share data
with each other; moving a concept across spaces requires an explicit
action (not yet implemented — out of scope for the initial cut).

``Space`` is the schema; the runtime manager lives in
``world0.spaces.api.SpaceRegistry``.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Best-effort ASCII slug — used as the on-disk directory name."""
    lowered = text.strip().lower()
    slug = _SLUG_RE.sub("-", lowered).strip("-")
    return slug or "space"


def new_space_id(name: str) -> str:
    """Stable, human-readable space id: ``<slug>-<random-6>``."""
    return f"{slugify(name)}-{uuid.uuid4().hex[:6]}"


class Space(BaseModel):
    """Metadata for one isolated concept world."""

    id: str
    name: str
    description: str = ""
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_active_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def touch(self) -> None:
        self.last_active_at = datetime.now(timezone.utc)


class SpaceRegistrySnapshot(BaseModel):
    """On-disk shape of ``spaces.json``."""

    spaces: list[Space] = Field(default_factory=list)
    active_space_id: str | None = None


__all__ = [
    "Space",
    "SpaceRegistrySnapshot",
    "new_space_id",
    "slugify",
]
