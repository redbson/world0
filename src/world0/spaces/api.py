"""Public API for the ``spaces`` brick.

A ``Space`` is an isolated World 0 concept world.  The registry lets
you list, create, switch between, and delete spaces under a single
root directory — each space owns its own concepts, relations, state,
and sessions.
"""

from __future__ import annotations

from world0.schemas.space import Space, SpaceRegistrySnapshot
from world0.spaces._registry import (
    DEFAULT_SPACE_ID,
    DEFAULT_SPACE_NAME,
    SpaceRegistry,
)

__all__ = [
    "Space",
    "SpaceRegistry",
    "SpaceRegistrySnapshot",
    "DEFAULT_SPACE_ID",
    "DEFAULT_SPACE_NAME",
]
