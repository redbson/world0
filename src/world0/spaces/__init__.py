"""Space registry — multi-tenant isolated concept worlds.

Each space is a complete, independent World 0 store.  The registry
persists the list of spaces, the active selection, and resolves space
names/ids into on-disk paths you can hand to ``World(store_path=...)``.

Public surface::

    from world0.spaces import SpaceRegistry, Space
"""

from world0.spaces.api import (
    DEFAULT_SPACE_ID,
    DEFAULT_SPACE_NAME,
    Space,
    SpaceRegistry,
    SpaceRegistrySnapshot,
)

__all__ = [
    "DEFAULT_SPACE_ID",
    "DEFAULT_SPACE_NAME",
    "Space",
    "SpaceRegistry",
    "SpaceRegistrySnapshot",
]
