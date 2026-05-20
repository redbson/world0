"""SpaceRegistry — manage multiple isolated concept worlds.

Storage layout under ``<root>/``::

    spaces.json              # registry: list of spaces + active id
    spaces/
        <space_id>/          # each space is a complete World 0 store
            concepts/
            relations/
            state.json
            sessions/        # conversation sessions, scoped to this space

Backwards compatibility: if ``<root>`` contains a *legacy* top-level
``concepts/`` directory (pre-space layout), a ``default`` space is
registered whose on-disk path is ``<root>`` itself — so the existing
store keeps working without any data movement.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from world0.schemas.space import (
    Space,
    SpaceRegistrySnapshot,
    new_space_id,
    slugify,
)

DEFAULT_SPACE_ID = "default"
DEFAULT_SPACE_NAME = "default"


class SpaceRegistry:
    """Persistence + lookup for the collection of spaces under ``root``."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser()
        self._root.mkdir(parents=True, exist_ok=True)
        self._registry_path = self._root / "spaces.json"
        self._spaces_dir = self._root / "spaces"
        self._snapshot = self._load_or_init()

    # ── Load / save ───────────────────────────────────────────────────

    def _load_or_init(self) -> SpaceRegistrySnapshot:
        if self._registry_path.exists():
            try:
                return SpaceRegistrySnapshot.model_validate_json(
                    self._registry_path.read_text(encoding="utf-8")
                )
            except Exception:
                # Corrupt registry — rebuild from scratch but do not
                # touch on-disk per-space directories.
                pass

        snapshot = SpaceRegistrySnapshot()
        # Legacy layout: top-level concepts/ → register a default space
        # that points at <root> itself.
        if (self._root / "concepts").exists():
            legacy = Space(
                id=DEFAULT_SPACE_ID,
                name=DEFAULT_SPACE_NAME,
                description="Legacy top-level store",
            )
            snapshot.spaces.append(legacy)
            snapshot.active_space_id = legacy.id
        self._save(snapshot)
        return snapshot

    def _save(self, snapshot: SpaceRegistrySnapshot | None = None) -> None:
        snap = snapshot if snapshot is not None else self._snapshot
        self._registry_path.write_text(
            snap.model_dump_json(indent=2), encoding="utf-8"
        )

    # ── Read surface ──────────────────────────────────────────────────

    @property
    def root(self) -> Path:
        return self._root

    def list(self) -> list[Space]:
        return list(self._snapshot.spaces)

    def get(self, space_id: str) -> Space | None:
        for s in self._snapshot.spaces:
            if s.id == space_id:
                return s
        return None

    def resolve(self, name_or_id: str) -> Space | None:
        """Match by id first, then by exact name, then by slug-equivalent."""
        hit = self.get(name_or_id)
        if hit:
            return hit
        target_slug = slugify(name_or_id)
        for s in self._snapshot.spaces:
            if s.name == name_or_id or slugify(s.name) == target_slug:
                return s
        return None

    def active(self) -> Space | None:
        if self._snapshot.active_space_id is None:
            return None
        return self.get(self._snapshot.active_space_id)

    def path_for(self, space_id: str) -> Path:
        """Return the on-disk store path for a space (creating nothing)."""
        if space_id == DEFAULT_SPACE_ID:
            # Legacy-compatible default lives at the registry root.
            legacy_root = self._root
            if (legacy_root / "concepts").exists() or not (
                self._spaces_dir / space_id
            ).exists():
                return legacy_root
        return self._spaces_dir / space_id

    # ── Write surface ─────────────────────────────────────────────────

    def create(self, name: str, *, description: str = "") -> Space:
        """Create a new space with a fresh store directory."""
        name = name.strip()
        if not name:
            raise ValueError("Space name must not be empty.")
        if self.resolve(name) is not None:
            raise ValueError(f"Space {name!r} already exists.")

        space = Space(
            id=new_space_id(name),
            name=name,
            description=description,
        )
        # Create an empty per-space directory so downstream code
        # (JsonStore, SessionStore) can immediately write into it.
        (self._spaces_dir / space.id).mkdir(parents=True, exist_ok=True)

        self._snapshot.spaces.append(space)
        if self._snapshot.active_space_id is None:
            self._snapshot.active_space_id = space.id
        self._save()
        return space

    def rename(self, space_id: str, new_name: str) -> bool:
        space = self.get(space_id)
        if space is None:
            return False
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("New name must not be empty.")
        # Reject duplicates (ignore the space itself).
        for s in self._snapshot.spaces:
            if s.id != space_id and (
                s.name == new_name
                or slugify(s.name) == slugify(new_name)
            ):
                raise ValueError(f"Space {new_name!r} already exists.")
        space.name = new_name
        self._save()
        return True

    def set_active(self, space_id: str) -> bool:
        if self.get(space_id) is None:
            return False
        self._snapshot.active_space_id = space_id
        self._save()
        return True

    def touch(self, space_id: str) -> None:
        """Bump ``last_active_at``. Swallow unknown ids — cheap telemetry."""
        space = self.get(space_id)
        if space is not None:
            space.touch()
            self._save()

    def delete(self, space_id: str, *, purge_data: bool = False) -> bool:
        """Remove a space from the registry.

        With ``purge_data=True`` the on-disk store directory is removed.
        The legacy default space (whose path == registry root) is never
        purged — callers must handle that themselves.
        """
        space = self.get(space_id)
        if space is None:
            return False

        self._snapshot.spaces = [
            s for s in self._snapshot.spaces if s.id != space_id
        ]
        if self._snapshot.active_space_id == space_id:
            self._snapshot.active_space_id = (
                self._snapshot.spaces[0].id
                if self._snapshot.spaces
                else None
            )

        if purge_data:
            path = self._spaces_dir / space_id
            # Refuse to rm anything outside spaces/<id>/ — guards the
            # legacy default case where path_for() returns the root.
            if path.exists() and path.resolve().is_relative_to(
                self._spaces_dir.resolve()
            ):
                shutil.rmtree(path)

        self._save()
        return True


__all__ = [
    "SpaceRegistry",
    "DEFAULT_SPACE_ID",
    "DEFAULT_SPACE_NAME",
]
