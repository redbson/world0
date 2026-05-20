"""SpaceRegistry — fully testable against ``tmp_path`` alone.

These tests exercise the registry without touching any other World 0
brick — proving that the space layer is a pure, isolated addition.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from world0.spaces import DEFAULT_SPACE_ID, SpaceRegistry


def test_fresh_root_starts_empty(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    assert reg.list() == []
    assert reg.active() is None


def test_create_registers_space_and_makes_first_one_active(
    tmp_path: Path,
) -> None:
    reg = SpaceRegistry(tmp_path)
    work = reg.create("work")
    assert work.name == "work"
    assert work.id.startswith("work-")
    assert reg.active() == work

    reload = SpaceRegistry(tmp_path)
    assert [s.id for s in reload.list()] == [work.id]
    assert reload.active() == work


def test_create_rejects_duplicate_name(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    reg.create("work")
    with pytest.raises(ValueError):
        reg.create("work")


def test_resolve_by_id_and_name(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    home = reg.create("Home Notes")
    assert reg.resolve(home.id) == home
    assert reg.resolve("Home Notes") == home
    assert reg.resolve("home-notes") == home
    assert reg.resolve("missing") is None


def test_path_for_creates_dedicated_subdir(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    proj = reg.create("proj")
    path = reg.path_for(proj.id)
    assert path == tmp_path / "spaces" / proj.id
    assert path.is_dir()


def test_set_active_switches_selection(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    a = reg.create("a")
    b = reg.create("b")
    assert reg.active() == a
    assert reg.set_active(b.id) is True
    assert reg.active() == b
    assert reg.set_active("nope") is False


def test_rename_updates_name_but_keeps_id(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    s = reg.create("old")
    assert reg.rename(s.id, "new") is True
    fresh = SpaceRegistry(tmp_path)
    rehydrated = fresh.get(s.id)
    assert rehydrated is not None
    assert rehydrated.name == "new"


def test_rename_rejects_duplicate(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    a = reg.create("a")
    b = reg.create("b")
    with pytest.raises(ValueError):
        reg.rename(b.id, "a")
    # untouched
    assert reg.get(b.id).name == "b"
    assert reg.get(a.id).name == "a"


def test_delete_reassigns_active_and_purges_data(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    a = reg.create("a")
    b = reg.create("b")
    # drop some bytes into a's dir to prove purge works
    (reg.path_for(a.id) / "concepts").mkdir()
    (reg.path_for(a.id) / "concepts" / "c1.json").write_text("{}")

    assert reg.active() == a
    assert reg.delete(a.id, purge_data=True) is True
    assert reg.active() == b
    assert reg.get(a.id) is None
    assert not (tmp_path / "spaces" / a.id).exists()


def test_delete_without_purge_preserves_data(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    s = reg.create("keep-data")
    path = reg.path_for(s.id)
    (path / "marker").write_text("still here")
    assert reg.delete(s.id) is True
    assert (path / "marker").read_text() == "still here"


def test_delete_unknown_returns_false(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    assert reg.delete("ghost") is False


def test_legacy_layout_registers_default_space(tmp_path: Path) -> None:
    # Simulate a pre-space install: top-level concepts/ already present.
    (tmp_path / "concepts").mkdir()
    (tmp_path / "relations").mkdir()

    reg = SpaceRegistry(tmp_path)
    spaces = reg.list()
    assert len(spaces) == 1
    default = spaces[0]
    assert default.id == DEFAULT_SPACE_ID
    # Legacy default reuses the registry root — no data movement.
    assert reg.path_for(default.id) == tmp_path
    assert reg.active() == default


def test_legacy_default_survives_reload(tmp_path: Path) -> None:
    (tmp_path / "concepts").mkdir()
    SpaceRegistry(tmp_path)  # initial registration
    reg = SpaceRegistry(tmp_path)
    assert reg.active() is not None
    assert reg.active().id == DEFAULT_SPACE_ID


def test_corrupt_registry_does_not_wipe_spaces_dir(tmp_path: Path) -> None:
    # Pre-seed a space's on-disk directory.
    target = tmp_path / "spaces" / "some-id"
    (target / "concepts").mkdir(parents=True)
    (target / "concepts" / "c.json").write_text("{}")

    # Write garbage registry JSON.
    (tmp_path / "spaces.json").write_text("{not json")

    # Registry should rebuild without touching the on-disk store.
    SpaceRegistry(tmp_path)
    assert (target / "concepts" / "c.json").exists()


def test_touch_updates_last_active(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    s = reg.create("x")
    original = s.last_active_at
    reg.touch(s.id)
    refreshed = SpaceRegistry(tmp_path).get(s.id)
    assert refreshed is not None
    assert refreshed.last_active_at >= original
