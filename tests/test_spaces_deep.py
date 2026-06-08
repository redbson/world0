"""Deep, deterministic tests for Spaces — isolated concept worlds.

Exercises ``world0.spaces.SpaceRegistry`` (and its schema helpers) against
``tmp_path`` only.  These tests focus on the *deep* behaviors not already
covered by ``src/world0/spaces/tests/test_registry.py``:

  * full create / list / switch / delete lifecycle invariants
  * real data isolation between spaces via a JsonStore bound to each
    space's ``path_for`` directory
  * legacy (pre-space) default-space backward compatibility, including
    real concept data living at the registry root
  * persistence round-trips across freshly constructed registries
  * edge / negative cases (empty names, duplicates, missing ids, deleting
    the active space, deleting the legacy default)
  * slug / id normalization behavior
"""

from __future__ import annotations

from pathlib import Path

import pytest

from world0.schemas.concept import ConceptNode
from world0.schemas.space import (
    Space,
    SpaceRegistrySnapshot,
    new_space_id,
    slugify,
)
from world0.spaces import (
    DEFAULT_SPACE_ID,
    DEFAULT_SPACE_NAME,
    SpaceRegistry,
)
from world0.store.json_store import JsonStore


# ── helpers ───────────────────────────────────────────────────────────


def _store_for(reg: SpaceRegistry, space_id: str) -> JsonStore:
    """A real JsonStore rooted at a space's on-disk directory."""
    return JsonStore(reg.path_for(space_id))


def _concept(name: str) -> ConceptNode:
    return ConceptNode(name=name)


def _concept_names(store: JsonStore) -> set[str]:
    return {c.name for c in store.load_all_concepts()}


# ── lifecycle: create / list / switch / delete ────────────────────────


def test_full_lifecycle_create_list_switch_delete(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)

    # create
    a = reg.create("Alpha")
    b = reg.create("Beta")
    c = reg.create("Gamma")

    # list preserves insertion order and contains exactly what we created
    assert [s.name for s in reg.list()] == ["Alpha", "Beta", "Gamma"]
    assert {s.id for s in reg.list()} == {a.id, b.id, c.id}

    # first created became active
    assert reg.active() == a

    # switch active
    assert reg.set_active(b.id) is True
    assert reg.active() == b

    # delete a non-active one — active unaffected, list shrinks
    assert reg.delete(c.id) is True
    assert reg.get(c.id) is None
    assert reg.active() == b
    assert [s.name for s in reg.list()] == ["Alpha", "Beta"]


def test_list_returns_independent_copy(tmp_path: Path) -> None:
    # ``list()`` must not hand callers a mutable view into internal state.
    reg = SpaceRegistry(tmp_path)
    reg.create("one")
    snapshot = reg.list()
    snapshot.clear()
    assert len(reg.list()) == 1


def test_list_returns_space_objects(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    reg.create("x")
    (only,) = reg.list()
    assert isinstance(only, Space)


# ── isolation: real data does not bleed across spaces ─────────────────


def test_data_written_in_a_not_visible_in_b(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    a = reg.create("a")
    b = reg.create("b")

    store_a = _store_for(reg, a.id)
    store_b = _store_for(reg, b.id)

    store_a.save_concept(_concept("only-in-a"))
    store_b.save_concept(_concept("only-in-b"))

    assert _concept_names(store_a) == {"only-in-a"}
    assert _concept_names(store_b) == {"only-in-b"}


def test_isolation_on_disk_paths_are_distinct(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    a = reg.create("a")
    b = reg.create("b")
    pa = reg.path_for(a.id)
    pb = reg.path_for(b.id)
    assert pa != pb
    assert pa == tmp_path / "spaces" / a.id
    assert pb == tmp_path / "spaces" / b.id


def test_isolation_survives_store_reopen(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    a = reg.create("a")
    b = reg.create("b")

    _store_for(reg, a.id).save_concept(_concept("alpha-fact"))

    # Reopen a *fresh* registry + fresh stores over the same root.
    reg2 = SpaceRegistry(tmp_path)
    assert _concept_names(_store_for(reg2, a.id)) == {"alpha-fact"}
    assert _concept_names(_store_for(reg2, b.id)) == set()


def test_deleting_space_with_purge_removes_only_that_space_data(
    tmp_path: Path,
) -> None:
    reg = SpaceRegistry(tmp_path)
    a = reg.create("a")
    b = reg.create("b")

    _store_for(reg, a.id).save_concept(_concept("a-data"))
    _store_for(reg, b.id).save_concept(_concept("b-data"))

    assert reg.delete(a.id, purge_data=True) is True
    assert not (tmp_path / "spaces" / a.id).exists()
    # b is untouched
    assert (tmp_path / "spaces" / b.id).exists()
    assert _concept_names(_store_for(reg, b.id)) == {"b-data"}


# ── legacy / default-space backward compatibility ─────────────────────


def test_legacy_store_exposed_as_default_space(tmp_path: Path) -> None:
    # A pre-space install: real concept data at the top-level root.
    legacy_store = JsonStore(tmp_path)
    legacy_store.save_concept(_concept("legacy-concept"))

    reg = SpaceRegistry(tmp_path)
    spaces = reg.list()
    assert len(spaces) == 1
    default = spaces[0]
    assert default.id == DEFAULT_SPACE_ID
    assert default.name == DEFAULT_SPACE_NAME
    assert reg.active() == default

    # path_for(default) points at the root, so the legacy data is visible.
    assert reg.path_for(default.id) == tmp_path
    reopened = JsonStore(reg.path_for(default.id))
    assert _concept_names(reopened) == {"legacy-concept"}


def test_legacy_default_not_purged_on_delete(tmp_path: Path) -> None:
    legacy_store = JsonStore(tmp_path)
    legacy_store.save_concept(_concept("precious"))

    reg = SpaceRegistry(tmp_path)
    # Even with purge_data, the registry must refuse to rm the root.
    assert reg.delete(DEFAULT_SPACE_ID, purge_data=True) is True
    assert (tmp_path / "concepts").exists()
    assert _concept_names(JsonStore(tmp_path)) == {"precious"}


def test_no_legacy_layout_starts_empty(tmp_path: Path) -> None:
    # No top-level concepts/ → no auto default space.
    reg = SpaceRegistry(tmp_path)
    assert reg.list() == []
    assert reg.active() is None
    assert reg.get(DEFAULT_SPACE_ID) is None


def test_default_path_falls_back_to_subdir_when_created_explicitly(
    tmp_path: Path,
) -> None:
    # When no legacy concepts/ exists and a spaces/default/ dir exists,
    # path_for(default) should resolve to that subdir, not the root.
    sub = tmp_path / "spaces" / DEFAULT_SPACE_ID
    sub.mkdir(parents=True)
    reg = SpaceRegistry(tmp_path)
    assert reg.path_for(DEFAULT_SPACE_ID) == sub


# ── persistence round-trips ───────────────────────────────────────────


def test_active_space_persists_across_registries(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    a = reg.create("a")
    b = reg.create("b")
    reg.set_active(b.id)

    reg2 = SpaceRegistry(tmp_path)
    assert reg2.active() is not None
    assert reg2.active().id == b.id
    assert {s.id for s in reg2.list()} == {a.id, b.id}


def test_space_metadata_persists(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    s = reg.create("Research", description="my research notes")

    reg2 = SpaceRegistry(tmp_path)
    fresh = reg2.get(s.id)
    assert fresh is not None
    assert fresh.name == "Research"
    assert fresh.description == "my research notes"
    assert fresh.id == s.id


def test_delete_persists_across_registries(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    a = reg.create("a")
    b = reg.create("b")
    reg.delete(a.id)

    reg2 = SpaceRegistry(tmp_path)
    assert reg2.get(a.id) is None
    assert {s.id for s in reg2.list()} == {b.id}


def test_registry_file_is_valid_snapshot_json(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    reg.create("a")
    reg.create("b")

    raw = (tmp_path / "spaces.json").read_text(encoding="utf-8")
    snap = SpaceRegistrySnapshot.model_validate_json(raw)
    assert {s.name for s in snap.spaces} == {"a", "b"}
    assert snap.active_space_id is not None


# ── edge / negative cases ─────────────────────────────────────────────


def test_create_empty_name_raises(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    with pytest.raises(ValueError):
        reg.create("")


def test_create_whitespace_name_raises(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    with pytest.raises(ValueError):
        reg.create("   ")


def test_create_strips_surrounding_whitespace(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    s = reg.create("  Padded  ")
    assert s.name == "Padded"


def test_create_duplicate_name_raises(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    reg.create("dup")
    with pytest.raises(ValueError):
        reg.create("dup")


def test_create_duplicate_via_slug_equivalence_raises(tmp_path: Path) -> None:
    # "My Space" and "my-space" slugify to the same id base → collide.
    reg = SpaceRegistry(tmp_path)
    reg.create("My Space")
    with pytest.raises(ValueError):
        reg.create("my-space")


def test_failed_duplicate_create_does_not_grow_list(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    reg.create("only")
    with pytest.raises(ValueError):
        reg.create("only")
    assert len(reg.list()) == 1


def test_set_active_nonexistent_returns_false(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    a = reg.create("a")
    assert reg.set_active("does-not-exist") is False
    # active is unchanged
    assert reg.active() == a


def test_delete_nonexistent_returns_false(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    reg.create("a")
    assert reg.delete("ghost") is False
    assert len(reg.list()) == 1


def test_delete_active_space_reassigns_active(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    a = reg.create("a")
    b = reg.create("b")
    assert reg.active() == a
    assert reg.delete(a.id) is True
    # active falls through to the remaining space
    assert reg.active() is not None
    assert reg.active().id == b.id


def test_delete_last_space_clears_active(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    only = reg.create("solo")
    assert reg.delete(only.id) is True
    assert reg.list() == []
    assert reg.active() is None


def test_delete_active_then_persists_new_active(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    a = reg.create("a")
    b = reg.create("b")
    reg.delete(a.id)

    reg2 = SpaceRegistry(tmp_path)
    assert reg2.active() is not None
    assert reg2.active().id == b.id


# ── slug / id normalization ───────────────────────────────────────────


def test_new_space_id_uses_slug_prefix(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    s = reg.create("My Cool Project!")
    # id is "<slug>-<6 hex>"; slug collapses non-alnum to single dashes.
    assert s.id.startswith("my-cool-project-")
    suffix = s.id.rsplit("-", 1)[-1]
    assert len(suffix) == 6
    assert all(ch in "0123456789abcdef" for ch in suffix)


def test_slugify_normalizes_punctuation_and_case() -> None:
    assert slugify("Hello World") == "hello-world"
    assert slugify("  Mixed__Case!! ") == "mixed-case"
    assert slugify("a---b") == "a-b"


def test_slugify_empty_input_falls_back_to_space() -> None:
    assert slugify("") == "space"
    assert slugify("!!!") == "space"
    assert slugify("   ") == "space"


def test_new_space_id_is_unique_per_call() -> None:
    ids = {new_space_id("same-name") for _ in range(50)}
    assert len(ids) == 50  # random suffix guarantees uniqueness


def test_resolve_matches_by_id_name_and_slug(tmp_path: Path) -> None:
    reg = SpaceRegistry(tmp_path)
    s = reg.create("Field Notes")
    assert reg.resolve(s.id) == s
    assert reg.resolve("Field Notes") == s
    assert reg.resolve("field-notes") == s
    assert reg.resolve("FIELD NOTES") == s  # slug-insensitive
    assert reg.resolve("unrelated") is None


def test_distinct_names_with_same_slug_base_get_distinct_ids() -> None:
    # The slug base may match, but the random suffix keeps ids distinct.
    id1 = new_space_id("Report")
    id2 = new_space_id("report")
    assert id1.startswith("report-")
    assert id2.startswith("report-")
    assert id1 != id2
