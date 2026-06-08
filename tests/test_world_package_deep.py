"""Deep, deterministic tests for the modular ``world`` package.

These exercise the public ``World`` facade END-TO-END (constructor wiring →
pipelines → flush/persistence → status), complementing
``src/world0/world/tests/test_pipelines_with_fakes.py`` which tests the
pipelines in isolation against Protocol doubles.

Everything here is deterministic: ``World(store_path=tmp_path)`` with
``llm=None`` for the structural paths, and a local in-process
``FakeLLM`` returning canned JSON for the ``ingest_text`` extraction path.
No network, no clock dependence in assertions.
"""

from __future__ import annotations

import json

import pytest

from world0 import (
    ConceptCandidate,
    Observation,
    Perspective,
    Projection,
    World,
)
from world0.schemas.types import IngestResult, ReflectResult, WorldStatus


# ── A deterministic LLMProvider for the ingest_text path ──────────────


class FakeLLM:
    """Canned ``LLMProvider`` — returns a fixed JSON blob.

    Satisfies the structural ``LLMProvider`` Protocol (a single
    ``complete_json(system, user)`` method).  Records calls so tests can
    assert the extractor actually invoked it.
    """

    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload if payload is not None else _DEFAULT_PAYLOAD
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return json.dumps(self.payload)


_DEFAULT_PAYLOAD: dict = {
    "domain": "infrastructure",
    "concepts": [
        {
            "uid": "c1",
            "name": "Redis",
            "kind": "entity",
            "domain": "infrastructure",
            "description": "In-memory cache",
        },
        {
            "uid": "c2",
            "name": "latency",
            "kind": "property",
            "domain": "infrastructure",
            "description": "Response time",
        },
    ],
    "relations": [
        {"source": "Redis", "target": "latency", "type": "reduces"},
    ],
}


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / ".world0"


@pytest.fixture
def world(store_path):
    return World(store_path=store_path)


# =====================================================================
# ingest(Observation)
# =====================================================================


class TestIngest:
    def test_new_concepts_reported(self, world):
        result = world.ingest(
            Observation(concepts=["Python", "deployment"], task="t", source="s")
        )
        assert isinstance(result, IngestResult)
        assert sorted(result.new_concepts) == ["Python", "deployment"]
        assert result.reinforced_concepts == []

    def test_reinforced_on_second_ingest(self, world):
        world.ingest(Observation(concepts=["Python"], task="t1"))
        result = world.ingest(Observation(concepts=["Python"], task="t2"))
        assert result.new_concepts == []
        assert result.reinforced_concepts == ["Python"]
        # Reinforcement is observable on the node itself.
        node = world.concepts.resolve("Python")
        assert node.activation_count == 2

    def test_explicit_relation_created(self, world):
        result = world.ingest(
            Observation(
                concepts=["A", "B"],
                relations=[("A", "B", "supports")],
                task="t",
            )
        )
        # "supports" normalizes to the "enables" semantic relation.
        assert result.new_relations == ["A → enables → B"]
        assert len(world.relations) == 1

    def test_hebbian_fires_on_repeated_cooccurrence(self, world):
        # No explicit relation: hebbian only creates an edge once the pair
        # has co-occurred COOCCURRENCE_THRESHOLD (=2) times within one
        # World instance (the co-occurrence counter is engine-local).
        r1 = world.ingest(Observation(concepts=["P", "Q"], task="t"))
        assert r1.hebbian_relations == []
        assert len(world.relations) == 0

        r2 = world.ingest(Observation(concepts=["P", "Q"], task="t"))
        assert r2.hebbian_relations == ["P ↔ Q"]
        assert len(world.relations) == 1

    def test_single_concept_does_not_trigger_hebbian(self, world):
        world.ingest(Observation(concepts=["solo"], task="t"))
        result = world.ingest(Observation(concepts=["solo"], task="t"))
        assert result.hebbian_relations == []
        assert len(world.relations) == 0

    def test_flush_boundary_persists_across_fresh_world(self, store_path):
        # The facade owns the flush boundary: after ingest() returns, a
        # brand-new World over the same store_path must see the data.
        w1 = World(store_path=store_path)
        w1.ingest(
            Observation(
                concepts=["Python", "ML"],
                relations=[("Python", "ML", "supports")],
                source="s",
            )
        )
        py_id = w1.concepts.resolve("Python").id

        w2 = World(store_path=store_path)
        reloaded = w2.concepts.resolve("Python")
        assert reloaded is not None
        assert reloaded.id == py_id
        assert len(w2.relations) == 1


# =====================================================================
# project(seeds, task)
# =====================================================================


class TestProject:
    @pytest.fixture
    def populated(self, store_path):
        w = World(store_path=store_path)
        # Build edges as separate strong pairs so the chain A-B-C has
        # well-defined propagation distances for max_depth testing.
        for _ in range(8):
            w.ingest(
                Observation(
                    concepts=["A", "B"],
                    relations=[("A", "B", "supports")],
                    task="build",
                )
            )
            w.ingest(
                Observation(
                    concepts=["B", "C"],
                    relations=[("B", "C", "supports")],
                    task="build",
                )
            )
        return w

    def test_returns_projection_with_scores_and_relations(self, populated):
        proj = populated.project(["A"], task="build", max_depth=2)
        assert isinstance(proj, Projection)
        names = {c.name for c in proj.concepts}
        assert "A" in names
        # Activation scores keyed by concept id, present for selected nodes.
        assert proj.activation_scores
        for c in proj.concepts:
            assert c.id in proj.activation_scores
        # Relations are internal to the selected concept set.
        ids = {c.id for c in proj.concepts}
        for rel in proj.relations:
            assert rel.source_id in ids and rel.target_id in ids

    def test_empty_seeds_yields_empty_projection(self, populated):
        proj = populated.project([], task="t")
        assert proj.concepts == []
        assert proj.relations == []
        assert proj.activation_scores == {}
        # Task still flows through onto the empty projection.
        assert proj.task == "t"

    def test_unknown_seed_yields_empty_projection(self, populated):
        proj = populated.project(["Nonexistent"], task="t")
        assert proj.concepts == []
        assert proj.activation_scores == {}

    def test_max_depth_respected(self, populated):
        # Depth 1 from A reaches only the direct neighbor B.
        shallow = populated.project(
            ["A"], max_depth=1, max_concepts=50, decay=0.9
        )
        shallow_names = {c.name for c in shallow.concepts}
        assert shallow_names == {"A", "B"}

        # Depth 2 reaches the two-hop neighbor C as well.
        deep = populated.project(
            ["A"], max_depth=2, max_concepts=50, decay=0.9
        )
        deep_names = {c.name for c in deep.concepts}
        assert "C" in deep_names
        # Deeper traversal is a superset of shallower traversal.
        assert shallow_names <= deep_names

    def test_max_concepts_respected(self, populated):
        proj = populated.project(["A"], max_depth=3, max_concepts=2, decay=0.9)
        assert len(proj.concepts) <= 2

    def test_perspective_task_overrides_task_argument(self, populated):
        p = Perspective(name="frame", task="OVERRIDDEN")
        proj = populated.project(["A"], task="ignored", perspective=p)
        assert proj.task == "OVERRIDDEN"

    def test_task_affinity_changes_results(self, store_path):
        # Two disjoint clusters reinforced under different tasks. A seed in
        # one cluster, projected under each task, should differ because the
        # projection engine applies a task-affinity discount.
        w = World(store_path=store_path)
        for _ in range(6):
            w.ingest(
                Observation(
                    concepts=["hub", "ml_a", "ml_b"],
                    relations=[
                        ("hub", "ml_a", "supports"),
                        ("hub", "ml_b", "supports"),
                    ],
                    task="research",
                )
            )
            w.ingest(
                Observation(
                    concepts=["hub", "ops_a", "ops_b"],
                    relations=[
                        ("hub", "ops_a", "supports"),
                        ("hub", "ops_b", "supports"),
                    ],
                    task="operations",
                )
            )

        research_view = w.project(
            ["hub"], task="research", max_concepts=2
        )
        ops_view = w.project(["hub"], task="operations", max_concepts=2)

        research_names = {c.name for c in research_view.concepts}
        ops_names = {c.name for c in ops_view.concepts}
        # The two task frames must not produce identical concept sets.
        assert research_names != ops_names


# =====================================================================
# reflect()
# =====================================================================


class TestReflect:
    def test_returns_reflect_result(self, world):
        world.ingest(Observation(concepts=["X", "Y"], task="t"))
        result = world.reflect()
        assert isinstance(result, ReflectResult)

    def test_persists_last_reflect_in_state(self, store_path):
        w1 = World(store_path=store_path)
        w1.ingest(Observation(concepts=["X"], source="s"))
        assert w1.status().last_reflect is None
        w1.reflect()

        # A fresh World reads last_reflect from state.json.
        w2 = World(store_path=store_path)
        assert w2.status().last_reflect is not None

    def test_state_json_contains_communities_snapshot(self, store_path):
        w = World(store_path=store_path)
        w.ingest(
            Observation(
                concepts=["A", "B", "C"],
                relations=[
                    ("A", "B", "supports"),
                    ("B", "C", "supports"),
                ],
                task="t",
            )
        )
        w.reflect()

        state_file = store_path / "state.json"
        assert state_file.exists()
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert "last_reflect" in state
        assert "communities" in state

    def test_communities_snapshot_reloads(self, store_path):
        # After a reflect, a fresh World rehydrates the community manager
        # from the persisted snapshot without raising.
        w1 = World(store_path=store_path)
        for _ in range(3):
            w1.ingest(
                Observation(
                    concepts=["A", "B", "C", "D"],
                    relations=[
                        ("A", "B", "supports"),
                        ("B", "C", "supports"),
                        ("C", "D", "supports"),
                    ],
                    task="t",
                )
            )
        w1.reflect()
        before = w1.status().total_communities

        w2 = World(store_path=store_path)
        # Reloaded community count matches the snapshot.
        assert w2.status().total_communities == before


# =====================================================================
# Identity operations via the facade
# =====================================================================


class TestIdentityOps:
    def test_merge_absorbs_alias_and_migrates_relations(self, world):
        # Build two concepts plus a relation hanging off the absorbed one.
        world.ingest(
            Observation(
                concepts=["Cat", "Feline", "Mammal"],
                relations=[("Feline", "Mammal", "supports")],
                task="t",
            )
        )
        feline_id = world.concepts.resolve("Feline").id

        ok = world.merge("Cat", "Feline")
        assert ok is True

        # Alias survives: "Feline" still resolves, now to the keeper.
        keeper = world.concepts.resolve("Feline")
        assert keeper is not None
        assert keeper.name == "Cat"
        assert keeper.id != feline_id

        # The Feline→Mammal relation migrated onto the keeper.
        cat_id = world.concepts.resolve("Cat").id
        mammal_id = world.concepts.resolve("Mammal").id
        migrated = world.relations.find_any_between(cat_id, mammal_id)
        assert len(migrated) == 1

    def test_merge_unknown_returns_false(self, world):
        world.ingest(Observation(concepts=["Cat"], task="t"))
        assert world.merge("Cat", "DoesNotExist") is False

    def test_merge_persists_across_fresh_world(self, store_path):
        w1 = World(store_path=store_path)
        w1.ingest(Observation(concepts=["Cat", "Feline"], task="t"))
        assert w1.merge("Cat", "Feline") is True

        w2 = World(store_path=store_path)
        resolved = w2.concepts.resolve("Feline")
        assert resolved is not None
        assert resolved.name == "Cat"

    def test_split_returns_new_id(self, world):
        world.ingest(Observation(concepts=["Bank"], task="t"))
        source_id = world.concepts.resolve("Bank").id

        new_id = world.split("Bank", "Riverbank")
        assert new_id is not None
        assert new_id != source_id
        new_node = world.concepts.get(new_id)
        assert new_node is not None
        assert new_node.name == "Riverbank"

    def test_split_unknown_returns_none(self, world):
        assert world.split("Ghost", "Spectre") is None

    def test_weaken_records_disconfirmation(self, world):
        world.ingest(Observation(concepts=["target"], task="t"))
        before = world.concepts.resolve("target").disconfirmation_count
        ok = world.weaken("target", source="s", task="t")
        assert ok is True
        after = world.concepts.resolve("target").disconfirmation_count
        assert after == before + 1

    def test_weaken_unknown_returns_false(self, world):
        assert world.weaken("nope") is False

    def test_find_similar_returns_ranked_matches(self, world):
        world.ingest(
            Observation(concepts=["Machine Learning", "Machine Vision"], task="t")
        )
        matches = world.find_similar("Machine Learning", min_similarity=0.1)
        assert matches
        # Each entry is (name, score) and scores are descending.
        names = [name for name, _ in matches]
        scores = [score for _, score in matches]
        assert "Machine Learning" in names
        assert scores == sorted(scores, reverse=True)


# =====================================================================
# ingest_text + set_llm
# =====================================================================


class TestIngestText:
    def test_without_llm_raises(self, world):
        with pytest.raises(RuntimeError):
            world.ingest_text("some text", task="t", source="s")

    def test_with_llm_argument_creates_concepts(self, world):
        fake = FakeLLM()
        result = world.ingest_text(
            "We added Redis caching and latency dropped.",
            task="perf",
            source="sess1",
            llm=fake,
        )
        assert fake.calls, "extractor must call the LLM"
        assert sorted(result.new_concepts) == ["Redis", "latency"]
        assert world.concepts.resolve("Redis") is not None
        # The explicit relation from the canned payload is created.
        assert result.new_relations

    def test_constructor_llm_enables_ingest_text(self, store_path):
        w = World(store_path=store_path, llm=FakeLLM())
        result = w.ingest_text("Redis and latency.", task="t")
        assert "Redis" in result.new_concepts

    def test_set_llm_toggles_extraction_capability(self, world):
        # Starts with no extractor → raises.
        with pytest.raises(RuntimeError):
            world.ingest_text("text", task="t")

        # Enable.
        world.set_llm(FakeLLM())
        result = world.ingest_text("Redis and latency.", task="t")
        assert "Redis" in result.new_concepts

        # Disable again.
        world.set_llm(None)
        with pytest.raises(RuntimeError):
            world.ingest_text("text", task="t")


# =====================================================================
# status()
# =====================================================================


class TestStatus:
    def test_empty_world_status(self, world):
        st = world.status()
        assert isinstance(st, WorldStatus)
        assert st.total_concepts == 0
        assert st.total_relations == 0
        assert st.by_maturity == {}
        assert st.avg_confidence == 0.0
        assert st.last_reflect is None

    def test_populated_status_fields(self, store_path):
        w = World(store_path=store_path)
        w.ingest(
            Observation(
                concepts=["A", "B", "C"],
                relations=[
                    ("A", "B", "supports"),
                    ("B", "C", "supports"),
                ],
                task="t",
            )
        )
        st = w.status()
        assert st.total_concepts == 3
        assert st.total_relations == 2
        # Maturity histogram sums to the concept count.
        assert sum(st.by_maturity.values()) == 3
        # Average confidence is a sane probability.
        assert 0.0 <= st.avg_confidence <= 1.0

    def test_status_diagnostic_fields_present_and_sane(self, store_path):
        w = World(store_path=store_path)
        for _ in range(3):
            w.ingest(
                Observation(
                    concepts=["A", "B", "C", "D"],
                    relations=[
                        ("A", "B", "supports"),
                        ("B", "C", "supports"),
                        ("C", "D", "supports"),
                    ],
                    task="t",
                )
            )
        w.reflect()
        st = w.status()
        # Community + color diagnostics.
        assert st.total_communities >= 0
        assert st.stable_communities >= 0
        assert st.bridge_concepts >= 0
        assert 0.0 <= st.avg_color_purity <= 1.0
        # Network-entropy diagnostics.
        assert st.avg_network_entropy >= 0.0
        assert st.high_entropy_concepts >= 0
        assert st.relation_type_entropy >= 0.0


# =====================================================================
# Engine swappability (Lego decoupling through the facade)
# =====================================================================


class _SpyProjectionEngine:
    """Wraps the real ProjectionEngine, counting project() calls."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls = 0

    def project(self, activations, **kwargs):
        self.calls += 1
        return self._inner.project(activations, **kwargs)


class _SpyWorld(World):
    """Overrides the projection engine *after* super().__init__.

    Per facade docs, every public method routes through the attribute,
    never through a direct symbol import — so this override must be used.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._projection = _SpyProjectionEngine(self._projection)


class TestEngineSwappability:
    def test_overridden_projection_engine_is_used(self, store_path):
        w = _SpyWorld(store_path=store_path)
        w.ingest(
            Observation(
                concepts=["A", "B"],
                relations=[("A", "B", "supports")],
                task="t",
            )
        )
        assert w._projection.calls == 0
        proj = w.project(["A"], task="t")
        # The spy was invoked exactly once and returned a real Projection.
        assert w._projection.calls == 1
        assert isinstance(proj, Projection)

    def test_empty_seed_short_circuits_before_engine(self, store_path):
        # When no seed resolves, the facade returns early without ever
        # touching the projection engine.
        w = _SpyWorld(store_path=store_path)
        w.ingest(Observation(concepts=["A"], task="t"))
        proj = w.project(["Unknown"], task="t")
        assert proj.concepts == []
        assert w._projection.calls == 0
