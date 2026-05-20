"""Pipelines — testable with only Protocol fakes.

The IngestPipeline and ReflectPipeline never import a concrete engine;
they accept anything satisfying the relevant Protocol.  We exercise
them here against the doubles in ``world0.core.test_doubles`` to prove
the wiring is fully Lego-decoupled.
"""

from __future__ import annotations

from world0.core.test_doubles import (
    FakeColorField,
    FakeConceptStore,
    FakeDecayPolicy,
    FakeHebbianLearner,
    FakeLifecyclePolicy,
    FakeRelationStore,
)
from world0.schemas.types import Observation
from world0.world._ingest import IngestPipeline
from world0.world._reflect import ReflectPipeline


# ── IngestPipeline ───────────────────────────────────────────────────


def _make_ingest_pipeline() -> tuple[
    IngestPipeline,
    FakeConceptStore,
    FakeRelationStore,
    FakeHebbianLearner,
    FakeColorField,
]:
    cs = FakeConceptStore()
    rs = FakeRelationStore()
    hebbian = FakeHebbianLearner()
    color = FakeColorField()
    pipeline = IngestPipeline(
        concepts=cs, relations=rs, hebbian=hebbian, color=color
    )
    return pipeline, cs, rs, hebbian, color


def test_ingest_creates_concepts_and_records_them_as_new() -> None:
    pipeline, cs, _, _, _ = _make_ingest_pipeline()
    obs = Observation(
        concepts=["Python", "deployment"],
        task="t",
        source="s",
    )
    result = pipeline.run(obs)
    assert sorted(result.new_concepts) == ["Python", "deployment"]
    assert result.reinforced_concepts == []
    assert len(cs) == 2


def test_ingest_reinforces_existing_concept() -> None:
    pipeline, cs, _, _, _ = _make_ingest_pipeline()
    pipeline.run(Observation(concepts=["Python"], task="t1"))
    result = pipeline.run(Observation(concepts=["Python"], task="t2"))
    assert result.new_concepts == []
    assert result.reinforced_concepts == ["Python"]


def test_ingest_creates_explicit_relations() -> None:
    pipeline, _, rs, _, _ = _make_ingest_pipeline()
    obs = Observation(
        concepts=["A", "B"],
        relations=[("A", "B", "depends_on")],
        task="t",
    )
    result = pipeline.run(obs)
    assert result.new_relations == ["A → depends_on → B"]
    assert len(rs) == 1


def test_ingest_invokes_hebbian_with_resolved_ids() -> None:
    pipeline, cs, _, hebbian, _ = _make_ingest_pipeline()
    pipeline.run(Observation(concepts=["a", "b", "c"], task="t"))
    assert hebbian.calls, "hebbian.learn should have been called"
    seen_ids, prov = hebbian.calls[0]
    assert prov == "t"
    assert len(seen_ids) == 3
    # Every id must exist in the concept store.
    for cid in seen_ids:
        assert cs.get(cid) is not None


def test_ingest_invokes_color_seed() -> None:
    pipeline, _, _, _, color = _make_ingest_pipeline()
    pipeline.run(
        Observation(concepts=["a", "b"], task="optimize", domain="ml")
    )
    assert len(color.seed_and_diffuse_calls) == 1
    ids, label = color.seed_and_diffuse_calls[0]
    assert label == "ml"
    assert len(ids) == 2


def test_ingest_handles_disconfirmation() -> None:
    pipeline, cs, _, _, _ = _make_ingest_pipeline()
    pipeline.run(Observation(concepts=["target"], task="t"))
    result = pipeline.run(Observation(weakened=["target"], task="t"))
    assert result.weakened_concepts == ["target"]


# ── ReflectPipeline ──────────────────────────────────────────────────


class _FakeCommunityManager:
    """Minimal CommunityManager stand-in for ReflectPipeline tests."""

    class _Update:
        def __init__(self) -> None:
            self.new: list[str] = ["c-new"]
            self.pruned: list[str] = ["c-old"]
            self.color_sources: list[str] = ["c-src"]

    def detect_and_update(self) -> "_FakeCommunityManager._Update":
        return self._Update()

    def all(self) -> list:
        return []

    def color_sources(self) -> list:
        return []


def test_reflect_runs_all_five_stages_in_order() -> None:
    decay = FakeDecayPolicy(
        decayed_concepts=["dc1"],
        decayed_relations=["dr1"],
        pruned_concepts=["pc1"],
        pruned_relations=["pr1"],
    )
    lifecycle = FakeLifecyclePolicy(promoted=["p1"], demoted=["d1"])
    color = FakeColorField()
    communities = _FakeCommunityManager()

    pipeline = ReflectPipeline(
        decay=decay, lifecycle=lifecycle, color=color, communities=communities
    )
    result = pipeline.run()

    # 1. Decay was called first (both concepts and relations).
    assert decay.calls[0] == "decay_concepts"
    assert decay.calls[1] == "decay_relations"
    assert result.decayed_concepts == ["dc1"]
    assert result.decayed_relations == ["dr1"]

    # 2. Community state propagated.
    assert result.new_communities == ["c-new"]
    assert result.pruned_communities == ["c-old"]
    assert result.color_sources == ["c-src"]

    # 3. Color field: fade ran, then seed_from_communities, then settle.
    assert color.fade_calls == 1
    assert len(color.seed_from_communities_calls) == 1
    assert color.settle_calls == 1

    # 4. Lifecycle was evaluated exactly once.
    assert lifecycle.calls == 1
    assert result.promoted_concepts == ["p1"]
    assert result.demoted_concepts == ["d1"]

    # 5. Pruning ran last.
    assert decay.calls[-2:] == ["prune_relations", "prune_concepts"]
    assert result.pruned_concepts == ["pc1"]
    assert result.pruned_relations == ["pr1"]
