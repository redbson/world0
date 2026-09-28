"""Sustained focus and attention schema (docs/mc/03-workspace.md).

Baseline probe: projections were stateless — after two Ops-focused views
the bridge concept ``deployment`` gave exactly its cold-start view
(1 Ops / 3 ML), and a view carried no record of why a concept was in it.
With ``World(sustained_attention=True)`` a limited-capacity focus fed by
ignition biases the next view (GWT-4 / GWT-2), releases on a task switch
and fades over unrelated views; every view carries an attention trace
(AST-1).
"""

from __future__ import annotations

import pytest

from world0 import Observation, World
from world0.context import FOCUS_CAPACITY, Focus

OPS = ["deployment", "kubernetes", "helm chart", "container registry", "rollout", "load balancer"]
ML = ["deployment", "pytorch", "gpu cluster", "training run", "checkpoint", "loss curve"]
OTHER = ["sourdough", "starter", "hydration", "proofing", "crumb"]


def _build(path, sustained: bool) -> World:
    w = World(store_path=path, sustained_attention=sustained)
    for _ in range(12):
        w.ingest(Observation(concepts=OPS, relations=[("deployment", "kubernetes", "depends_on")], source="s"))
        w.ingest(Observation(concepts=ML, relations=[("deployment", "pytorch", "depends_on")], source="s"))
        w.ingest(Observation(concepts=OTHER, source="s"))
    w.reflect()
    return w


def _view(w: World, seeds, task: str = "", k: int = 5) -> list[str]:
    return [c.name for c in w.project(seeds, task=task, max_concepts=k).concepts]


def _ops(names: list[str]) -> int:
    return sum(n in OPS for n in names[1:])


@pytest.fixture
def focused(tmp_path):
    return _build(tmp_path / "f", True)


@pytest.fixture
def plain(tmp_path):
    return _build(tmp_path / "p", False)


class TestStateDependentAttention:
    def test_default_is_stateless(self, plain):
        cold = _view(plain, ["deployment"])
        _view(plain, ["kubernetes", "rollout"])
        assert _view(plain, ["deployment"]) == cold
        assert len(plain.focus) == 0

    def test_ops_focus_steers_bridge_concept(self, focused, plain):
        cold = _view(plain, ["deployment"])
        _view(focused, ["kubernetes", "rollout"])
        _view(focused, ["helm chart", "load balancer"])
        after = _view(focused, ["deployment"])
        assert _ops(after) >= 3 > _ops(cold)

    def test_no_perseveration_on_unrelated_seeds(self, focused, plain):
        _view(focused, ["kubernetes", "rollout"])
        assert _view(focused, ["sourdough"]) == _view(plain, ["sourdough"])

    def test_focus_fades_over_unrelated_views(self, focused, plain):
        _view(focused, ["kubernetes", "rollout"])
        for _ in range(6):
            _view(focused, ["sourdough"])
        assert not any(n in OPS for n in _names(focused))
        assert _view(focused, ["deployment"]) == _view(plain, ["deployment"])

    def test_task_switch_releases(self, focused, plain):
        _view(focused, ["kubernetes", "rollout"], task="cluster ops")
        assert len(focused.focus) > 0 and focused.focus.task == "cluster ops"
        switched = _view(focused, ["deployment"], task="model training")
        assert switched == _view(plain, ["deployment"], task="model training")

    def test_capacity_is_bounded(self, focused):
        for seeds in (["kubernetes"], ["pytorch"], ["sourdough"], ["rollout"], ["checkpoint"], ["crumb"]):
            _view(focused, seeds, k=10)
            assert len(focused.focus) <= FOCUS_CAPACITY


def _names(w: World) -> set[str]:
    by_id = {c.id: c.name for c in w.concepts.all()}
    return {by_id[cid] for cid in w.focus.items()}


class TestIgnition:
    def test_seeds_ignite_and_enter_at_full_strength(self, focused):
        p = focused.project(["kubernetes", "rollout"], max_concepts=5)
        seeds = {c.id for c in p.concepts if c.name in ("kubernetes", "rollout")}
        assert seeds <= set(p.ignited_ids())
        items = focused.focus.items()
        assert all(items[cid] == 1.0 for cid in p.ignited_ids())

    def test_ignition_is_selective(self, focused):
        p = focused.project(["deployment"], max_concepts=10)
        reached = [t for t in p.attention.values() if t.kind == "reached"]
        assert any(t.ignited for t in reached) and not all(t.ignited for t in reached)


class TestAttentionSchema:
    def test_trace_for_every_selected_concept(self, plain):
        p = plain.project(["deployment"], task="helm upgrade", max_concepts=6)
        assert set(p.attention) == {c.id for c in p.concepts}
        by_name = {c.name: p.attention[c.id] for c in p.concepts}
        assert by_name["deployment"].kind == "seed"
        ids = {c.id for c in plain.concepts.all()}
        for name, trace in by_name.items():
            if trace.kind == "reached":
                assert trace.via in ids and trace.relation
        if "helm chart" in by_name:
            assert by_name["helm chart"].task_named

    def test_in_focus_vs_next_to_focus(self, focused):
        _view(focused, ["kubernetes", "rollout"])
        p = focused.project(["deployment"], max_concepts=5)
        by_name = {c.name: p.attention[c.id] for c in p.concepts}
        assert by_name["kubernetes"].in_focus and by_name["kubernetes"].sustained
        assert not by_name["pytorch"].sustained  # bridge seed passes no focus to its ML side
        text = p.render()
        assert "### Why These Concepts" in text and "still in focus" in text


class TestFocusUnit:
    def test_retention_and_min_strength(self):
        f = Focus()
        f.update(["a"])
        strengths = []
        for _ in range(5):
            f.update([])
            strengths.append(f.items().get("a", 0.0))
        assert strengths[0] > strengths[1] > strengths[2] > 0.0
        assert strengths[-1] == 0.0

    def test_seed_exclusion(self):
        f = Focus()
        f.update(["bridge"])
        nbrs = {"x": {"bridge"}, "y": {"bridge"}}
        assert f.affinity(["x", "y"], nbrs) == {"x": 0.5, "y": 0.5}
        assert f.affinity(["x", "y"], nbrs, exclude=["bridge"]) == {}

    def test_clear(self):
        f = Focus()
        f.update(["a"], task="t")
        f.clear()
        assert len(f) == 0 and f.task == ""
