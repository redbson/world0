"""Hard retraction and read-time settlement (docs §7.25).

LongRun: a correction ("X no longer depends on Y") only weakened the old
claim, so it stayed in every view (strict stale ≈ 0, and real readers named
the stale claim in 100 % of cases).  A claim that no longer holds is now
*withdrawn*: kept as history, out of the live world.  Separately, reads
between events used the stored confidence, so a view depended on when a
reflect last settled it.
"""

from __future__ import annotations

import pytest

from world0 import Observation, World
from world0.dynamics.activation import ActivationEngine
from world0.dynamics.decay import settle_concept, settled_confidence
from world0.schemas.concept import Maturity


def _names(p, rels):
    names = {**p.outside_names, **{c.id: c.name for c in p.concepts}}
    return {(names[r.source_id], names[r.target_id]) for r in rels if r.is_explicit}


def _revised_world(tmp_path) -> World:
    w = World(store_path=tmp_path)
    for _ in range(5):
        w.ingest(Observation(concepts=["api", "db"], relations=[("api", "db", "depends_on")]))
    w.ingest(Observation(
        concepts=["api", "cache"],
        relations=[("api", "cache", "depends_on")],
        retracted_relations=[("api", "db", "depends_on")],
    ))
    return w


class TestRetraction:
    def test_a_withdrawn_claim_leaves_the_view_and_is_reported(self, tmp_path):
        w = _revised_world(tmp_path)
        p = w.project(["api"], max_concepts=5)
        assert ("api", "cache") in _names(p, p.relations)
        assert ("api", "db") not in _names(p, p.relations)
        assert _names(p, p.retracted) == {("api", "db")}
        assert "### No Longer Holds" in p.render()

    def test_the_ingest_result_reports_it(self, tmp_path):
        w = World(store_path=tmp_path)
        w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "depends_on")]))
        r = w.ingest(Observation(concepts=["a"], retracted_relations=[("a", "b", "depends_on")]))
        assert r.retracted_relations == ["a → dependence → b"]
        again = w.ingest(Observation(concepts=["a"], retracted_relations=[("a", "b", "depends_on")]))
        assert again.retracted_relations == []

    def test_retraction_is_not_disconfirmation(self, tmp_path):
        """The claim held; it no longer does.  Belief is untouched."""
        w = World(store_path=tmp_path)
        w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "depends_on")]))
        edge = w.relations.all()[0]
        p = edge.probability
        w.ingest(Observation(concepts=["a"], retracted_relations=[("a", "b", "depends_on")]))
        assert edge.is_retracted and edge.probability == p and edge.disconfirmation_count == 0

    def test_a_withdrawn_claim_carries_no_activation(self, tmp_path):
        w = _revised_world(tmp_path)
        api, db = w.concepts.resolve("api"), w.concepts.resolve("db")
        out = ActivationEngine(w.concepts, w.relations, clock=w.clock).activate([api.id], record=False)
        assert db.id not in out

    def test_restating_restores_the_claim(self, tmp_path):
        w = _revised_world(tmp_path)
        w.ingest(Observation(concepts=["api", "db"], relations=[("api", "db", "depends_on")]))
        p = w.project(["api"], max_concepts=5)
        assert ("api", "db") in _names(p, p.relations) and p.retracted == []

    def test_a_withdrawn_claim_is_forgotten_and_not_a_connection(self, tmp_path):
        w = _revised_world(tmp_path)
        edge = next(e for e in w.relations.all() if e.is_retracted)
        api = w.concepts.resolve("api")
        assert all(e.id != edge.id for e in w.relations.for_concept(api.id) if not e.is_retracted)
        assert w._lifecycle._connections(api) == 1  # api → cache only
        w.clock.advance(3000)
        w.reflect()
        assert w.relations.get(edge.id) is None  # relaxed to zero, pruned

    def test_retracting_an_unknown_claim_changes_nothing(self, tmp_path):
        w = World(store_path=tmp_path)
        w.ingest(Observation(concepts=["a", "b"]))
        r = w.ingest(Observation(concepts=["a"], retracted_relations=[("a", "b", "depends_on")]))
        assert r.retracted_relations == [] and r.weakened_concepts == []

    def test_retraction_survives_restart(self, tmp_path):
        w = _revised_world(tmp_path)
        w.close()
        w2 = World(store_path=tmp_path)
        assert sum(e.is_retracted for e in w2.relations.all()) == 1


class TestReadTimeSettlement:
    def test_settled_confidence_is_what_settle_would_leave(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(4):
            w.ingest(Observation(concepts=["x"]))
        node = w.concepts.resolve("x")
        w.clock.advance(400)
        read = settled_confidence(node, w.clock.tick)
        stored = node.confidence
        assert read < stored  # nothing mutated yet
        settle_concept(node, w.clock.tick)
        assert node.confidence == pytest.approx(read, abs=1e-12)

    def test_a_view_between_uses_does_not_depend_on_reflect(self, tmp_path):
        scores = []
        for i, reflect in enumerate((False, True)):
            w = World(store_path=tmp_path / str(i))
            for _ in range(6):
                w.ingest(Observation(concepts=["a", "b", "c"], relations=[("a", "b", "depends_on")]))
            w.clock.advance(300)
            if reflect:
                w.reflect(light=True)
            a = w.concepts.resolve("a")
            out = ActivationEngine(w.concepts, w.relations, clock=w.clock).activate([a.id], record=False)
            scores.append({w.concepts.get(k).name: v for k, v in out.items()})
        assert scores[0].keys() == scores[1].keys()
        for name in scores[0]:
            assert scores[0][name] == pytest.approx(scores[1][name], rel=1e-6)

    def test_reading_does_not_mutate(self, tmp_path):
        w = World(store_path=tmp_path)
        w.ingest(Observation(concepts=["x"]))
        node = w.concepts.resolve("x")
        w.clock.advance(200)
        before = (node.confidence, node.maturity, node.last_decayed_tick)
        settled_confidence(node, w.clock.tick)
        w.project(["x"])
        assert (node.confidence, node.maturity, node.last_decayed_tick) == before
        assert before[1] != Maturity.FADING  # the settled value would be fading
