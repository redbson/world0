"""Relation identity under label noise (round 25).

LongRun, one extraction error type at a time (p = 0.3): a wrong label cost
World 0 the most (0.79 → 0.52).  A second label on the same (source,
target, axis) overwrote the first, so one mislabelled "contains" replaced
a "depends on" stated many times (and a revision that makes both true lost
one); and an explicit parallel claim on a co-occurrence edge relabelled it
but left it a co-occurrence edge.
"""

from __future__ import annotations

from world0 import Observation, World


def _edge(w: World, a: str, b: str):
    ia, ib = w.concepts.resolve(a).id, w.concepts.resolve(b).id
    return next(e for e in w.relations.all() if {e.source_id, e.target_id} == {ia, ib} and e.is_explicit)


def _claims(w: World, a: str, b: str) -> dict[str, int]:
    ia, ib = w.concepts.resolve(a).id, w.concepts.resolve(b).id
    return {e.semantic_relation: e.support for e in w.relations.all()
            if (e.source_id, e.target_id) == (ia, ib) and e.is_explicit}


class TestLabelIdentity:
    """A claim is (source, target, label): another label is another claim."""

    def test_a_second_label_does_not_overwrite_the_first(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(3):
            w.ingest(Observation(concepts=["api", "db"], relations=[("api", "db", "depends_on")]))
        w.ingest(Observation(concepts=["api", "db"], relations=[("api", "db", "contains")]))
        assert _claims(w, "api", "db") == {"dependence": 3, "inclusion": 1}

    def test_two_well_stated_labels_are_both_current(self, tmp_path):
        """A revision can make "A contains B" and "A depends on B" both true."""
        w = World(store_path=tmp_path)
        for _ in range(2):
            w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "contains")]))
            w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "depends_on")]))
        text = w.project(["a"], max_concepts=3).render()
        assert "- a contains b (belief" in text and "- a depends on b (belief" in text
        assert "also stated as" not in text

    def test_a_rarely_stated_label_is_held_loosely(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(3):
            w.ingest(Observation(concepts=["api", "db"], relations=[("api", "db", "depends_on")]))
        w.ingest(Observation(concepts=["api", "db"], relations=[("api", "db", "contains")]))
        text = w.project(["api"], max_concepts=3).render()
        assert "- api depends on db (belief" in text
        assert "- api contains db (belief" not in text
        assert "- also stated as: api contains db (1 vs 3 statements)" in text

    def test_withdrawing_one_label_leaves_the_other(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(2):
            w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "contains"), ("a", "b", "depends_on")]))
        w.ingest(Observation(concepts=["a"], retracted_relations=[("a", "b", "contains")]))
        live = {e.semantic_relation for e in w.relations.all() if e.is_explicit and not e.is_retracted}
        assert live == {"dependence"}

    def test_both_labels_survive_restart(self, tmp_path):
        w = World(store_path=tmp_path)
        w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "depends_on")]))
        w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "enables")]))
        w.close()
        assert _claims(World(store_path=tmp_path), "a", "b") == {"dependence": 1, "enables": 1}

    def test_merging_twins_merges_same_labels_only(self, tmp_path):
        w = World(store_path=tmp_path)
        w.ingest(Observation(concepts=["auth svc", "db"], relations=[("auth svc", "db", "contains")]))
        w.ingest(Observation(concepts=["auth svc", "db"], relations=[("auth svc", "db", "depends_on")]))
        for _ in range(2):
            w.ingest(Observation(concepts=["auth service", "db"], relations=[("auth service", "db", "depends_on")]))
        a, b = w.concepts.resolve("auth svc"), w.concepts.resolve("auth service")
        w.concepts.merge(a.id, b.id, w.relations)
        labels = sorted(e.semantic_relation for e in w.relations.for_concept(a.id) if e.is_explicit)
        assert labels == ["dependence", "inclusion"]


class TestClaimOnCooccurrence:
    def test_a_stated_claim_on_a_cooccurrence_edge_becomes_a_claim(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(4):
            w.ingest(Observation(concepts=["a", "b"]))
        r = w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "similar_to")]))
        e = _edge(w, "a", "b")
        assert e.semantic_relation == "similarity_kernel" and e.is_explicit
        assert e.probability >= 0.64 and e.support == 1
        assert r.new_relations == ["a → similarity_kernel → b"]
        assert "- a is similar to b (belief" in w.project(["a"], max_concepts=3).render()

    def test_cooccurrence_never_relabels_a_claim(self, tmp_path):
        w = World(store_path=tmp_path)
        w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "similar_to")]))
        for _ in range(6):
            w.ingest(Observation(concepts=["a", "b"]))
        assert _edge(w, "a", "b").semantic_relation == "similarity_kernel"


class TestReviewFindings:
    """Regressions for the independent review of round 25."""

    def test_a_mislabel_does_not_hide_a_real_contest(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(2):
            w.ingest(Observation(concepts=["api", "db"], relations=[("api", "db", "depends_on")]))
        for _ in range(2):
            w.ingest(Observation(concepts=["api", "db"], relations=[("api", "db", "conflict")]))
        w.ingest(Observation(concepts=["api", "db"], relations=[("api", "db", "contains")]))
        loose = w.project(["api"], max_concepts=3).render().split("Hold loosely:")[1]
        assert "contested: " in loose and "conflicts with db" in loose and "depends on db" in loose

    def test_merging_adds_statements(self, tmp_path):
        w = World(store_path=tmp_path)
        w.ingest(Observation(concepts=["auth svc", "db"], relations=[("auth svc", "db", "depends_on")]))
        w.ingest(Observation(concepts=["auth service", "db"], relations=[("auth service", "db", "depends_on")]))
        a, b = w.concepts.resolve("auth svc"), w.concepts.resolve("auth service")
        w.concepts.merge(a.id, b.id, w.relations)
        (edge,) = [e for e in w.relations.for_concept(a.id) if e.is_explicit]
        assert edge.support == 2

    def test_a_claim_merged_into_its_cooccurrence_twin_stays_a_claim(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(4):
            w.ingest(Observation(concepts=["svc b", "db"]))
        for _ in range(2):
            w.ingest(Observation(concepts=["svc a", "db"], relations=[("svc a", "db", "related_to")]))
        a, b = w.concepts.resolve("svc a"), w.concepts.resolve("svc b")
        w.concepts.merge(a.id, b.id, w.relations)
        edges = [e for e in w.relations.for_concept(a.id) if w.concepts.get(e.other_end(a.id)).name == "db"]
        assert len(edges) == 1 and edges[0].is_explicit and edges[0].support == 2

    def test_a_claim_on_a_cooccurrence_edge_does_not_depend_on_what_came_first(self, tmp_path):
        from world0.schemas.types import RelationPrior

        states = []
        for i, cooccur in enumerate((0, 6)):
            w = World(store_path=tmp_path / str(i))
            for _ in range(cooccur):
                w.ingest(Observation(concepts=["a", "b"]))
            w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "equivalence")],
                                 relation_priors=[RelationPrior(source="a", target="b",
                                                                relation_type="equivalence", probability=0.3)]))
            e = _edge(w, "a", "b")
            states.append((round(e.probability, 6), e.belief_prior, e.support,
                           len([r for r in w.relations.all() if r.relation_type == e.relation_type])))
        assert states[0] == states[1]

    def test_restatements_with_a_prior_add_support(self, tmp_path):
        from world0.schemas.types import RelationPrior

        w = World(store_path=tmp_path)
        for _ in range(3):
            w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "depends_on")],
                                 relation_priors=[RelationPrior(source="a", target="b",
                                                                relation_type="depends_on", probability=0.8)]))
        assert _edge(w, "a", "b").support == 3

    def test_an_undirected_minority_label_is_found_either_way(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(4):
            w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "similar_to")]))
        w.ingest(Observation(concepts=["a", "b"], relations=[("b", "a", "equivalence")]))
        text = w.project(["a"], max_concepts=3).render()
        assert "equivalent" not in text.split("Hold loosely:")[0]

    def test_withdrawal_never_takes_a_cooccurrence_edge(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(6):
            w.ingest(Observation(concepts=["a", "b"]))
        r = w.ingest(Observation(concepts=["a"], retracted_relations=[("a", "b", "equivalence")]))
        assert r.retracted_relations == [] and not any(e.is_retracted for e in w.relations.all())

    def test_a_paraphrased_withdrawal_takes_the_only_claim(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(3):
            w.ingest(Observation(concepts=["api", "db"], relations=[("api", "db", "supports")]))
        w.ingest(Observation(concepts=["api"], retracted_relations=[("api", "db", "depends_on")]))
        assert _edge(w, "api", "db").is_retracted

    def test_two_labels_are_one_activation_channel(self, tmp_path):
        from world0.dynamics.activation import ActivationEngine

        scores = []
        for i, extra in enumerate((("api", "db", "depends_on"), ("api", "db", "contains"))):
            w = World(store_path=tmp_path / str(i))
            for _ in range(3):
                w.ingest(Observation(concepts=["api", "db"], relations=[("api", "db", "depends_on")]))
            w.ingest(Observation(concepts=["api", "db"], relations=[extra]))
            api, db = w.concepts.resolve("api"), w.concepts.resolve("db")
            out = ActivationEngine(w.concepts, w.relations, clock=w.clock).activate([api.id], record=False)
            scores.append(out[db.id])
        assert scores[1] <= scores[0] * 1.2  # a misread label does not double the signal

    def test_core_counts_neighbours_not_claims(self, tmp_path):
        w = World(store_path=tmp_path)
        w.ingest(Observation(concepts=["api", "db"],
                             relations=[("api", "db", "depends_on"), ("api", "db", "contains"), ("api", "db", "enables")]))
        assert w._lifecycle._connections(w.concepts.resolve("api")) == 1

    def test_cooccurrence_keeps_a_withdrawn_claim_as_history(self, tmp_path):
        w = World(store_path=tmp_path)
        w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "similar_to")]))
        w.ingest(Observation(concepts=["a"], retracted_relations=[("a", "b", "similar_to")]))
        for _ in range(3):
            w.ingest(Observation(concepts=["a", "b"]))
        assert any(e.is_explicit and e.is_retracted for e in w.relations.all())
