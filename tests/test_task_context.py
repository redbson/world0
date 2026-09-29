"""Task context: distinctive task words and claims seen in other contexts (docs §7.24).

LongRun found that the task label had no measurable effect on World 0:
every benchmark task was ``"<domain> work"``, so word-level matching gave
every concept affinity ≥ 0.5 to every task, and the projection showed all
claims between the concepts in view whatever task they were stated under.
"""

from __future__ import annotations

import math

import pytest

from world0 import Observation, World
from world0.projection.engine import CONTEXT_MATCH
from world0.schemas.concept import TASK_WORD_FLOOR, TaskVocabulary, task_match_score


def _vocab(*labels: str) -> TaskVocabulary:
    v = TaskVocabulary()
    for label in labels:
        v.add(label)
    return v


class TestTaskVocabulary:
    def test_a_word_every_label_carries_weighs_almost_nothing(self):
        v = _vocab("login bug fix", "payment bug fix", "refund bug fix")
        assert v.weight("bug") == pytest.approx(TASK_WORD_FLOOR)
        assert v.weight("login") == pytest.approx(math.log(4 / 2) + TASK_WORD_FLOOR)
        assert v.weight("unseen") == pytest.approx(math.log(4) + TASK_WORD_FLOOR)

    def test_shared_boilerplate_no_longer_matches_another_task(self):
        v = _vocab("login bug fix", "payment bug fix", "refund bug fix")
        assert task_match_score("login bug fix", "payment bug fix") == pytest.approx(2 / 3)
        assert task_match_score("login bug fix", "payment bug fix", v) < 0.05
        assert task_match_score("login bug fix", "login bug fix", v) == 1.0
        assert task_match_score("login", "login bug fix", v) == 1.0

    def test_a_query_of_common_words_still_matches_by_them(self):
        v = _vocab("login bug fix", "payment bug fix")
        assert task_match_score("bug fix", "payment bug fix", v) == 1.0

    def test_counts_are_per_label_holder(self):
        v = _vocab("a work", "a work", "b work")
        assert len(v) == 2
        v.discard("a work")
        assert len(v) == 2 and v.weight("a") > TASK_WORD_FLOOR
        v.discard("a work")
        assert len(v) == 1 and v.weight("work") == pytest.approx(TASK_WORD_FLOOR)


class TestWorldVocabulary:
    @staticmethod
    def _rebuilt(w: World) -> dict:
        ref = TaskVocabulary()
        for node in w.concepts.all():
            for label in node.task_profile:
                ref.add(label)
        return ref._labels

    def test_incremental_vocabulary_equals_the_profiles(self, tmp_path):
        w = World(store_path=tmp_path)
        for i in range(30):
            w.ingest(Observation(concepts=[f"c{i % 7}", f"d{i % 5}"], task=f"t{i % 4} work"))
        assert w.concepts.task_vocabulary._labels == self._rebuilt(w)
        w.clock.advance(5000)
        w.reflect()  # prunes and removes concepts
        assert w.concepts.task_vocabulary._labels == self._rebuilt(w)

    def test_vocabulary_survives_restart_and_merge(self, tmp_path):
        w = World(store_path=tmp_path)
        for i in range(12):
            w.ingest(Observation(concepts=["alpha", "beta", f"g{i}"], task=f"t{i % 3} work"))
        before = dict(w.concepts.task_vocabulary._labels)
        w.close()
        w2 = World(store_path=tmp_path)
        assert w2.concepts.task_vocabulary._labels == before
        a, b = w2.concepts.resolve("alpha"), w2.concepts.resolve("beta")
        w2.concepts.merge(a.id, b.id, w2.relations)
        assert w2.concepts.task_vocabulary._labels == self._rebuilt(w2)


def _polysemous_world(tmp_path) -> World:
    """``pipeline`` in two domains, each with its own claims."""
    w = World(store_path=tmp_path)
    for _ in range(6):
        w.ingest(Observation(
            concepts=["pipeline", "etl job", "warehouse"],
            relations=[("pipeline", "etl job", "depends_on"), ("pipeline", "warehouse", "enables")],
            task="data engineering work",
        ))
        w.ingest(Observation(
            concepts=["pipeline", "optimizer", "checkpoint"],
            relations=[("pipeline", "optimizer", "depends_on"), ("pipeline", "checkpoint", "contains")],
            task="model training work",
        ))
        w.ingest(Observation(concepts=["release notes", "changelog"], task="release work"))
    return w


def _claims(p, rels, about: str | None = "pipeline") -> set[tuple[str, str]]:
    """Explicit claims (optionally only those touching ``about``)."""
    names = {c.id: c.name for c in p.concepts}
    return {
        (names[r.source_id], names[r.target_id])
        for r in rels
        if r.is_explicit and (about is None or about in (names[r.source_id], names[r.target_id]))
    }


class TestContextSplit:
    def test_a_task_shows_the_claims_of_its_own_context(self, tmp_path):
        w = _polysemous_world(tmp_path)
        p = w.project(["pipeline"], task="data engineering work", max_concepts=10)
        assert {c.name for c in p.concepts} >= {"etl job", "warehouse", "optimizer", "checkpoint"}
        assert _claims(p, p.relations) == {("pipeline", "etl job"), ("pipeline", "warehouse")}
        assert _claims(p, p.other_contexts) == {("pipeline", "optimizer"), ("pipeline", "checkpoint")}
        other = w.project(["pipeline"], task="model training work", max_concepts=10)
        assert _claims(other, other.relations) == {("pipeline", "optimizer"), ("pipeline", "checkpoint")}

    @pytest.mark.parametrize("task", ["", "incident review", "release work"])
    def test_without_a_matching_context_nothing_moves(self, tmp_path, task):
        """No task, a new task, or a task the concept never served under:
        there is no evidence that any claim belongs elsewhere."""
        w = _polysemous_world(tmp_path)
        p = w.project(["pipeline"], task=task, max_concepts=10)
        assert p.other_contexts == []
        assert len(_claims(p, p.relations)) == 4

    def test_claims_of_a_concept_with_no_claims_in_context_stay(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(5):
            w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "depends_on")], task="alpha work"))
            w.ingest(Observation(concepts=["b", "c"], relations=[("b", "c", "depends_on")], task="beta work"))
        # Under "beta work", b has an in-context claim (b→c), so its alpha
        # claim moves; under a task nobody used, nothing moves.
        p = w.project(["a", "b"], task="beta work", max_concepts=5)
        assert _claims(p, p.other_contexts, about=None) == {("a", "b")}
        q = w.project(["a"], task="gamma work", max_concepts=5)
        assert q.other_contexts == []

    def test_untagged_claims_are_neutral(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(4):
            w.ingest(Observation(concepts=["x", "y"], relations=[("x", "y", "depends_on")]))
            w.ingest(Observation(concepts=["x", "z"], relations=[("x", "z", "depends_on")], task="zeta work"))
        p = w.project(["x"], task="zeta work", max_concepts=5)
        assert _claims(p, p.relations, about="x") >= {("x", "y"), ("x", "z")}
        assert p.other_contexts == []

    def test_a_co_mention_does_not_move_an_untagged_claim(self, tmp_path):
        """Context is decided by statements, not by co-occurrence provenance."""
        w = World(store_path=tmp_path)
        w.ingest(Observation(concepts=["api", "db"], relations=[("api", "db", "depends_on")]))
        w.ingest(Observation(concepts=["api", "db"], task="search work"))
        for _ in range(3):
            w.ingest(Observation(concepts=["api", "q"], relations=[("api", "q", "depends_on")], task="billing work"))
        p = w.project(["api"], task="billing work", max_concepts=6)
        assert ("api", "db") in _claims(p, p.relations, about="api")

    def test_a_withdrawn_claim_does_not_put_a_concept_in_context(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(3):
            w.ingest(Observation(concepts=["pipe", "etl"], relations=[("pipe", "etl", "depends_on")], task="data work"))
            w.ingest(Observation(concepts=["pipe", "opt"], relations=[("pipe", "opt", "depends_on")], task="train work"))
        w.ingest(Observation(concepts=["pipe"], retracted_relations=[("pipe", "etl", "depends_on")], task="data work"))
        p = w.project(["pipe"], task="data work", max_concepts=6)
        assert p.other_contexts == []
        assert ("pipe", "opt") in _claims(p, p.relations, about="pipe")

    def test_a_contested_pair_stays_contested_across_contexts(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(3):
            w.ingest(Observation(concepts=["cache", "db"], relations=[("cache", "db", "enables")], task="alpha work"))
        for _ in range(4):
            w.ingest(Observation(concepts=["cache", "db"], relations=[("cache", "db", "conflict")], task="beta work"))
        free = w.project(["cache", "db"], max_concepts=4)
        scoped = w.project(["cache", "db"], task="alpha work", max_concepts=4)
        assert len(free.epistemic.contested) == 1
        assert len(scoped.epistemic.contested) == 1

    def test_render_lists_the_other_contexts(self, tmp_path):
        w = _polysemous_world(tmp_path)
        text = w.project(["pipeline"], task="data engineering work", max_concepts=10).render()
        assert "### Seen in Other Tasks" in text
        assert "model training work" in text

    def test_threshold_is_a_majority_of_the_task_information(self):
        assert CONTEXT_MATCH == 0.5
