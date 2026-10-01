"""A stated negative claim carries a belief, not an inhibition gain.

Before: ``RelationManager.discover`` seeded an explicit edge's
``probability`` (the belief that the claim is true) from
``SemanticRelationSpec.propagation_strength``.  On the negative axis that
number is the gain of the *inhibition* channel (0.05-0.12), so a stated
conflict started with a seventh of the belief of a stated dependence, and
its relation floor (``RELATION_FLOOR_SHARE x probability``) sat below the
0.02 prune threshold from the first tick: with both concepts kept alive
and the claim never restated it was pruned after 450 idle observations
(``conflict``), 350 (``disjointness``) against 8 100 for ``depends_on``.
A contested pair (``enables`` vs ``conflict``) started 0.76 vs 0.10, so
"enables x10 then conflict x10" only looked balanced (0.45 vs 0.43).

Now the default belief of a negative-axis claim is ``NEGATIVE_CLAIM_PRIOR``
(0.70, the belief a once-stated ``dependence`` carries) while the
inhibition gain (``weight``) is untouched; positive and parallel claims are
seeded exactly as before; stores written earlier are rebased once on load.

The survival probe drives the real ``World.ingest`` and ``DecayEngine`` the
same way ``docs/paper/verify.py`` §4.2 does: both concepts are kept alive,
the clock advances 50 ticks at a time, the relation is never restated.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from world0 import Observation, RelationPrior, World
from world0.dynamics.activation import ActivationEngine
from world0.dynamics.decay import DecayEngine, relation_floor
from world0.projection.metacognition import CONTEST_MARGIN, assess
from world0.relations.manager import RelationManager
from world0.schemas.relation import (
    NEGATIVE_CLAIM_PRIOR,
    SEMANTIC_RELATION_SPECS,
    RelationEdge,
    RelationType,
    semantic_relation_names,
)
from world0.store.json_store import JsonStore

NEGATIVE = semantic_relation_names("negative")
NON_NEGATIVE = semantic_relation_names("positive") + semantic_relation_names("parallel")
STEP = 50


# ── helpers ────────────────────────────────────────────────────────────


def _state(w: World, src: str, tgt: str, rel: str, times: int = 1, prior: float | None = None):
    for _ in range(times):
        kw = {}
        if prior is not None:
            kw["relation_priors"] = [
                RelationPrior(source=src, target=tgt, relation_type=rel, probability=prior)
            ]
        w.ingest(Observation(concepts=[src, tgt], relations=[(src, tgt, rel)], source="s", **kw))


def _claims(w: World, a: str = "a", b: str = "b") -> list[RelationEdge]:
    ids = {w.concepts.resolve(a).id, w.concepts.resolve(b).id}
    return [e for e in w.relations.all() if {e.source_id, e.target_id} == ids and e.is_explicit]


def _claim(w: World, semantic: str, a: str = "a", b: str = "b") -> RelationEdge:
    return next(e for e in _claims(w, a, b) if e.semantic_relation == semantic)


def _survival(w: World, edge: RelationEdge, names=("a", "b"), step: int = STEP, cap: int = 40_000) -> int | None:
    """Idle observations until ``edge`` is pruned (concepts kept alive)."""
    nodes = [w.concepts.resolve(n) for n in names]
    dec = DecayEngine(w.concepts, w.relations, clock=w.clock)
    start = w.clock.tick
    while w.clock.tick - start < cap:
        w.clock.advance(step)
        for n in nodes:
            n.last_activated_tick = w.clock.tick
            n.last_activated = datetime.now(timezone.utc)
        dec.decay_relations()
        if edge.id in dec.prune_relations():
            return w.clock.tick - start
    return None


def _once(tmp_path, rel: str, times: int = 1, **kw):
    w = World(store_path=tmp_path / "w")
    _state(w, "a", "b", rel, times, **kw)
    semantic = w.relations.all()[0].semantic_relation
    return w, _claim(w, semantic)


def _contest(tmp_path, seq: str, pos: str = "depends_on", neg: str = "conflict"):
    """Ingest E / C statements in order; return (positive edge, negative edge)."""
    w = World(store_path=tmp_path / "w")
    for ch in seq:
        _state(w, "A", "B", pos if ch == "E" else neg)
    edges = _claims(w, "A", "B")
    e = next(x for x in edges if x.relation_type != RelationType.NEGATIVE)
    c = next(x for x in edges if x.relation_type == RelationType.NEGATIVE)
    return w, e, c


# ── the belief of a stated claim ───────────────────────────────────────


class TestClaimPrior:
    @pytest.mark.parametrize("semantic", NEGATIVE)
    def test_stated_negative_claim_starts_with_the_claim_prior(self, tmp_path, semantic):
        spec = SEMANTIC_RELATION_SPECS[semantic]
        rm = RelationManager(JsonStore(tmp_path / "s"))
        edge, is_new = rm.discover("x", "y", spec.axis, semantic_relation=semantic)
        assert is_new
        assert edge.probability == pytest.approx(NEGATIVE_CLAIM_PRIOR)
        assert edge.belief_prior == pytest.approx(NEGATIVE_CLAIM_PRIOR)
        # the inhibition gain and the structural strength are the spec's,
        # not the belief
        assert edge.weight == pytest.approx(spec.propagation_strength)
        assert edge.confidence == pytest.approx(spec.structural_strength)
        assert spec.propagation_strength <= 0.15 < edge.probability

    @pytest.mark.parametrize("semantic", NON_NEGATIVE)
    def test_positive_and_parallel_claims_are_seeded_as_before(self, tmp_path, semantic):
        spec = SEMANTIC_RELATION_SPECS[semantic]
        rm = RelationManager(JsonStore(tmp_path / "s"))
        edge, _ = rm.discover("x", "y", spec.axis, semantic_relation=semantic)
        assert edge.probability == pytest.approx(spec.propagation_strength)
        assert edge.weight == pytest.approx(spec.propagation_strength)
        assert spec.claim_prior == spec.propagation_strength

    def test_claim_prior_is_a_property_of_the_spec(self):
        for name, spec in SEMANTIC_RELATION_SPECS.items():
            if spec.axis == RelationType.NEGATIVE:
                assert spec.claim_prior == NEGATIVE_CLAIM_PRIOR, name
            else:
                assert spec.claim_prior == spec.propagation_strength, name

    def test_claim_prior_equals_the_reference_dependence(self):
        """Why 0.70: a stated claim has the same standing on either axis."""
        assert NEGATIVE_CLAIM_PRIOR == SEMANTIC_RELATION_SPECS["dependence"].propagation_strength

    @pytest.mark.parametrize("semantic", ["conflict", "disjointness", "instability"])
    @pytest.mark.parametrize("prior", [0.05, 0.3, 0.35, 0.9])
    def test_extractor_prior_still_wins(self, tmp_path, semantic, prior):
        """An extractor's own belief is never replaced by the default.

        ``0.3`` used to collide with the "unset" sentinel of the model
        validator and was silently replaced by the label's default.
        """
        w = World(store_path=tmp_path / "w")
        _state(w, "a", "b", semantic, prior=prior)
        assert _claim(w, semantic).probability == pytest.approx(prior)

    @pytest.mark.parametrize("semantic", ["conflict", "depends_on"])
    def test_an_extractor_prior_of_exactly_0_3_survives_a_reload(self, tmp_path, semantic):
        """``ensure_probability`` (the backfill for stores that predate the
        probability field) also read 0.3 as "missing" and replaced the belief
        by the structural confidence on the next load."""
        w = World(store_path=tmp_path / "w")
        _state(w, "a", "b", semantic, prior=0.3)
        w.close()
        again = World(store_path=tmp_path / "w")
        edge = next(e for e in _claims(again) if e.is_explicit)
        assert edge.probability == pytest.approx(0.3)

    def test_cooccurrence_edges_are_not_claims(self, tmp_path):
        w = World(store_path=tmp_path / "w")
        for _ in range(4):
            w.ingest(Observation(concepts=["a", "b"], source="s"))
        (heb,) = [e for e in w.relations.all() if not e.is_explicit]
        assert heb.probability == pytest.approx(0.15) and heb.belief_prior is None
        assert relation_floor(heb) == 0.0

    def test_bare_explicit_edges_get_the_same_defaults(self):
        neg = RelationEdge(source_id="a", target_id="b", relation_type="negative", semantic_relation="disjointness", is_explicit=True)
        assert neg.probability == pytest.approx(NEGATIVE_CLAIM_PRIOR)
        assert neg.weight == pytest.approx(0.05)  # gain untouched
        pos = RelationEdge(source_id="a", target_id="b", relation_type="positive", semantic_relation="enables", is_explicit=True)
        assert pos.probability == pytest.approx(0.76)
        # a non-explicit edge keeps the legacy default
        heb = RelationEdge(source_id="a", target_id="b", relation_type="negative", semantic_relation="conflict")
        assert heb.probability == pytest.approx(0.10) and heb.belief_prior is None

    def test_belief_prior_persists(self, tmp_path):
        w = World(store_path=tmp_path / "w")
        _state(w, "a", "b", "conflict")
        w.close()
        again = World(store_path=tmp_path / "w")
        edge = _claim(again, "conflict")
        assert edge.belief_prior == pytest.approx(NEGATIVE_CLAIM_PRIOR)
        assert edge.probability == pytest.approx(NEGATIVE_CLAIM_PRIOR)


# ── the inhibition gain is untouched ───────────────────────────────────


class TestInhibitionGainUnchanged:
    def test_fresh_gain_is_spec_gain_plus_one_reinforcement(self, tmp_path):
        """The first ingest reinforces once: gain 0.10 -> 0.176, as before."""
        w, edge = _once(tmp_path, "conflict")
        assert edge.weight == pytest.approx(0.10 + 0.08 / 1.05, abs=1e-6)
        assert edge.propagation_strength == pytest.approx(0.10)

    def test_inhibition_is_computed_from_the_gain_not_the_belief(self, tmp_path):
        """Two negative edges with identical gain but different belief inhibit identically."""
        scores = []
        for belief in (0.05, 0.95):
            w = World(store_path=tmp_path / f"w{belief}")
            _state(w, "A", "X", "conflict")
            _state(w, "A", "Y", "enables", 3)
            _state(w, "Y", "X", "enables", 3)
            edge = _claim(w, "conflict", "A", "X")
            edge.probability = belief
            # Read at the instant of the statement: once time passes, the
            # settled weight relaxes toward the belief-anchored floor
            # (0.1 × belief) — by design, and the same for every axis.
            edge.last_reinforced_tick, edge.last_decayed_tick = w.clock.tick, None
            engine = ActivationEngine(w.concepts, w.relations, clock=w.clock)
            out = engine.activate([w.concepts.resolve("A").id], min_activation=1e-6, record=False)
            scores.append(out.get(w.concepts.resolve("X").id, 0.0))
        assert scores[0] == pytest.approx(scores[1], rel=1e-6)

    def test_evidence_update_does_not_write_a_belief_into_the_gain(self):
        """A restated negative claim with an extractor prior keeps its gain.

        Now that a stated negative claim carries a real belief (0.70), copying
        the mixed belief into ``weight`` would multiply the inhibition of an
        extractor-restated conflict by ~3 (0.25 -> 0.72).
        """
        neg = RelationEdge(source_id="a", target_id="b", relation_type="negative", semantic_relation="conflict", is_explicit=True)
        gain = neg.weight
        neg.update_probability(prior_probability=0.5, prior_strength=1.0)
        assert neg.probability < NEGATIVE_CLAIM_PRIOR
        assert neg.weight == pytest.approx(gain)
        # positive claims keep the existing rule: evidence rescales the weight
        pos = RelationEdge(source_id="a", target_id="b", relation_type="positive", semantic_relation="dependence", is_explicit=True)
        pos.update_probability(prior_probability=0.5, prior_strength=1.0)
        assert pos.weight == pytest.approx(pos.probability)


# ── forgetting ─────────────────────────────────────────────────────────


class TestForgetting:
    @pytest.mark.parametrize("semantic", NEGATIVE)
    def test_stated_once_negative_claim_survives_like_a_dependence(self, tmp_path, semantic):
        wd, dep = _once(tmp_path / "d", "depends_on")
        wn, neg = _once(tmp_path / "n", semantic)
        kept_dep = _survival(wd, dep)
        kept_neg = _survival(wn, neg)
        assert kept_dep is not None and kept_neg is not None
        # was 350-500 for negatives against 8 100
        assert kept_neg > 3000
        assert abs(kept_neg - kept_dep) <= 2 * STEP
        assert kept_neg < 20_000  # era forgetting stays: not immortal

    @pytest.mark.parametrize("semantic", ["conflict", "disjointness"])
    def test_restated_negative_claim_survives_like_a_restated_dependence(self, tmp_path, semantic):
        wd, dep = _once(tmp_path / "d", "depends_on", 6)
        wn, neg = _once(tmp_path / "n", semantic, 6)
        assert neg.probability == pytest.approx(dep.probability)  # 0.768 = 1 - 0.3 * 0.95^5
        assert _survival(wn, neg) == _survival(wd, dep)

    def test_the_floor_is_anchored_on_belief_not_on_gain(self, tmp_path):
        w, edge = _once(tmp_path, "disjointness")
        assert edge.propagation_strength == pytest.approx(0.05)
        assert relation_floor(edge) == pytest.approx(0.1 * edge.probability)
        assert relation_floor(edge) > 0.02  # above the prune threshold: protected

    def test_belief_is_never_time_decayed(self, tmp_path):
        w, edge = _once(tmp_path, "conflict")
        p0 = edge.probability
        dec = DecayEngine(w.concepts, w.relations, clock=w.clock)
        for _ in range(60):
            w.clock.advance(STEP)
            dec.decay_relations()
        assert edge.probability == p0

    @pytest.mark.parametrize("semantic", ["conflict", "depends_on"])
    def test_a_claim_the_extractor_doubts_is_still_forgotten_fast(self, tmp_path, semantic):
        """Noise is not promoted: an extractor's 0.05 is below the floor threshold."""
        w, edge = _once(tmp_path, semantic, prior=0.05)
        assert edge.probability == pytest.approx(0.05)
        assert _survival(w, edge) < 1000

    def test_a_disconfirmed_negative_claim_is_still_forgotten(self, tmp_path):
        w, edge = _once(tmp_path, "conflict")
        undisturbed = _survival(*_once(tmp_path / "u", "conflict"))
        while edge.probability >= 0.2:
            edge.weaken()
        assert relation_floor(edge) < 0.02
        assert _survival(w, edge) < undisturbed / 2

    def test_reflect_pipeline_keeps_then_forgets_on_the_era_scale(self, tmp_path):
        def idle_world(idle: int):
            w = World(store_path=tmp_path / f"i{idle}")
            _state(w, "A", "B", "conflict")
            for _ in range(3):
                w.ingest(Observation(concepts=["E", "F"], source="s"))  # co-occurrence only
            # Concepts confirmed often enough are not noise; only the
            # relations are under test (a concept confirmed <= 6 times goes
            # with its relations after ~720 idle ticks).
            for _ in range(30):
                for n in "ABEF":
                    w.ingest(Observation(concepts=[n], source="s"))
            ids = {n: w.concepts.resolve(n).id for n in "ABEF"}
            w.clock.advance(idle)
            for n in "ABEF":
                w.ingest(Observation(concepts=[n], source="s"))
            w.reflect(light=True)
            return w, ids

        w, ids = idle_world(3000)
        assert w.relations.find_any_between(ids["A"], ids["B"]), "stated conflict was pruned"
        assert w.relations.find_any_between(ids["E"], ids["F"]) == []  # Hebbian edge still fades
        w, ids = idle_world(12_000)
        assert w.relations.find_any_between(ids["A"], ids["B"]) == []

    def test_survival_does_not_depend_on_the_settle_cadence(self, tmp_path):
        out = []
        for step in (10, 50, 200, 1000):
            w, edge = _once(tmp_path / str(step), "conflict")
            out.append(_survival(w, edge, step=step))
        # Settling is an exact semigroup, so the edge dies at the same instant
        # whatever the cadence; the probe only sees it at multiples of its own
        # step (was 440..1000 before the fix).
        steps = (10, 50, 200, 1000)
        assert all(abs(v - out[0]) <= step + steps[0] for v, step in zip(out, steps)), out


# ── contested pairs ────────────────────────────────────────────────────

SEQUENCES = ["EC", "EEECCC", "E" * 10 + "C" * 10, "EC" * 10, "EEEC" * 5, "E" * 5 + "C", "CEE"]


class TestContestedPairs:
    @pytest.mark.parametrize("seq", SEQUENCES)
    def test_swapping_the_axes_swaps_the_result(self, tmp_path, seq):
        """dependence and a negative claim start equal, so nothing is axis-biased."""
        swapped = seq.translate(str.maketrans("EC", "CE"))
        _, e1, c1 = _contest(tmp_path / "a", seq)
        _, e2, c2 = _contest(tmp_path / "b", swapped)
        assert e1.probability == pytest.approx(c2.probability, abs=1e-9)
        assert c1.probability == pytest.approx(e2.probability, abs=1e-9)

    @pytest.mark.parametrize("pos", ["depends_on", "enables"])
    def test_one_statement_each_is_contested_not_leaning(self, tmp_path, pos):
        """Was ``leaning`` 0.705 vs 0.100 (margin 0.61) for enables vs conflict."""
        w, e, c = _contest(tmp_path, "EC", pos=pos)
        (claim,) = assess([], _claims(w, "A", "B")).contested
        assert claim.status == "contested" and claim.margin < CONTEST_MARGIN
        assert c.probability > 0.6

    def test_equal_interleaved_restatement_is_contested(self, tmp_path):
        w, e, c = _contest(tmp_path, "EC" * 10)
        (claim,) = assess([], _claims(w, "A", "B")).contested
        assert claim.status == "contested"
        assert min(e.probability, c.probability) > 0.4  # neither is crushed

    @pytest.mark.parametrize("seq,leader", [("EEEC" * 5, "positive"), ("CCCE" * 5, "negative")])
    def test_the_more_restated_side_leads(self, tmp_path, seq, leader):
        w, e, c = _contest(tmp_path, seq)
        (claim,) = assess([], _claims(w, "A", "B")).contested
        lead = next(x for x in _claims(w, "A", "B") if x.id == claim.leading)
        assert lead.relation_type.value == leader
        assert claim.status == "leaning"

    def test_a_negative_claim_can_lead(self, tmp_path):
        w, e, c = _contest(tmp_path, "ECCC")
        assert c.probability > e.probability
        (claim,) = assess([], _claims(w, "A", "B")).contested
        assert claim.leading == c.id

    def test_blocked_restatement_is_recency_dominated_and_symmetric(self, tmp_path):
        """E x10 then C x10: the later block leads (was 0.45 vs 0.43 'contested')."""
        _, e, c = _contest(tmp_path / "a", "E" * 10 + "C" * 10)
        assert e.probability < 0.5 < c.probability
        _, e2, c2 = _contest(tmp_path / "b", "C" * 10 + "E" * 10)
        assert c2.probability < 0.5 < e2.probability

    def test_projection_reports_the_belief_of_a_stated_conflict(self, tmp_path):
        w = World(store_path=tmp_path / "w")
        _state(w, "pytorch", "gpu", "enables")
        _state(w, "pytorch", "gpu", "conflict")
        text = w.project(["pytorch"]).render(style="full")
        line = next(ln for ln in text.splitlines() if "→ conflict [negative]" in ln)
        assert "belief: 0.70" in line
        assert "Contested:" in text


# ── stores written before the change ───────────────────────────────────


def _write_legacy(root, mutate) -> None:
    """Rewrite every stored relation the way the previous code stored it."""
    for path in (root / "relations").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("belief_prior", None)
        mutate(data)
        path.write_text(json.dumps(data), encoding="utf-8")


class TestLegacyStores:
    def _fresh_and_legacy(self, tmp_path, build, legacy_probability):
        """The same history under the new code, and as a legacy store."""
        fresh = World(store_path=tmp_path / "fresh")
        build(fresh)
        fresh.close()
        ref = {e.semantic_relation: e.probability for e in fresh.relations.all() if e.is_explicit}

        legacy = World(store_path=tmp_path / "legacy")
        build(legacy)
        legacy.close()
        _write_legacy(
            tmp_path / "legacy",
            lambda d: d.update(probability=legacy_probability(d)) if d["is_explicit"] else None,
        )
        return ref, World(store_path=tmp_path / "legacy")

    def test_stated_once_legacy_claim_is_rebased(self, tmp_path):
        ref, w = self._fresh_and_legacy(
            tmp_path, lambda w: _state(w, "a", "b", "conflict"), lambda d: 0.10
        )
        edge = _claim(w, "conflict")
        assert edge.probability == pytest.approx(ref["conflict"]) == pytest.approx(NEGATIVE_CLAIM_PRIOR)
        assert edge.belief_prior == pytest.approx(NEGATIVE_CLAIM_PRIOR)
        assert edge.weight == pytest.approx(0.10 + 0.08 / 1.05, abs=1e-6)  # gain untouched

    def test_restated_legacy_claim_lands_where_the_new_code_would(self, tmp_path):
        # legacy: 1 - 0.9 * 0.95^5 = 0.304 after five bare confirmations
        ref, w = self._fresh_and_legacy(
            tmp_path,
            lambda w: _state(w, "a", "b", "conflict", 6),
            lambda d: 1 - 0.9 * 0.95**5,
        )
        assert _claim(w, "conflict").probability == pytest.approx(ref["conflict"], abs=1e-9)
        assert ref["conflict"] == pytest.approx(1 - 0.3 * 0.95**5)

    def test_disconfirmed_legacy_claim_keeps_its_disconfirmations(self, tmp_path):
        def build(w):
            _state(w, "a", "b", "conflict")
            for _ in range(4):  # contradicted, endpoints not mentioned: no co-occurrence
                w.ingest(Observation(contradicted_relations=[("a", "b", "conflict")], source="s"))

        ref, w = self._fresh_and_legacy(tmp_path, build, lambda d: 0.01)  # clamped by the old code
        edge = _claim(w, "conflict")
        assert edge.probability == pytest.approx(ref["conflict"], abs=1e-9)
        assert NEGATIVE_CLAIM_PRIOR - 0.25 < edge.probability < NEGATIVE_CLAIM_PRIOR

    @pytest.mark.parametrize("stored", [0.9, 0.05])
    def test_a_belief_the_legacy_seed_does_not_explain_is_kept(self, tmp_path, stored):
        """An extractor's own belief (high or low) is not rebased."""
        ref, w = self._fresh_and_legacy(
            tmp_path, lambda w: _state(w, "a", "b", "conflict"), lambda d: stored
        )
        edge = _claim(w, "conflict")
        assert edge.probability == pytest.approx(stored)
        assert edge.belief_prior == pytest.approx(stored)

    def test_positive_and_parallel_legacy_claims_are_untouched(self, tmp_path):
        def build(w):
            _state(w, "a", "b", "depends_on", 3)
            _state(w, "c", "d", "similar_to", 2)

        fresh = World(store_path=tmp_path / "f")
        build(fresh)
        before = {e.id: e.probability for e in fresh.relations.all() if e.is_explicit}
        fresh.close()
        _write_legacy(tmp_path / "f", lambda d: None)
        w = World(store_path=tmp_path / "f")
        for e in w.relations.all():
            if e.is_explicit:
                assert e.probability == pytest.approx(before[e.id])
                assert e.belief_prior is None

    def test_migration_is_idempotent_and_persisted(self, tmp_path):
        _, w = self._fresh_and_legacy(
            tmp_path, lambda w: _state(w, "a", "b", "disjointness", 3), lambda d: 0.143
        )
        first = _claim(w, "disjointness").probability
        w.close()
        again = World(store_path=tmp_path / "legacy")
        edge = _claim(again, "disjointness")
        assert edge.probability == pytest.approx(first)
        assert edge.belief_prior == pytest.approx(NEGATIVE_CLAIM_PRIOR)
        assert edge.adopt_claim_prior() is False  # already stamped

    def test_migrated_claim_survives_idle_time_like_a_new_one(self, tmp_path):
        _, w = self._fresh_and_legacy(
            tmp_path, lambda w: _state(w, "a", "b", "conflict"), lambda d: 0.10
        )
        assert _survival(w, _claim(w, "conflict")) > 3000

    def test_sqlite_store_is_migrated_too(self, tmp_path):
        from world0.store.sqlite_store import SqliteStore

        path = tmp_path / "w.sqlite"
        w = World(store_path=path, backend="sqlite")
        _state(w, "a", "b", "conflict")
        w.close()
        store = SqliteStore(path)
        edge = next(e for e in store.load_all_relations() if e.is_explicit)
        edge.probability, edge.belief_prior = 0.10, None
        store.save_relation(edge)
        store.close()
        again = World(store_path=path, backend="sqlite")
        assert _claim(again, "conflict").probability == pytest.approx(NEGATIVE_CLAIM_PRIOR)

    def test_adopt_only_touches_explicit_negative_edges(self):
        for kw in (
            dict(relation_type="positive", semantic_relation="dependence", is_explicit=True),
            dict(relation_type="parallel", semantic_relation="similarity_kernel", is_explicit=True),
            dict(relation_type="negative", semantic_relation="conflict", is_explicit=False),
        ):
            edge = RelationEdge(source_id="a", target_id="b", **kw)
            edge.belief_prior = None
            p = edge.probability
            assert edge.adopt_claim_prior() is False
            assert edge.probability == p and edge.belief_prior is None


class TestMigrationProperties:
    """The rebase is monotone, idempotent and only ever explains what it can."""

    @staticmethod
    def _legacy_edge(steps: str) -> RelationEdge:
        """An edge as the previous code left it: bare ``confirm`` (c) /
        ``weaken`` (w) applied to a belief that started at the inhibition gain."""
        edge = RelationEdge(
            source_id="a", target_id="b", relation_type="negative",
            semantic_relation="conflict", is_explicit=True,
        )
        edge.belief_prior = None
        edge.probability = 0.10  # the legacy seed
        for step in steps:
            edge.confirm() if step == "c" else edge.weaken()
        return edge

    @staticmethod
    def _histories():
        import random

        rng = random.Random(7)
        for _ in range(300):
            k, d = rng.randint(0, 25), rng.randint(0, 25)
            yield "confirm-first", "c" * k + "w" * d
            yield "weaken-first", "w" * d + "c" * k
            yield "interleaved", "".join(rng.choice("cw") for _ in range(k + d))

    def test_default_seeded_legacy_edges_are_rebased_never_lowered(self):
        for order, steps in self._histories():
            edge = self._legacy_edge(steps)
            stored, weight = edge.probability, edge.weight
            assert edge.adopt_claim_prior() is True, (order, steps)
            assert edge.belief_prior == NEGATIVE_CLAIM_PRIOR, (order, steps)  # explained, so rebased
            assert edge.probability >= stored - 1e-12, (order, steps)
            assert edge.weight == weight
            assert edge.adopt_claim_prior() is False  # idempotent

    def test_a_disconfirmed_legacy_edge_lands_on_the_fresh_replay(self):
        for weakens in range(0, 30):
            edge = self._legacy_edge("w" * weakens)
            edge.adopt_claim_prior()
            fresh = RelationEdge(
                source_id="a", target_id="b", relation_type="negative",
                semantic_relation="conflict", is_explicit=True,
            )
            for _ in range(weakens):
                fresh.weaken()
            assert edge.probability == pytest.approx(fresh.probability, abs=1e-9)

    def test_migration_cost_is_negligible(self):
        import time

        edges = [self._legacy_edge("ccc" + "w") for _ in range(2000)]
        start = time.perf_counter()
        for e in edges:
            e.adopt_claim_prior()
        assert time.perf_counter() - start < 1.0  # measured ~5 us per edge
