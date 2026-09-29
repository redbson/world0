"""Formal properties of World 0's dynamics (docs/paper/world0-formal.md).

Each test states one proposition of the paper and checks it against the
engines.  The first group are regressions for the four gaps the
formalisation found (F1–F4); the rest pin the closed forms the proofs
rely on, so a change that breaks a proof breaks a test.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone

import pytest

from world0 import Observation, World
from world0.context.focus import FOCUS_CAPACITY, Focus
from world0.dynamics.activation import ActivationEngine, _accumulate
from world0.dynamics.decay import (
    FADING_THRESHOLD,
    evidence_floor,
    settle_concept,
)
from world0.schemas.concept import ConceptNode, Maturity
from world0.schemas.relation import SEMANTIC_RELATION_SPECS, RelationEdge

NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def _node(n: int, **kw) -> ConceptNode:
    node = ConceptNode(name="x", **kw)
    node.activation_count = n
    node.last_activated = NOW
    return node


# ── Regressions for the gaps found by the formalisation ─────────────────


class TestFoundGaps:
    def test_f1_faded_seed_is_still_projected(self, tmp_path):
        """Prop 7.1: every seed is in its projection (capped by max_concepts)."""
        w = World(store_path=tmp_path)
        for _ in range(20):
            w.ingest(Observation(concepts=["strong", "friend"]))
        w.ingest(Observation(concepts=["weak"]))
        weak = w.concepts.resolve("weak")
        weak.confidence = 0.004
        weak.maturity = Maturity.FADING
        names = [c.name for c in w.project(["strong", "weak"]).concepts]
        assert names[:2] == ["strong", "weak"]

    def test_f1_inhibited_seed_is_returned_by_activation(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(10):
            w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "conflict")]))
        a, b = w.concepts.resolve("a"), w.concepts.resolve("b")
        b.confidence = 0.02
        act = ActivationEngine(w.concepts, w.relations, clock=w.clock).activate(
            [a.id, b.id], record=False
        )
        assert b.id in act and act[b.id] >= 0.0

    def test_f2_generic_claim_does_not_weaken_a_conflict(self, tmp_path):
        """Prop 4.4: opposition is symmetric; generic claims oppose nothing."""
        w = World(store_path=tmp_path)
        w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "conflict")]))
        r = w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "related_to")]))
        assert r.weakened_relations == []

    def test_f2_opposition_is_symmetric_for_every_pair_of_semantics(self):
        for a in SEMANTIC_RELATION_SPECS:
            for b in SEMANTIC_RELATION_SPECS:
                ea = RelationEdge(source_id="x", target_id="y", semantic_relation=a)
                eb = RelationEdge(source_id="x", target_id="y", semantic_relation=b)
                assert ea.opposes(eb.relation_type, eb.semantic_relation) == eb.opposes(
                    ea.relation_type, ea.semantic_relation
                ), (a, b)

    def test_f3_revalidation_uses_exact_cooccurrence(self, tmp_path):
        """Prop 5.4: a link delayed by the gate is judged on its true count."""
        w = World(store_path=tmp_path)
        for i in range(12):
            w.ingest(Observation(concepts=["x", f"n{i}"]))
            w.ingest(Observation(concepts=["y", f"m{i}"]))
        for _ in range(8):
            w.ingest(Observation(concepts=["x", "y"]))
        x, y = w.concepts.resolve("x"), w.concepts.resolve("y")
        edge = w.relations.find_any_between(x.id, y.id)[0]
        # true association 8 / (20 + 20 − 8) = 0.25 ≥ 0.1; the old estimate
        # (r + 2 = 4 co-occurrences) put it at 4 / 36 ≈ 0.11, next to the cut.
        assert w._hebbian.cooccurrences(x.id, y.id) == 8
        assert edge.reinforcement_count + 2 < 8
        assert edge.id not in w._hebbian.revalidate()

    def test_f5_a_pair_is_predicted_once_however_many_claims_link_it(self, tmp_path):
        """§9: predictions are per companion, not per relation."""
        w = World(store_path=tmp_path)
        for _ in range(10):
            w.ingest(Observation(
                concepts=["a", "b", "c"],
                relations=[("a", "b", "enables"), ("a", "b", "conflict")],
            ))
        r = w.ingest(Observation(concepts=["a", "c"]))
        # a→b, a→c, c→a, c→b, all certain; a→b and c→b absent → 2 / 4
        assert r.prediction.missing_ratio == pytest.approx(0.5)

    @pytest.mark.parametrize("cadence", [24, 168])
    def test_f4_forgetting_does_not_depend_on_reflect_cadence(self, tmp_path, cadence):
        """Theorem 3.2: decay owed before a use is settled, not dropped."""
        finals = []
        for i, every in enumerate((1, 1000, None)):
            w = World(store_path=tmp_path / str(i))
            for _ in range(20):
                w.ingest(Observation(concepts=["c", "d"], relations=[("c", "d", "depends_on")]))
                for _ in range(cadence - 1):
                    w.clock.advance(1)
                    if every and w.clock.tick % every == 0:
                        w._decay.decay_concepts()
                        w._decay.decay_relations()
            w._decay.decay_concepts()
            w._decay.decay_relations()
            finals.append(
                (w.concepts.resolve("c").confidence, w.relations.all()[0].weight)
            )
        for conf, weight in finals[1:]:
            assert conf == pytest.approx(finals[0][0], abs=0.005)
            assert weight == pytest.approx(finals[0][1], abs=0.005)


# ── Concept dynamics ──────────────────────────────────────────────────


class TestConceptDynamics:
    def test_decay_split_is_exact(self):
        """Prop 3.1: settlement is the exact flow of ``c' = -λ (c − f(t))⁺``
        with a moving floor, hence a semigroup — splitting an interval leaves
        the same confidence.  (The earlier scheme froze the floor at the
        interval end and left the era correction r2 (1 − r1)(f1 − f2) ≈ 0.008
        for 2000 + 3000 ticks; that residual is what made the maturity
        trajectory depend on the reflect cadence up to a 0.01 bound.)"""
        node_a = _node(40, maturity=Maturity.DEVELOPING, confidence=0.9)
        node_b = _node(40, maturity=Maturity.DEVELOPING, confidence=0.9)
        h1, h2 = 2000, 3000
        settle_concept(node_a, h1, NOW)
        settle_concept(node_a, h1 + h2, NOW)
        settle_concept(node_b, h1 + h2, NOW)
        assert node_a.confidence == pytest.approx(node_b.confidence, abs=1e-12)
        # ... and the floor still moves: the result sits above the floor at the end
        assert node_b.confidence > evidence_floor(
            _node(40), now_tick=0, now=NOW
        ) * 2 ** (-(h1 + h2) / 4380)

    def test_settle_is_idempotent_at_one_instant(self):
        node = _node(5, maturity=Maturity.DEVELOPING, confidence=0.8)
        settle_concept(node, 300, NOW)
        once = node.confidence
        settle_concept(node, 300, NOW)
        assert node.confidence == once

    def test_noise_threshold_is_six_confirmations(self):
        """Prop 3.3: the evidence floor is below FADING_THRESHOLD iff n ≤ 6."""
        for n in range(1, 30):
            floor = evidence_floor(_node(n), now_tick=0, now=NOW)
            assert (floor < FADING_THRESHOLD) == (n <= 6)

    def test_evidence_is_monotone_in_confirmations(self):
        values = [_node(n).evidence() for n in range(0, 100)]
        assert all(a < b for a, b in zip(values, values[1:]))


# ── Relations and Hebbian statistics ─────────────────────────────────


class TestRelations:
    def test_confirm_closed_form(self):
        """Prop 4.1: p_k = 1 − (1 − p0) · 0.95^k."""
        edge = RelationEdge(source_id="a", target_id="b", semantic_relation="dependence")
        p0 = edge.probability
        for k in range(1, 30):
            edge.confirm()
            assert edge.probability == pytest.approx(1 - (1 - p0) * 0.95**k)

    def test_jaccard_is_a_function_of_the_two_conditionals(self):
        """Lemma 5.1: J = 1 / (1/P(b|a) + 1/P(a|b) − 1)."""
        rng = random.Random(0)
        for _ in range(500):
            c = rng.randint(1, 40)
            na, nb = c + rng.randint(0, 80), c + rng.randint(0, 80)
            assert c / (na + nb - c) == pytest.approx(1 / (na / c + nb / c - 1))


# ── Activation ───────────────────────────────────────────────────────


class TestActivation:
    def test_noisy_or_is_a_bounded_t_conorm(self):
        """Prop 6.1."""
        rng = random.Random(1)
        for _ in range(1000):
            S = rng.uniform(0.01, 1.0)
            a, b, c = (rng.uniform(0, S) for _ in range(3))
            ab = _accumulate(a, b, S)
            assert ab == pytest.approx(_accumulate(b, a, S))
            assert _accumulate(ab, c, S) == pytest.approx(_accumulate(a, _accumulate(b, c, S), S))
            assert max(a, b) - 1e-12 <= ab <= S + 1e-12
            assert _accumulate(0.0, rng.uniform(S, 5 * S), S) <= S + 1e-12

    def test_floor_is_strictly_increasing(self):
        """Prop 6.2."""
        xs = [i / 500 for i in range(1, 1500)]
        ys = [ActivationEngine._apply_floor(x, 1.0) for x in xs]
        assert all(b > a for a, b in zip(ys, ys[1:]))
        assert min(ys) >= 0.9

    def test_seed_dominance_on_random_worlds(self, tmp_path):
        """Theorem 6.3: no propagated concept exceeds the strongest seed."""
        rng = random.Random(2)
        for trial in range(8):
            w = World(store_path=tmp_path / str(trial))
            names = [f"v{i}" for i in range(15)]
            for _ in range(40):
                obs = rng.sample(names, 3)
                rel = rng.choice(["depends_on", "contains", "similar_to", "conflict"])
                w.ingest(Observation(concepts=obs, relations=[(obs[0], obs[1], rel)]))
            seeds = [w.concepts.resolve(n).id for n in rng.sample(names, 2)]
            act = ActivationEngine(w.concepts, w.relations, clock=w.clock).activate(
                seeds, max_depth=3, decay=0.9, record=False
            )
            S = max(w.concepts.get(s).confidence for s in seeds)
            assert all(v <= S + 1e-9 for k, v in act.items() if k not in seeds)


# ── Focus ────────────────────────────────────────────────────────────


class TestFocus:
    def test_single_ignition_lives_three_updates(self):
        """Prop 8.1: ⌊log 0.15 / log 0.6⌋ = 3."""
        focus = Focus()
        focus.update(["a"])
        lifetime = 0
        while True:
            focus.update([])
            if "a" not in focus.items():
                break
            lifetime += 1
        assert lifetime == math.floor(math.log(0.15) / math.log(0.6)) == 3

    def test_capacity_is_never_exceeded(self):
        rng = random.Random(3)
        focus = Focus()
        for _ in range(300):
            focus.update(rng.sample([f"c{i}" for i in range(40)], rng.randint(0, 15)))
            assert len(focus) <= FOCUS_CAPACITY
