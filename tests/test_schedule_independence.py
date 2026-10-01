"""Existence, liveness and maturity are properties of the observation stream.

Regression tests for the leaks the independent review of the maturity
change found (docs/paper/world0-formal.md §3.5–§3.6, Theorem 3.7): a
concept or edge that is dead is dead whether or not a reflect has
physically deleted it, so the state after a stream cannot depend on when
``reflect()`` ran.  Every scenario runs the *same* events under reflect
cadences of 1, 50, 1000 and never, and requires identical outcomes.
"""

from __future__ import annotations

import random

import pytest

from world0 import Observation, World
from world0.concepts._identity_ops import split_concept
from world0.dynamics.decay import (
    FADING_THRESHOLD,
    PRUNE_MIN_IDLE_TICKS,
    concept_expired,
    relation_dead,
    settle_relation,
)
from world0.schemas.concept import (
    RECURRENCE_WINDOW,
    REVIVAL_MIN_CONFIDENCE,
    ConceptNode,
    Maturity,
)
from world0.schemas.relation import RelationEdge
from world0.schemas.types import RelationPrior

CADENCES = (1, 50, 1000, None)


def _drive(events: dict[int, Observation], horizon: int, every: int | None, path):
    """Ingest ``events`` (tick → observation), reflecting every ``every`` ticks."""
    w = World(store_path=path)
    for t in sorted(events):
        if every:
            r = (w.clock.tick // every + 1) * every
            while r <= t - 1:
                w.clock.advance(r - w.clock.tick)
                w.reflect(light=True)
                r += every
        w.clock.advance(t - 1 - w.clock.tick)
        w.ingest(events[t])
    w.clock.advance(max(0, horizon - w.clock.tick))
    w.reflect()
    return w


def _state(w: World) -> dict[str, tuple[str, float, int]]:
    return {
        c.name: (c.maturity.value, c.confidence, c.activation_count)
        for c in w.concepts.all()
    }


def _same_everywhere(events, horizon, tmp_path, tol=1e-6):
    states = [
        _state(_drive(events, horizon, every, tmp_path / str(i)))
        for i, every in enumerate(CADENCES)
    ]
    ref = states[-1]
    for every, state in zip(CADENCES, states):
        assert state.keys() == ref.keys(), (every, sorted(state), sorted(ref))
        for name, (maturity, conf, n) in state.items():
            assert (maturity, n) == (ref[name][0], ref[name][2]), (every, name, state[name], ref[name])
            assert conf == pytest.approx(ref[name][1], abs=tol), (every, name)
    return ref


# ── Existence: a concept past recovery is gone, reflect or not ───────────


class TestExistence:
    @pytest.mark.parametrize("T", [721, 800, 2000])
    def test_sparse_use_beyond_the_grace_is_forgotten_in_every_world(self, T, tmp_path):
        events = {1 + T * k: Observation(concepts=["c"], source="s") for k in range(15)}
        ref = _same_everywhere(events, 1 + T * 14 + 5, tmp_path)
        assert ref["c"][2] == 1  # recreated at the last mention: n = 1

    @pytest.mark.parametrize("T", [24, 720])
    def test_use_within_the_grace_reaches_established_in_every_world(self, T, tmp_path):
        events = {1 + T * k: Observation(concepts=["c"], source="s") for k in range(12)}
        ref = _same_everywhere(events, 1 + T * 12, tmp_path)
        assert ref["c"][0] == "established" and ref["c"][2] == 12

    def test_the_predicate_is_pure_and_agrees_with_prune(self, tmp_path):
        w = World(store_path=tmp_path)
        w.ingest(Observation(concepts=["one-off"], source="s"))
        node = w.concepts.resolve("one-off")
        tick = w.clock.tick
        assert not concept_expired(node, tick + 100)
        before = (node.confidence, node.maturity, node.last_decayed_tick)
        assert concept_expired(node, tick + int(PRUNE_MIN_IDLE_TICKS) + 5)
        assert (node.confidence, node.maturity, node.last_decayed_tick) == before  # not mutated
        w.clock.advance(int(PRUNE_MIN_IDLE_TICKS) + 5)
        w._decay.decay_concepts()
        assert w._decay.prune_concepts() == [node.id]


# ── Connections: dead edges and expired neighbours do not count ─────────


class TestConnectionsAreLive:
    def test_ghost_neighbours_do_not_make_a_concept_core(self, tmp_path):
        """Five one-shot partners expire; the edges outlive them in a world
        that never reflects but must not count toward CORE there."""
        events = {
            1: Observation(
                concepts=["c"] + [f"p{i}" for i in range(5)],
                relations=[("c", f"p{i}", "depends_on") for i in range(5)],
                source="s",
            )
        }
        for k in range(1, 41):
            events[1 + 48 * k] = Observation(concepts=["c"], source="s")
        ref = _same_everywhere(events, 1 + 48 * 41, tmp_path)
        assert ref["c"][0] == "established"  # not core: its partners are gone

    def test_a_dead_edge_is_not_revived_by_a_restatement(self, tmp_path):
        """A relation whose weight fell below the prune line is a fresh edge
        when restated, whether or not a reflect deleted it first."""
        outcomes = []
        for i, every in enumerate(CADENCES):
            w = World(store_path=tmp_path / str(i))

            def step(names=None):
                if names:
                    w.ingest(Observation(concepts=names, source="s", task="t"))
                else:
                    w.clock.advance(1)
                if every and w.clock.tick % every == 0:
                    w.reflect(light=True)

            for _ in range(3):
                step(["A", "C"])
            for _ in range(400):
                step()  # the co-occurrence edge decays below the prune line
            for _ in range(95):
                step(["A", "B"])
            step(["A", "C"])
            w.reflect()
            a = w.concepts.resolve("A")
            outcomes.append((a.maturity.value, a.activation_count))
        assert len(set(outcomes)) == 1, outcomes

    def test_connection_events_fire_the_lifecycle_when_a_prior_restates_an_edge(self, tmp_path):
        w = World(store_path=tmp_path)
        w.ingest(Observation(
            concepts=["A", "C", "D"],
            relations=[("A", "C", "supports"), ("A", "D", "supports")],
            source="s",
        ))
        a, c, d = (w.concepts.resolve(n) for n in "ACD")
        for cid in (c.id, d.id):
            edge = w.relations.find_between(a.id, cid)
            edge.weight = edge.confidence = 0.01  # dead, not yet pruned
        for _ in range(95):
            w.ingest(Observation(concepts=["A"], source="s"))
        assert a.maturity == Maturity.ESTABLISHED
        # Restate both edges through relation_priors (the update_probability
        # path): promotion must happen at the event, not at the next reflect.
        w.ingest(Observation(
            concepts=["C", "D"],
            relations=[("A", "C", "supports"), ("A", "D", "supports")],
            relation_priors=[
                RelationPrior(source="A", target=t, relation_type="supports", probability=0.9, strength=1.0)
                for t in ("C", "D")
            ],
            source="s",
        ))
        assert a.maturity == Maturity.CORE


# ── Relations settle exactly and before they are modified ───────────────


class TestRelationSettle:
    def _edge(self, **kw) -> RelationEdge:
        edge = RelationEdge(source_id="a", target_id="b", semantic_relation="dependence", **kw)
        edge.last_reinforced_tick = 0
        return edge

    def test_settle_is_an_exact_semigroup(self):
        rng = random.Random(4)
        for _ in range(300):
            p = rng.uniform(0.3, 1.0)
            whole, split = self._edge(), self._edge()
            w0 = rng.uniform(0.2, 1.0)
            for e in (whole, split):
                e.probability = p
                e.weight = e.confidence = w0
            total = rng.randint(30, 9000)
            cut = rng.randint(1, total - 1)
            settle_relation(whole, total)
            settle_relation(split, cut)
            settle_relation(split, total)
            assert split.weight == pytest.approx(whole.weight, abs=1e-6)
            assert split.confidence == pytest.approx(whole.confidence, abs=1e-6)

    def test_relation_dead_matches_the_prune_line_without_mutating(self):
        edge = self._edge()
        edge.probability = 0.05  # floor below the prune line: mortal
        edge.weight = 0.5
        assert not relation_dead(edge, 10)
        assert relation_dead(edge, 5000)
        assert edge.weight == 0.5

    def test_negative_claim_keeps_its_gain_when_a_prior_seeds_the_belief(self, tmp_path):
        gains = []
        for prior in (None, 0.7, 0.9, 0.3):
            w = World(store_path=tmp_path / str(prior))
            kw = {}
            if prior is not None:
                kw["relation_priors"] = [
                    RelationPrior(source="a", target="b", relation_type="conflict", probability=prior, strength=1.0)
                ]
            w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "conflict")], source="s", **kw))
            edge = w.relations.all()[0]
            gains.append(edge.weight)
            if prior is not None:
                assert edge.probability == pytest.approx(prior)
        assert max(gains) - min(gains) < 1e-9, gains


# ── Hooks: every path that can earn a promotion evaluates it ──────────


class TestHooks:
    def test_promotion_by_recorded_activation(self, tmp_path):
        """The record=True path judges promotion at the event, like ingest."""
        w = World(store_path=tmp_path)
        for _ in range(11):
            w.ingest(Observation(concepts=["x", "y"], source="s"))
            w.clock.advance(RECURRENCE_WINDOW)
        w.ingest(Observation(concepts=["y"], source="s"))
        x = w.concepts.resolve("x")
        assert x.maturity == Maturity.DEVELOPING
        w.clock.advance(RECURRENCE_WINDOW)
        w._activation.activate([x.id], record=True)
        assert x.maturity == Maturity.ESTABLISHED

    def test_positive_confidence_adjustment_is_judged(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(10):
            w.ingest(Observation(concepts=["z"], source="s"))
            w.clock.advance(RECURRENCE_WINDOW)
        z = w.concepts.resolve("z")
        z.confidence = 0.5
        w.concepts.adjust_confidence(z.id, 0.2)
        assert z.maturity == Maturity.ESTABLISHED  # dense gate: n>=10, rho>=3, c>=0.6

    def test_a_swapped_lifecycle_policy_keeps_sole_authority(self, tmp_path):
        """The documented override point: replace ``World._lifecycle`` after
        construction.  The event hooks call through it, so a policy without
        them is evaluated at reflect only and the original engine is inert."""

        class Inert:
            def evaluate(self):
                return [], []

        w = World(store_path=tmp_path)
        w._lifecycle = Inert()
        for _ in range(60):
            w.ingest(Observation(concepts=["x", "y"], source="s"))
            w.clock.advance(RECURRENCE_WINDOW)
        assert w.concepts.resolve("x").maturity == Maturity.EMBRYONIC

    def test_split_concept_is_born_now(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(3):
            w.ingest(Observation(concepts=["Mercury"], source="s"))
        w.clock.advance(1000)
        new = split_concept(w.concepts, w.concepts.resolve("Mercury").id, "Mercury planet")
        assert new.created_tick == w.clock.tick == new.last_activated_tick
        w.reflect()
        assert w.concepts.resolve("Mercury planet") is not None


# ── Recurrence spacing and revival ───────────────────────────────────


class TestRecurrence:
    @pytest.mark.parametrize("start", list(range(1, 49, 3)))
    def test_a_burst_counts_at_most_twice_at_any_phase(self, start):
        """Spacing is measured from the last counted activation, not on a
        grid: 31 consecutive ticks span 31 > 24 ticks and count twice; the
        dense gate's three windows are never met by a burst."""
        node = ConceptNode(name="b")
        for t in range(start, start + 31):
            node.activate(tick=t)
        assert node.recurrence_count == 2

    def test_cadence_of_a_full_window_counts_every_use(self):
        node = ConceptNode(name="c")
        for k in range(10):
            node.activate(tick=1 + RECURRENCE_WINDOW * k)
        assert node.recurrence_count == 10

    def test_grid_records_written_before_load_keep_counting(self):
        node = ConceptNode(name="legacy", recurrence_count=4, last_recurrence_window=10)
        node.last_activated_tick = 250
        node.activate(tick=251)  # window 10 spans ticks 240-263: 251 - 240 < 24
        assert node.recurrence_count == 4
        node.activate(tick=270)
        assert node.recurrence_count == 5

    def test_revival_lands_at_the_fading_line(self):
        assert REVIVAL_MIN_CONFIDENCE == FADING_THRESHOLD
        node = ConceptNode(name="old", maturity=Maturity.FADING, confidence=0.001)
        node.activation_count = 40
        node.recurrence_count = 40
        node.activate(tick=100)
        assert node.maturity == Maturity.DEVELOPING
        assert node.confidence >= FADING_THRESHOLD  # would be re-marked FADING otherwise


# ── Projection: ranking survives faded seeds ────────────────────────────


class TestProjectionQuantum:
    @pytest.mark.parametrize("seed_confidence", [0.3, 1e-3, 1e-6, 1e-9])
    def test_neighbour_order_does_not_depend_on_the_absolute_scale(self, seed_confidence, tmp_path):
        w = World(store_path=tmp_path)
        for name, rel in (("n_strong", "depends_on"), ("n_weak", "similar_to")):
            for _ in range(4):
                w.ingest(Observation(concepts=["seed", name], relations=[("seed", name, rel)], source="s"))
        seed = w.concepts.resolve("seed")
        seed.confidence = seed_confidence
        order = [c.name for c in w.project(["seed"]).concepts]
        assert order[0] == "seed"
        assert order.index("n_strong") < order.index("n_weak")


# ── The property: random streams are schedule independent ────────────


def _random_stream(seed: int, horizon: int = 2500):
    r = random.Random(seed)
    names = [f"c{i}" for i in range(5)]
    cadence = {n: r.choice([12, 24, 72, 168, 400, 720, 1500]) for n in names}
    ticks: dict[int, dict] = {}

    def at(t):
        return ticks.setdefault(t, {"concepts": set(), "weak": set(), "rels": [], "contra": []})

    for n in names:
        t = r.randint(1, cadence[n])
        burst = r.random() < 0.3
        while t < horizon:
            at(t)["concepts"].add(n)
            if burst and r.random() < 0.5:
                t += r.randint(1, 3)
            else:
                t += max(1, int(r.expovariate(1 / cadence[n]))) if r.random() < 0.5 else cadence[n]
            if r.random() < 0.1 and t < horizon:
                at(t)["weak"].add(n)
    for e in ticks.values():
        cs = sorted(e["concepts"])
        if len(cs) >= 2 and r.random() < 0.6:
            a, b = r.sample(cs, 2)
            e["rels"].append((a, b, r.choice(["supports", "conflict", "depends_on", "excludes"])))
        if len(cs) >= 2 and r.random() < 0.1:
            a, b = r.sample(cs, 2)
            e["contra"].append((a, b, "supports"))
    events = {
        t: Observation(
            concepts=sorted(e["concepts"]),
            weakened=sorted(e["weak"]),
            relations=e["rels"],
            contradicted_relations=e["contra"],
            source="s",
            task="t",
        )
        for t, e in ticks.items()
    }
    return events, horizon


@pytest.mark.parametrize("seed", range(8))
def test_random_streams_end_in_the_same_state_under_any_reflect_schedule(seed, tmp_path):
    events, horizon = _random_stream(seed)
    _same_everywhere(events, horizon, tmp_path)
