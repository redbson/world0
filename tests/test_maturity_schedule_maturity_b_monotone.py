"""Maturity dynamics: sparse cadence reaches ESTABLISHED, and the maturity
trajectory no longer depends on when ``reflect()`` runs.

Baseline probes (each failed before the change; numbers in
``docs/world0-cognitive-dynamics-analysis.md`` §7.22):

- a concept used every 720 observations needed 324 uses (≈ 233 000
  observations) to reach ESTABLISHED, because the promotion gate read the
  *decayed* confidence, whose sparse-cadence equilibrium is capped by the
  evidence floor;
- the same observation stream ended ESTABLISHED / DEVELOPING / EMBRYONIC
  depending on whether reflect ran every observation or never, because
  promotion was evaluated only inside reflect (and a promotion changes the
  half-life used afterwards), FADING was entered whenever a settle happened
  to run, and CORE counted connections at reflect time.

The gates are now functions of counters that only move at events
(``n``, spaced recurrence ``ρ``, disconfirmation balance, connections) and
are evaluated at the events; the FADING boundary is solved exactly inside a
settle.  See ``dynamics/lifecycle.py`` and docs/paper Theorem 3.7.
"""

from __future__ import annotations

import random
import tempfile

import pytest

from world0 import Observation, World
from world0.dynamics.decay import (
    EVIDENCE_FLOOR_ERA_HL,
    FADING_THRESHOLD,
    concept_half_life,
    settled_confidence,
    evidence_floor,
    relax_confidence,
    settle_concept,
)
from world0.dynamics.lifecycle import (
    SPACED_DEVELOPING_EVIDENCE,
    SPACED_ESTABLISHED_EVIDENCE,
    LifecycleEngine,
    core_connections_required,
)
from world0.projection.engine import ProjectionEngine
from world0.projection.metacognition import TENTATIVE_EVIDENCE, WELL_EVIDENCED
from world0.schemas.concept import RECURRENCE_CHAIN_GAP, ConceptNode, Maturity
from world0.schemas.relation import RelationType

E = EVIDENCE_FLOOR_ERA_HL


# ── helpers ───────────────────────────────────────────────────────────


def use(*, weaken: bool = False) -> Observation:
    return Observation(
        concepts=["c", "anchor"],
        relations=[("c", "anchor", "depends_on")],
        weakened=["c"] if weaken else [],
        task="routine",
        source="s",
    )


def link(partner: str) -> Observation:
    """Create c → partner without activating c (c is resolved by name)."""
    return Observation(
        concepts=[partner], relations=[("c", partner, "supports")], task="link", source="s"
    )


def drive(events: dict[int, Observation], horizon: int, every: int | None, observe=None):
    """Run a fixed stream with reflect every ``every`` ticks (None: never),
    one final reflect at the horizon; returns (maturity, confidence) or None.
    ``observe(world)`` is called after every event and every reflect."""
    world = World(store_path=tempfile.mkdtemp(prefix="mat_"))
    ticks = sorted(events)
    i = 0
    while world.clock.tick < horizon:
        now = world.clock.tick
        nxt_event = ticks[i] if i < len(ticks) else None
        nxt_reflect = (now // every + 1) * every if every else None
        stop = min(x for x in (nxt_event, nxt_reflect, horizon) if x is not None)
        if nxt_event == stop:
            world.clock.advance(stop - 1 - now)
            world.ingest(events[stop])
            i += 1
            if observe:
                observe(world)
        else:
            world.clock.advance(stop - now)
        if every and stop % every == 0 and stop < horizon:
            world.reflect(light=True)
            if observe:
                observe(world)
    world.reflect()
    if observe:
        observe(world)
    node = world.concepts.resolve("c")
    return None if node is None else (node.maturity, node.confidence)


def cadence(T: int, uses: int = 30):
    return {k * T: use() for k in range(1, uses + 1)}, (uses + 1) * T


def burst():
    return {t: use() for t in range(1, 31)}, 5030


def disconfirmed(every: int, T: int = 24, uses: int = 30):
    return {k * T: use(weaken=(k % every == 0)) for k in range(1, uses + 1)}, (uses + 1) * T


def late_links():
    events = {k * 24: use() for k in range(1, 41)}
    t0 = 40 * 24
    for i in range(5):
        events[t0 + 300 + i] = link(f"p{i}")
    for k in range(1, 6):
        events[t0 + 720 * k] = use()
    return events, t0 + 720 * 6


def random_noise(seed: int, dis: float, uses: int = 40):
    rnd = random.Random(seed)
    t, events = 0, {}
    for _ in range(uses):
        t += rnd.randint(24, 2500)
        events[t] = use(weaken=rnd.random() < dis)
    return events, t + 500


def ever_established(events, horizon: int, every: int | None) -> bool:
    """Whether the concept is ESTABLISHED / CORE at any check point."""
    seen = []

    def observe(world):
        node = world.concepts.resolve("c")
        if node is not None and node.maturity in (Maturity.ESTABLISHED, Maturity.CORE):
            seen.append(world.clock.tick)

    drive(events, horizon, every, observe)
    return bool(seen)


# ═══════════════════════════════════════════════════════════════════════
# Theorem 3.7 — the trajectory is a function of the observation stream
# ═══════════════════════════════════════════════════════════════════════


class TestScheduleIndependence:
    @pytest.mark.parametrize(
        "name, stream, cadences, expected",
        [
            ("daily", cadence(24), (1, 50, 1000, None), Maturity.ESTABLISHED),
            ("every 72", cadence(72), (1, 50, 1000, None), Maturity.ESTABLISHED),
            ("weekly", cadence(168), (50, 1000, None), Maturity.ESTABLISHED),
            ("monthly", cadence(720), (50, 1000, None), Maturity.ESTABLISHED),
            ("burst then abandon", burst(), (25, 1000, None), Maturity.DEVELOPING),
            ("gains connections late", late_links(), (100, 1000, None), Maturity.CORE),
            ("disconfirmed every 3rd", disconfirmed(3), (1, 50, None), Maturity.DEVELOPING),
            ("disconfirmed every use", disconfirmed(1), (1, 50, None), Maturity.FADING),
        ],
    )
    def test_final_state_does_not_depend_on_reflect_cadence(
        self, name, stream, cadences, expected
    ):
        events, horizon = stream
        finals = [drive(events, horizon, every) for every in cadences]
        assert all(f is not None for f in finals), (name, finals)
        assert {m for m, _ in finals} == {expected}, (name, finals)
        confidences = [c for _, c in finals]
        # exact up to the wall-clock drift term (≈ 1e-9), far below the 0.01
        # era-correction bound the frozen-floor scheme needed
        assert max(confidences) - min(confidences) < 1e-6, (name, finals)

    def test_one_shot_concept_is_pruned_under_every_cadence(self):
        events, horizon = {1: use()}, 5000
        assert [drive(events, horizon, every) for every in (1000, None)] == [None, None]

    def test_reflect_reports_promotions_applied_at_events(self, tmp_path):
        """Promotion happens at the activation; the next reflect still
        reports it (ids that rose since the previous reflect)."""
        world = World(store_path=tmp_path)
        for _ in range(4):
            world.ingest(use())
        node = world.concepts.resolve("c")
        assert node.maturity == Maturity.DEVELOPING  # before any reflect
        result = world.reflect()
        assert node.id in result.promoted_concepts
        assert node.id not in world.reflect().promoted_concepts


class TestExactSettle:
    def test_relaxation_is_an_exact_semigroup(self):
        rng = random.Random(7)
        for _ in range(3000):
            c, f = rng.uniform(0.01, 1.0), rng.uniform(0.0, 0.35)
            H = rng.choice([24.0, 168.0, 720.0, 2160.0, 8760.0, 4380.0000001])
            t1, t2 = rng.uniform(0, 3000), rng.uniform(0, 3000)
            whole = relax_confidence(c, f, H, t1 + t2)
            split = relax_confidence(
                relax_confidence(c, f, H, t1), f * 2 ** (-t1 / E), H, t2
            )
            assert whole == pytest.approx(split, abs=1e-9)

    def test_relaxation_never_dips_below_the_moving_floor(self):
        for H in (24.0, 720.0, 8760.0):
            for dt in (1, 100, 5000, 50000):
                assert relax_confidence(0.9, 0.2, H, dt) >= 0.2 * 2 ** (-dt / E) - 1e-12

    @pytest.mark.parametrize(
        "n, c0, maturity",
        [(4, 0.5, Maturity.DEVELOPING), (6, 0.9, Maturity.ESTABLISHED), (5, 0.3, Maturity.CORE)],
    )
    def test_fading_boundary_does_not_depend_on_where_settle_runs(self, n, c0, maturity):
        """M2(b): a settle inside the gap in which confidence crosses the
        FADING line must not change the outcome (the half-life switches
        exactly at the crossing)."""
        results = []
        for cuts in ([], [200], [200, 900], [1, 2, 3, 500, 1500]):
            node = ConceptNode(name="x", confidence=c0, maturity=maturity)
            node.activation_count = n
            for cut in [*cuts, 2500]:
                settle_concept(node, cut)
            results.append((node.maturity, node.confidence))
        assert len({m for m, _ in results}) == 1
        confidences = [c for _, c in results]
        assert max(confidences) - min(confidences) < 1e-6

    def test_crossing_switches_to_the_fading_half_life(self):
        """Once confidence crosses the line the remaining gap decays with the
        FADING half-life, so it ends lower than one pre-crossing half-life
        over the whole gap would leave it."""
        node = ConceptNode(name="x", confidence=0.5, maturity=Maturity.DEVELOPING)
        node.activation_count = 4
        floor = evidence_floor(node, now_tick=0, now=node.last_activated)
        unswitched = relax_confidence(0.5, floor, concept_half_life(node), 3000)
        fell = settle_concept(node, 3000, node.last_activated)
        assert fell and node.maturity == Maturity.FADING
        assert node.confidence < FADING_THRESHOLD
        assert node.confidence < unswitched

    def test_disconfirmation_settles_first_and_marks_fading_at_the_event(self, tmp_path):
        """weaken() used to subtract its penalty from the *unsettled*
        confidence, so the penalty itself was decayed for the whole gap —
        by an amount that depended on whether a reflect had run."""
        finals = []
        for i, reflect_mid in enumerate((False, True)):
            world = World(store_path=tmp_path / str(i))
            for _ in range(8):
                world.ingest(use())
            world.clock.advance(400)
            if reflect_mid:
                world.reflect(light=True)
            world.clock.advance(400)
            world.weaken("c")
            node = world.concepts.resolve("c")
            finals.append((node.maturity, node.confidence))
        assert finals[0][0] == finals[1][0]
        assert finals[0][1] == pytest.approx(finals[1][1], abs=1e-6)

        world = World(store_path=tmp_path / "fade")
        world.ingest(use())
        for _ in range(6):
            world.weaken("c")
        node = world.concepts.resolve("c")
        assert node.confidence < FADING_THRESHOLD
        assert node.maturity == Maturity.FADING  # judged at the event, not at the next settle


# ═══════════════════════════════════════════════════════════════════════
# M1 — a sparse cadence reaches ESTABLISHED
# ═══════════════════════════════════════════════════════════════════════


class TestSparseCadenceReachesEstablished:
    @staticmethod
    def _uses_to_established(T: int, cap: int = 60) -> tuple[int, ConceptNode]:
        world = World(store_path=tempfile.mkdtemp())
        for uses in range(1, cap + 1):
            world.ingest(Observation(concepts=["c"], source="s"))
            node = world.concepts.resolve("c")
            if node.maturity == Maturity.ESTABLISHED:
                return uses, node
            world.clock.advance(T - 1)
        return cap + 1, node

    @pytest.mark.parametrize("T", [24, 72, 168, 720])
    def test_twelve_uses_at_any_cadence_within_the_prune_grace(self, T):
        uses, node = self._uses_to_established(T)
        # Baseline: 13 / 15 / 27 / 324 uses at T = 24 / 72 / 168 / 720.
        assert uses == 12
        assert node.recurrence_count == 12
        assert node.evidence() >= SPACED_ESTABLISHED_EVIDENCE

    @pytest.mark.parametrize("T", [721, 800, 2000])
    def test_a_cadence_beyond_the_prune_grace_is_forgotten_as_noise(self, T):
        """Between uses the concept has faded (n <= 6 keeps its floor below
        the prune line) and idled past PRUNE_MIN_IDLE_TICKS, so every mention
        meets a fresh node: the same in a world that never reflects."""
        uses, node = self._uses_to_established(T, cap=20)
        assert uses > 20 and node.maturity != Maturity.ESTABLISHED
        assert node.activation_count == 1

    def test_a_cadence_beyond_half_an_era_never_qualifies(self):
        uses, node = self._uses_to_established(int(RECURRENCE_CHAIN_GAP), cap=40)
        assert uses > 40 and node.maturity != Maturity.ESTABLISHED
        assert node.recurrence_count == 1  # every use opens a new chain

    def test_eleven_uses_are_not_enough(self):
        world = World(store_path=tempfile.mkdtemp())
        for _ in range(11):
            world.ingest(Observation(concepts=["c"], source="s"))
            world.clock.advance(719)
        assert world.concepts.resolve("c").maturity != Maturity.ESTABLISHED


# ═══════════════════════════════════════════════════════════════════════
# Constraints — noise stays out; nothing is immortal; fading is reversible
# ═══════════════════════════════════════════════════════════════════════


class TestNoiseStaysOut:
    @pytest.mark.parametrize("every", [25, 1000, None])
    def test_thirty_mentions_in_one_window_are_not_established(self, every):
        events = {t: use() for t in range(1, 24)}
        for t in range(1, 8):  # seven of them mention the concept twice
            events[t] = Observation(
                concepts=["c", "c", "anchor"],
                relations=[("c", "anchor", "depends_on")],
                source="s",
            )
        assert not ever_established(events, 5023, every)
        node_state = drive(events, 5023, every)
        assert node_state[0] == Maturity.DEVELOPING  # dense evidence still earns DEVELOPING

    def test_a_burst_of_thirty_across_two_windows_is_not_established(self):
        events, horizon = burst()
        assert not ever_established(events, horizon, 25)
        assert not ever_established(events, horizon, None)

    @pytest.mark.parametrize("cadence_T", [24, 168, 720])
    @pytest.mark.parametrize("every", [1, 2, 3])
    def test_heavily_disconfirmed_concepts_are_not_established(self, cadence_T, every):
        events, horizon = disconfirmed(every, T=cadence_T, uses=60)
        assert not ever_established(events, horizon, 50)

    @pytest.mark.parametrize("seed, dis", [(1, 0.5), (2, 0.5), (3, 0.3), (4, 0.7)])
    def test_random_remention_with_disconfirmations_is_not_established(self, seed, dis):
        events, horizon = random_noise(seed, dis)
        assert not ever_established(events, horizon, 250)
        assert not ever_established(events, horizon, None)

    def test_a_one_off_still_fades_and_is_pruned_after_the_grace(self, tmp_path):
        world = World(store_path=tmp_path)
        world.ingest(Observation(concepts=["once"], source="s"))
        node = world.concepts.resolve("once")
        world.clock.advance(60)
        world.reflect(light=True)
        assert node.maturity == Maturity.FADING
        world.clock.advance(700)
        world.reflect(light=True)
        assert world.concepts.resolve("once") is None  # 720-observation grace kept

    def test_disconfirmation_balance_gate(self):
        """Confirmations must outnumber disconfirmations about four to one."""
        for d, expected in ((0, True), (3, True), (5, True), (6, False), (12, False)):
            node = ConceptNode(name="x", maturity=Maturity.DEVELOPING, confidence=0.1)
            node.activation_count = 24
            node.recurrence_count = 24
            node.disconfirmation_count = d
            engine = LifecycleEngine(_Concepts(node), _NoRelations())
            engine.promote(node)
            assert (node.maturity == Maturity.ESTABLISHED) is expected, d


class TestEraForgettingAndRevival:
    def test_established_is_not_immortal_and_its_revival_re_earns_the_rung(self, tmp_path):
        world = World(store_path=tmp_path)
        for _ in range(30):
            world.ingest(use())
            world.clock.advance(23)
        node = world.concepts.resolve("c")
        assert node.maturity == Maturity.ESTABLISHED
        # Thirty spaced uses also consolidate it into long-term memory
        # (paper §3.6), so forgetting is on the slow curve: ~165 000 idle
        # observations instead of ~22 000 — slow, not immortal.
        assert node.long_term
        # Find the crossing on the read path (exact, Prop. 3.1'), then step
        # just past it: on the fast FADING tail the concept is prunable within
        # a few dozen ticks, so a coarse stepping would only see it gone.
        lo, hi = 0, 400_000
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if settled_confidence(node, node.last_activated_tick + mid) >= FADING_THRESHOLD:
                lo = mid
            else:
                hi = mid
        assert 100_000 < hi < 250_000, hi  # forgetting: the slow curve, still finite
        world.clock.advance(node.last_activated_tick + hi + 5 - world.clock.tick)
        world.reflect(light=True)
        node = world.concepts.resolve("c")
        assert node is not None and node.maturity == Maturity.FADING

        world.ingest(use())  # one mention revives the same node ...
        revived = world.concepts.resolve("c")
        assert revived.id == node.id
        assert revived.maturity == Maturity.DEVELOPING  # ... one rung up the ladder, not ESTABLISHED
        assert revived.recurrence_count == 1  # the stale chain is spent
        assert not revived.long_term  # ... and long-term memory is re-earned, not kept

        for _ in range(12):  # ... and it earns ESTABLISHED again by spaced use
            world.clock.advance(23)
            world.ingest(use())
        assert world.concepts.resolve("c").maturity == Maturity.ESTABLISHED

    def test_the_recurrence_chain_breaks_after_half_an_era_but_revival_uses_the_old_chain(self):
        node = ConceptNode(name="x", maturity=Maturity.FADING, confidence=0.02)
        node.activation_count = 20
        for k in range(5):
            node.activate(tick=k * 30)
        assert node.recurrence_count >= 3
        node.maturity = Maturity.FADING
        node.activate(tick=int(RECURRENCE_CHAIN_GAP) + 200)
        assert node.maturity == Maturity.DEVELOPING  # landed on the rung the old chain earned
        assert node.recurrence_count == 1

    def test_a_gap_below_the_chain_limit_keeps_counting(self):
        node = ConceptNode(name="x")
        for k in range(12):
            node.activate(tick=int(RECURRENCE_CHAIN_GAP) * k - 24 * k)
        assert node.recurrence_count == 12


# ═══════════════════════════════════════════════════════════════════════
# CORE tracks connection changes
# ═══════════════════════════════════════════════════════════════════════


class TestCoreTracksConnections:
    @staticmethod
    def _established_hub(tmp_path, uses: int = 40) -> World:
        world = World(store_path=tmp_path)
        for _ in range(uses):
            world.ingest(use())
            world.clock.advance(23)
        assert world.concepts.resolve("c").maturity == Maturity.ESTABLISHED
        return world

    def test_promotion_happens_at_the_connection_event_not_at_reflect(self, tmp_path):
        world = self._established_hub(tmp_path)
        node = world.concepts.resolve("c")
        need = core_connections_required(node.activation_count)
        assert len(world.relations.for_concept(node.id)) == 1  # the anchor
        for i in range(need - 2):
            world.ingest(link(f"p{i}"))
            assert node.maturity == Maturity.ESTABLISHED
        world.ingest(link("last"))
        assert len(world.relations.for_concept(node.id)) == need
        assert node.maturity == Maturity.CORE  # no reflect has run

    @pytest.mark.parametrize("reflect_between", [False, True])
    def test_dead_connections_do_not_count_whether_or_not_pruned(self, tmp_path, reflect_between):
        """Auto-discovered edges die in ~200 observations.  Linking one every
        400 observations must never reach the requirement, and a reflect
        (which physically prunes the dead ones) must not change that."""
        world = World(store_path=tmp_path)
        hub, _ = world.concepts.get_or_create("hub")
        hub.maturity = Maturity.ESTABLISHED
        hub.activation_count = 40
        hub.recurrence_count = 40
        hub.confidence = 0.5
        need = core_connections_required(hub.activation_count)
        for i in range(need + 2):
            other, _ = world.concepts.get_or_create(f"other{i}")
            world.relations.discover(hub.id, other.id, RelationType.PARALLEL, is_explicit=False)
            world.clock.advance(400)
            if reflect_between:
                world.reflect(light=True)
        assert hub.maturity == Maturity.ESTABLISHED
        world.reflect()
        assert hub.maturity == Maturity.ESTABLISHED

    def test_late_connection_state_is_cadence_independent(self):
        events, horizon = late_links()
        finals = [drive(events, horizon, every) for every in (25, 1000, None)]
        assert {m for m, _ in finals} == {Maturity.CORE}
        assert max(c for _, c in finals) - min(c for _, c in finals) < 1e-6


# ═══════════════════════════════════════════════════════════════════════
# Gates: legacy records, constants shared with the metacognition layer
# ═══════════════════════════════════════════════════════════════════════


class _Concepts:
    """Minimal ConceptStore double for gate checks on a hand-built node."""

    def __init__(self, node: ConceptNode) -> None:
        self._node = node

    def get(self, cid):
        return self._node

    def all(self):
        return [self._node]

    def update_maturity(self, cid, maturity):
        self._node.maturity = maturity

    def mark_dirty(self, cid):
        pass


class _NoRelations:
    def for_concept(self, cid):
        return []


class TestGates:
    def test_gate_constants_mirror_the_metacognition_lines(self):
        assert SPACED_DEVELOPING_EVIDENCE == TENTATIVE_EVIDENCE
        assert SPACED_ESTABLISHED_EVIDENCE == WELL_EVIDENCED

    def test_dense_established_needs_spacing_but_untracked_records_keep_the_old_rule(self):
        def ladder(rho: int) -> Maturity:
            node = ConceptNode(name="x", maturity=Maturity.DEVELOPING, confidence=0.7)
            node.activation_count = 12
            node.recurrence_count = rho
            LifecycleEngine(_Concepts(node), _NoRelations()).promote(node)
            return node.maturity

        assert ladder(1) == Maturity.DEVELOPING  # a burst is not durable
        assert ladder(2) == Maturity.DEVELOPING
        assert ladder(3) == Maturity.ESTABLISHED
        assert ladder(0) == Maturity.ESTABLISHED  # predates recurrence tracking

    def test_the_whole_ladder_is_climbed_in_one_evaluation(self, tmp_path):
        world = World(store_path=tmp_path)
        node, _ = world.concepts.get_or_create("legacy")
        node.activation_count = 40
        node.recurrence_count = 30
        node.confidence = 0.9
        for i in range(6):
            other, _ = world.concepts.get_or_create(f"n{i}")
            world.relations.discover(node.id, other.id)
        node.maturity = Maturity.EMBRYONIC
        world._lifecycle.evaluate()
        assert node.maturity == Maturity.CORE

    def test_an_established_concept_is_never_demoted_by_a_reflect(self, tmp_path):
        world = World(store_path=tmp_path)
        for _ in range(30):
            world.ingest(use())
            world.clock.advance(23)
        before = world.concepts.resolve("c").maturity
        world.clock.advance(500)
        for _ in range(3):
            world.reflect()
        assert world.concepts.resolve("c").maturity == before


# ═══════════════════════════════════════════════════════════════════════
# Determinism of the projection under equal evidence
# ═══════════════════════════════════════════════════════════════════════


class TestProjectionTiesIgnoreClockNoise:
    def test_two_concepts_with_equal_evidence_rank_by_recency_not_by_noise(self, tmp_path):
        """Identical concepts differ by ~1e-11 of wall-clock drift; the order
        of a tie must not follow that noise (it made projections flip with
        scheduling jitter and with the reflect cadence)."""
        world = World(store_path=tmp_path)
        world.ingest(Observation(concepts=["seed", "first", "second"],
                                 relations=[("seed", "first", "supports"),
                                            ("seed", "second", "supports")], source="s"))
        seed = world.concepts.resolve("seed")
        first, second = world.concepts.resolve("first"), world.concepts.resolve("second")
        engine = ProjectionEngine(world.concepts, world.relations, clock=world.clock)
        orders = set()
        for eps in (-3e-11, -1e-11, 0.0, 1e-11, 3e-11):
            acts = {seed.id: 1.0, first.id: 0.5 + eps, second.id: 0.5 - eps}
            proj = engine.project(acts, seed_ids=[seed.id], max_concepts=3)
            orders.add(tuple(c.name for c in proj.concepts))
        assert len(orders) == 1
        assert list(orders)[0][1] == "second"  # activated last → first among equals
