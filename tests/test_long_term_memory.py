"""Long-term memory: past a consolidation gate a concept forgets on a very
slow curve (docs/paper §3.6, analysis doc §7.34).

The gate — well evidenced, recurring in spaced windows, uncontested — is
made of event-time counters, so it turns true only at an activation; the
slow curve is then the state at the start of every gap, which keeps the
settle operator an exact semigroup and the world's state a function of
the observation stream (Theorems 3.2 / 3.7 unchanged).
"""

from __future__ import annotations

import random

import pytest

from world0 import Observation, World
from world0.dynamics.decay import (
    FADING_THRESHOLD,
    LONG_TERM_HALF_LIFE,
    concept_half_life,
    concept_prunable,
    relax_confidence,
    settled_confidence,
)
from world0.dynamics.lifecycle import (
    LONG_TERM_BALANCE,
    LONG_TERM_RECURRENCE,
    consolidation_gate,
)
from world0.schemas.concept import LONG_TERM_ERA_HL, SALIENCE_ERA_HL, ConceptNode, Maturity


def _use(task="t"):
    return Observation(concepts=["c", "d"], relations=[("c", "d", "depends_on")], task=task, source="s")


def _spaced(world: World, uses: int, gap: int) -> int | None:
    """Ingest ``uses`` mentions ``gap`` ticks apart; the use at which ``c`` consolidated."""
    at = None
    for i in range(uses):
        if i:
            world.clock.advance(gap - 1)
        r = world.ingest(_use())
        if "c" in r.consolidated_concepts and at is None:
            at = i + 1
    return at


@pytest.fixture
def world(tmp_path):
    w = World(store_path=tmp_path / ".world0")
    yield w
    w.close()


class TestGate:
    def test_spaced_well_evidenced_use_consolidates_at_the_event(self, world):
        at = _spaced(world, 14, 24)
        c = world.concepts.resolve("c")
        assert at is not None and c.long_term and c.consolidated_tick is not None
        assert c.recurrence_count >= LONG_TERM_RECURRENCE and c.evidence() >= 0.5
        assert world.card("c").long_term is True
        assert world.status().long_term_concepts == 2  # c and d

    def test_a_burst_does_not_consolidate(self, world):
        assert _spaced(world, 30, 1) is None  # thirty mentions, one window
        assert not world.concepts.resolve("c").long_term

    def test_mode_off_never_consolidates(self, tmp_path):
        w = World(store_path=tmp_path / "off", long_term_memory=False)
        assert _spaced(w, 20, 24) is None
        assert not w.concepts.resolve("c").long_term and w.status().long_term_concepts == 0
        w.close()

    def test_gate_turns_true_only_at_an_event(self):
        node = ConceptNode(name="x")
        for i in range(LONG_TERM_RECURRENCE + 6):
            node.activate(tick=24 * (i + 1))
        node.recurrence_count = LONG_TERM_RECURRENCE - 1  # one spaced window short
        assert not consolidation_gate(node)
        node.last_activated_tick += 100_000  # time alone changes nothing
        assert not consolidation_gate(node)
        node.activate(tick=node.last_activated_tick + 24)
        assert consolidation_gate(node)

    def test_contested_concept_does_not_consolidate_and_a_denied_one_leaves(self, world):
        _spaced(world, 14, 24)
        c = world.concepts.resolve("c")
        assert c.long_term
        while world.concepts.resolve("c").evidence_balance() >= LONG_TERM_BALANCE:
            world.weaken("c", task="t")
        c = world.concepts.resolve("c")
        assert not c.long_term and c.consolidated_tick is None
        assert concept_half_life(c) < LONG_TERM_HALF_LIFE
        # it can re-earn the mode with further spaced, uncontested use
        for _ in range(30):
            world.clock.advance(23)
            world.ingest(_use())
        assert world.concepts.resolve("c").long_term

    def test_revival_from_fading_clears_the_mode(self):
        node = ConceptNode(name="x", maturity=Maturity.FADING, confidence=0.01, consolidated_tick=5)
        assert not node.long_term  # fading: already on the fast curve
        node.activate(tick=10)
        assert node.consolidated_tick is None and not node.long_term

    def test_legacy_record_loads_without_the_field(self):
        node = ConceptNode.model_validate({"name": "old", "maturity": "established"})
        assert node.consolidated_tick is None and not node.long_term


class TestSlowCurve:
    def test_long_term_concept_outlives_its_burst_twin(self, tmp_path):
        slow = World(store_path=tmp_path / "slow")
        fast = World(store_path=tmp_path / "fast")
        _spaced(slow, 12, 24)
        _spaced(fast, 12, 1)
        c, b = slow.concepts.resolve("c"), fast.concepts.resolve("c")
        assert c.long_term and not b.long_term
        assert c.activation_count == b.activation_count == 12
        assert concept_half_life(c) == LONG_TERM_HALF_LIFE
        for idle in (5_000, 20_000, 60_000):
            lt = settled_confidence(c, c.last_activated_tick + idle)
            bt = settled_confidence(b, b.last_activated_tick + idle)
            assert lt > bt, idle
        assert settled_confidence(c, c.last_activated_tick + 20_000) > 0.2
        assert settled_confidence(b, b.last_activated_tick + 20_000) < FADING_THRESHOLD
        # slow, not flat: it does fade eventually and is then prunable
        far = c.last_activated_tick + 200_000
        assert settled_confidence(c, far) < FADING_THRESHOLD
        slow.close(); fast.close()

    def test_salience_persists_on_the_long_era(self):
        node = ConceptNode(name="x", maturity=Maturity.ESTABLISHED, confidence=0.6)
        for i in range(20):
            node.activate(tick=24 * (i + 1))
        plain = node.salience(now_tick=node.last_activated_tick + 20_000)
        node.consolidated_tick = node.last_activated_tick
        assert node.salience(now_tick=node.last_activated_tick + 20_000) > plain

    def test_settle_is_an_exact_semigroup_on_the_long_era(self):
        rng = random.Random(7)
        for _ in range(2000):
            c, f = rng.uniform(0.05, 1.0), rng.uniform(0.0, 0.3)
            t1, t2 = rng.uniform(0, 40_000), rng.uniform(0, 40_000)
            whole = relax_confidence(c, f, LONG_TERM_HALF_LIFE, t1 + t2, LONG_TERM_ERA_HL)
            split = relax_confidence(
                relax_confidence(c, f, LONG_TERM_HALF_LIFE, t1, LONG_TERM_ERA_HL),
                f * 2 ** (-t1 / LONG_TERM_ERA_HL), LONG_TERM_HALF_LIFE, t2, LONG_TERM_ERA_HL)
            assert abs(whole - split) < 1e-9

    def test_crossing_into_fading_switches_back_to_the_fast_curve(self):
        node = ConceptNode(name="x", maturity=Maturity.ESTABLISHED, confidence=0.6)
        for i in range(12):
            node.activate(tick=24 * (i + 1))
        node.confidence = 0.6
        node.consolidated_tick = node.last_activated_tick
        t0 = node.last_activated_tick
        lo, hi = 0, 400_000  # the crossing instant, by bisection on the monotone curve
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if settled_confidence(node, t0 + mid) >= FADING_THRESHOLD:
                lo = mid
            else:
                hi = mid
        assert 100_000 < hi < 200_000, hi  # the slow curve
        # after the crossing the FADING half-life (24) applies: the tail falls
        # to the (era-decaying) floor at once instead of taking another
        # 35 000 ticks; on the slow curve 2 000 ticks would leave ~0.048
        tail = settled_confidence(node, t0 + hi + 2_000)
        assert tail < 0.01, tail
        assert tail < relax_confidence(FADING_THRESHOLD, 0.0, LONG_TERM_HALF_LIFE, 2_000, LONG_TERM_ERA_HL) / 4


class TestScheduleIndependence:
    @staticmethod
    def _drive(path, every, uses=12, gap=48, idle=30_000):
        w = World(store_path=path)
        for i in range(uses):
            if i:
                for _ in range(gap - 1):
                    w.clock.advance(1)
                    if every and w.clock.tick % every == 0:
                        w.reflect(light=True)
            w.ingest(_use())
        for _ in range(idle):
            w.clock.advance(1)
            if every and w.clock.tick % every == 0:
                w.reflect(light=True)
        w.reflect()
        c = w.concepts.resolve("c")
        state = None if c is None else (c.maturity.value, round(c.confidence, 6), c.activation_count,
                                        c.consolidated_tick, c.long_term)
        w.close()
        return state

    def test_consolidation_and_the_slow_curve_do_not_depend_on_reflect_cadence(self, tmp_path):
        states = [self._drive(tmp_path / str(i), every) for i, every in enumerate((1, 50, None))]
        assert states[0] is not None and states[0][4] is True  # alive, long-term, after 30 000 idle
        assert states[0] == states[1] == states[2], states


class TestReporting:
    def test_reflect_reports_catch_up_consolidations(self, world):
        _spaced(world, 14, 24)
        c = world.concepts.resolve("c")
        c.consolidated_tick = None  # a record written before the mode existed
        world.concepts.mark_dirty(c.id)
        r = world.reflect()
        assert c.id in r.consolidated_concepts
        assert world.concepts.resolve("c").long_term

    def test_full_render_marks_long_term(self, world):
        _spaced(world, 14, 24)
        text = world.project(["c"], task="t").render(style="full")
        assert "long-term" in text
