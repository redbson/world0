"""Behavioral tests for the cognitive-dynamics analysis.

Every test here corresponds to a finding in
``docs/world0-cognitive-dynamics-analysis.md`` and to an empirical probe
that *failed* on the previous implementation.  They pin down the
mathematical properties a continuously updating cognitive layer needs:

- time is cognitive time: one tick per observation, persisted with the
  world, with the calendar as a slow secondary drift only
- decay is a function of elapsed cognitive time, not of how often
  ``reflect()`` runs
- evidence buys persistence: a concept re-observed regularly can mature,
  a one-off mention still fades, and nothing is immortal
- semantic relation probability is evidence-driven, never time-driven
- task association is bounded in size and matched at word level
- activation rewards convergence, preserves distance ordering, cannot be
  inflated by cycles, and counts a recorded activation once
- projections are identical across processes (PYTHONHASHSEED-independent)
- Hebbian co-occurrence learning survives a restart
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from world0 import Observation, World
from world0.dynamics.activation import ActivationEngine
from world0.dynamics.decay import (
    CONCEPT_MAX_HALF_LIFE,
    FADING_THRESHOLD,
    concept_half_life,
    evidence_floor,
)
from world0.dynamics.hebbian import MAX_PAIRS
from world0.schemas.concept import (
    MAX_REINFORCEMENT_LOG,
    ConceptNode,
    Maturity,
    ReinforcementEntry,
    task_match_score,
)


@pytest.fixture
def world(tmp_path):
    return World(store_path=tmp_path / ".world0")


def _advance(world: World, ticks: int) -> None:
    """Let ``ticks`` observations pass without touching any concept.

    Time in World 0 is cognitive time (one tick per ingest), so simulating
    the passage of time is simply advancing the world clock.
    """
    world.clock.advance(int(ticks))


def _tick(world: World) -> None:
    world._decay.decay_concepts()
    world._decay.decay_relations()
    world._lifecycle.evaluate()


def _simulate_cadence(
    world: World, name: str, *, gap_ticks: int, total_ticks: int
) -> ConceptNode:
    """Re-observe ``name`` every ``gap_ticks`` observations for
    ``total_ticks`` observations, decaying and evaluating lifecycle between
    activations."""
    anchor = f"{name}_anchor"
    steps = int(total_ticks / gap_ticks)
    for _ in range(steps):
        world.ingest(
            Observation(
                concepts=[name, anchor],
                relations=[(name, anchor, "depends_on")],
                task="routine work",
                source="sim",
            )
        )
        # The ingest itself is one observation; gap_ticks - 1 more pass.
        _advance(world, gap_ticks - 1)
        _tick(world)
    return world.concepts.resolve(name)


# ═══════════════════════════════════════════════════════════════════════
# 1. Decay is a function of elapsed time
# ═══════════════════════════════════════════════════════════════════════


class TestDecayIdempotency:
    def _seed(self, world: World) -> None:
        for _ in range(5):
            world.ingest(
                Observation(
                    concepts=["alpha", "beta"],
                    relations=[("alpha", "beta", "depends_on")],
                    task="t",
                    source="s",
                )
            )
        world.concepts.resolve("alpha").maturity = Maturity.DEVELOPING

    def test_repeated_decay_at_the_same_instant_is_a_no_op(self, world):
        self._seed(world)
        _advance(world, 48)
        world._decay.decay_concepts()
        world._decay.decay_relations()
        once_c = world.concepts.resolve("alpha").confidence
        once_r = world.relations.all()[0].weight

        for _ in range(5):
            world._decay.decay_concepts()
            world._decay.decay_relations()

        assert world.concepts.resolve("alpha").confidence == pytest.approx(once_c)
        assert world.relations.all()[0].weight == pytest.approx(once_r)

    def test_decay_tracks_elapsed_time_not_call_count(self, tmp_path):
        split = World(store_path=tmp_path / "split")
        whole = World(store_path=tmp_path / "whole")
        self._seed(split)
        self._seed(whole)

        # 48 h applied as 2 × 24 h ...
        _advance(split, 24)
        _tick(split)
        _advance(split, 24)
        _tick(split)
        # ... versus 48 h applied once.
        _advance(whole, 48)
        _tick(whole)

        assert split.concepts.resolve("alpha").confidence == pytest.approx(
            whole.concepts.resolve("alpha").confidence, abs=1e-3
        )
        assert split.relations.all()[0].weight == pytest.approx(
            whole.relations.all()[0].weight, abs=1e-3
        )

    def test_grace_period_defers_but_does_not_lose_decay(self, world):
        self._seed(world)
        node = world.concepts.resolve("alpha")
        before = node.confidence
        world._decay.decay_concepts()  # same observation → skipped
        assert node.confidence == before
        assert node.last_decayed_tick is None
        _advance(world, 1)
        world._decay.decay_concepts()  # one observation later → applied
        assert node.confidence < before
        assert node.last_decayed_tick == world.clock.tick


# ═══════════════════════════════════════════════════════════════════════
# 2. Evidence buys persistence — the maturity ladder is reachable
# ═══════════════════════════════════════════════════════════════════════


class TestEvidenceAnchoredDecay:
    def test_concept_reobserved_every_24_observations_matures(self, world):
        # Before the fix such a concept sat at confidence ≈ 0.06 forever.
        node = _simulate_cadence(world, "habit", gap_ticks=24, total_ticks=2160)
        assert node.maturity in (Maturity.ESTABLISHED, Maturity.CORE)
        assert node.confidence >= 0.6

    def test_concept_reobserved_every_168_observations_matures_by_recurrence(self, world):
        # Before the fix such a concept was FADING with confidence 0.03; the
        # confidence equilibrium alone (≈0.3) would leave it DEVELOPING, and
        # the recurrence gate promotes it once it has recurred 10 times.
        node = _simulate_cadence(world, "weekly", gap_ticks=168, total_ticks=4368)
        assert node.maturity in (Maturity.DEVELOPING, Maturity.ESTABLISHED)
        assert node.confidence > 0.25
        assert node.recurrence_count == 26

    def test_concept_reobserved_every_720_observations_survives_on_floor(self, world):
        node = _simulate_cadence(world, "monthly", gap_ticks=720, total_ticks=8760)
        assert node.maturity != Maturity.EMBRYONIC
        assert node.confidence > FADING_THRESHOLD

    def test_one_shot_concept_still_fades(self, world):
        world.ingest(Observation(concepts=["one_off"], source="s"))
        node = world.concepts.resolve("one_off")
        _advance(world, 72)
        world._decay.decay_concepts()
        assert node.maturity == Maturity.FADING
        assert node.confidence < FADING_THRESHOLD

    def test_reflect_frequency_between_observations_is_irrelevant(self, world):
        """Ten reflects between two observations == one reflect."""
        self_seed = lambda w: [  # noqa: E731
            w.ingest(Observation(concepts=["alpha"], task="t", source="s"))
            for _ in range(5)
        ]
        self_seed(world)
        _advance(world, 40)
        for _ in range(10):
            world.reflect()
        many = world.concepts.resolve("alpha").confidence

        other = World(store_path=world._store._root.parent / "other")
        self_seed(other)
        _advance(other, 40)
        other.reflect()
        once = other.concepts.resolve("alpha").confidence
        assert many == pytest.approx(once, abs=1e-6)

    def test_burst_then_abandon_declines_slowly_but_surely(self, world):
        for _ in range(30):
            world.ingest(Observation(concepts=["burst"], task="t", source="s"))
        node = world.concepts.resolve("burst")
        node.maturity = Maturity.ESTABLISHED
        peak = node.confidence

        _advance(world, 720)  # 720 unrelated observations
        world._decay.decay_concepts()
        after_month = node.confidence
        assert after_month < peak
        assert after_month > 0.3, "30 confirmations must survive 720 observations"

        _advance(world, 8760)  # 8760 more
        world._decay.decay_concepts()
        after_year = node.confidence
        assert FADING_THRESHOLD < after_year < after_month

        _advance(world, 8760 * 2)  # and 17 520 more
        world._decay.decay_concepts()
        assert node.confidence < FADING_THRESHOLD, "nothing is immortal"
        assert node.maturity == Maturity.FADING

    def test_evidence_floor_grows_with_confirmation_and_shrinks_with_doubt(self):
        fresh = ConceptNode(name="x", activation_count=30)
        doubted = ConceptNode(name="y", activation_count=30, disconfirmation_count=20)
        weak = ConceptNode(name="z", activation_count=2)
        assert evidence_floor(fresh) > evidence_floor(doubted) > 0
        assert evidence_floor(weak) < FADING_THRESHOLD
        assert evidence_floor(ConceptNode(name="n")) == 0.0

    def test_half_life_scales_with_evidence_but_is_capped(self):
        single = ConceptNode(name="a", maturity=Maturity.DEVELOPING, activation_count=1)
        many = ConceptNode(name="b", maturity=Maturity.DEVELOPING, activation_count=40)
        core = ConceptNode(name="c", maturity=Maturity.CORE, activation_count=500)
        assert concept_half_life(single) == pytest.approx(168.0)
        assert concept_half_life(many) > concept_half_life(single)
        assert concept_half_life(core) == CONCEPT_MAX_HALF_LIFE


# ═══════════════════════════════════════════════════════════════════════
# 3. Relation probability is evidence-driven, not time-driven
# ═══════════════════════════════════════════════════════════════════════


class TestRelationProbability:
    def test_time_decay_leaves_probability_untouched(self, world):
        world.ingest(
            Observation(
                concepts=["p", "q"],
                relations=[("p", "q", "depends_on")],
                task="t",
                source="s",
            )
        )
        edge = world.relations.all()[0]
        probability = edge.probability
        weight = edge.weight
        _advance(world, 720)
        world._decay.decay_relations()
        assert edge.weight < weight * 0.1
        assert edge.probability == pytest.approx(probability)

    def test_disconfirmation_still_lowers_probability(self, world):
        world.ingest(
            Observation(
                concepts=["p", "q"],
                relations=[("p", "q", "depends_on")],
                task="t",
                source="s",
            )
        )
        edge = world.relations.all()[0]
        before = edge.probability
        world.relations.weaken(edge.id, provenance="t")
        assert edge.probability < before


# ═══════════════════════════════════════════════════════════════════════
# 4. Task association: bounded, complete, word-level
# ═══════════════════════════════════════════════════════════════════════


class TestTaskProfile:
    def test_log_is_bounded_while_profile_stays_complete(self, world):
        for i in range(200):
            world.ingest(
                Observation(concepts=["busy"], task=f"task {i % 7}", source="s")
            )
        node = world.concepts.resolve("busy")
        assert len(node.reinforcement_log) <= MAX_REINFORCEMENT_LOG
        assert sum(node.task_profile.values()) == 200
        assert len(node.task_profile) == 7
        assert node.task_affinity("task 3") == 1.0

    def test_task_match_is_word_level_not_substring(self):
        assert task_match_score("ml", "html parsing") == 0.0
        assert task_match_score("ml", "ml training") == 1.0
        assert task_match_score("ml serving", "ml training") == pytest.approx(0.5)
        assert task_match_score("ML Training", "ml training") == 1.0
        assert task_match_score("认知", "认知系统设计") == 1.0
        assert task_match_score("", "anything") == 0.0

    def test_legacy_record_backfills_profile_from_log(self):
        now = datetime.now(timezone.utc)
        node = ConceptNode.model_validate(
            {
                "name": "legacy",
                "reinforcement_log": [
                    ReinforcementEntry(timestamp=now, task="ML Training").model_dump(),
                    ReinforcementEntry(timestamp=now, task="ml training").model_dump(),
                    ReinforcementEntry(timestamp=now, task="ops").model_dump(),
                ],
            }
        )
        assert node.task_profile == {"ml training": 2, "ops": 1}

    def test_graded_task_boost(self):
        assert ActivationEngine._task_boost(0.0) == 1.0
        assert ActivationEngine._task_boost(0.5) == pytest.approx(1.25)
        assert ActivationEngine._task_boost(1.0) == pytest.approx(1.5)

    def test_merge_combines_task_profiles(self, world):
        world.ingest(Observation(concepts=["k"], task="alpha", source="s"))
        world.ingest(Observation(concepts=["k"], task="alpha", source="s"))
        world.ingest(Observation(concepts=["dup"], task="beta", source="s"))
        assert world.merge("k", "dup")
        keeper = world.concepts.resolve("k")
        assert keeper.task_profile == {"alpha": 2, "beta": 1}


# ═══════════════════════════════════════════════════════════════════════
# 5. Activation: convergence, ordering, cycles, recording
# ═══════════════════════════════════════════════════════════════════════


class TestActivationAggregation:
    def _ids(self, world, names):
        return {n: world.concepts.resolve(n).id for n in names}

    def test_convergence_outscores_single_path(self, world):
        for _ in range(3):
            world.ingest(
                Observation(
                    concepts=["A", "B", "C", "D"],
                    relations=[
                        ("A", "C", "depends_on"),
                        ("B", "C", "depends_on"),
                        ("A", "D", "depends_on"),
                    ],
                    task="t",
                    source="s",
                )
            )
        ids = self._ids(world, "ABCD")
        act = world._activation.activate(
            [ids["A"], ids["B"]], max_depth=1, decay=0.5, record=False
        )
        assert act[ids["C"]] > act[ids["D"]]

    def test_no_propagated_concept_outscores_strongest_seed(self, world):
        seeds = [f"s{i}" for i in range(6)]
        for _ in range(5):
            world.ingest(
                Observation(
                    concepts=seeds + ["target"],
                    relations=[(s, "target", "depends_on") for s in seeds],
                    task="t",
                    source="s",
                )
            )
        ids = self._ids(world, seeds + ["target"])
        act = world._activation.activate(
            [ids[s] for s in seeds], max_depth=1, decay=0.6, record=False
        )
        strongest_seed = max(act[ids[s]] for s in seeds)
        assert act[ids["target"]] <= strongest_seed + 1e-12
        # ...but convergence from six seeds still lifts it far above a
        # single-path score.
        single = world._activation.activate(
            [ids["s0"]], max_depth=1, decay=0.6, record=False
        )
        assert act[ids["target"]] > 2 * single[ids["target"]]

    def test_distance_ordering_is_strict_even_inside_the_floor_band(self, world):
        names = [f"c{i}" for i in range(7)]
        for _ in range(10):
            for a, b in zip(names, names[1:]):
                world.ingest(
                    Observation(
                        concepts=[a, b],
                        relations=[(a, b, "depends_on")],
                        task="corridor",
                        source="s",
                    )
                )
        ids = self._ids(world, names)
        act = world._activation.activate(
            [ids["c0"]], max_depth=6, decay=0.5, record=False
        )
        scores = [act[ids[n]] for n in names]
        assert all(a > b for a, b in zip(scores, scores[1:])), scores
        assert len(set(round(s, 12) for s in scores)) == len(scores)

    def test_floor_band_preserves_rank(self):
        floor = 0.02
        lifted = [ActivationEngine._apply_floor(r, floor) for r in (1e-6, 1e-4, 1e-3, 0.01, 0.019)]
        assert all(a < b for a, b in zip(lifted, lifted[1:]))
        assert lifted[0] >= 0.9 * floor
        assert lifted[-1] < floor
        assert ActivationEngine._apply_floor(0.05, floor) == 0.05
        assert ActivationEngine._apply_floor(0.0, floor) == 0.0

    def test_cycle_does_not_inflate_seed(self, world):
        for _ in range(5):
            world.ingest(
                Observation(
                    concepts=["A", "B", "C"],
                    relations=[
                        ("A", "B", "depends_on"),
                        ("B", "C", "depends_on"),
                        ("C", "A", "depends_on"),
                    ],
                    task="t",
                    source="s",
                )
            )
        ids = self._ids(world, "ABC")
        seed_conf = world.concepts.get(ids["A"]).confidence
        act = world._activation.activate(
            [ids["A"]], max_depth=4, decay=0.6, record=False
        )
        assert act[ids["A"]] == pytest.approx(seed_conf)
        assert act[ids["A"]] > act[ids["B"]] > 0
        assert act[ids["A"]] > act[ids["C"]] > 0

    def test_recorded_activation_counts_once_per_concept(self, world):
        for _ in range(3):
            world.ingest(
                Observation(
                    concepts=["A", "B", "C"],
                    relations=[("A", "C", "depends_on"), ("B", "C", "depends_on")],
                    task="t",
                    source="s",
                )
            )
        ids = self._ids(world, "ABC")
        before = world.concepts.get(ids["C"]).activation_count
        world._activation.activate(
            [ids["A"], ids["B"]], max_depth=2, decay=0.6, record=True
        )
        assert world.concepts.get(ids["C"]).activation_count == before + 1


# ═══════════════════════════════════════════════════════════════════════
# 6. Projection determinism across processes
# ═══════════════════════════════════════════════════════════════════════


_CHILD = r"""
import sys
from world0 import World
w = World(store_path=sys.argv[1])
p = w.project(["hub"], task="star", max_concepts=4, max_depth=2)
print(p.render())
"""


class TestProjectionDeterminism:
    def test_projection_is_identical_across_hash_seeds(self, tmp_path):
        root = tmp_path / "det"
        world = World(store_path=root)
        leaves = [f"leaf{i:02d}" for i in range(12)]
        for _ in range(10):
            world.ingest(
                Observation(
                    concepts=["hub"] + leaves,
                    relations=[("hub", leaf, "depends_on") for leaf in leaves],
                    task="star",
                    source="s",
                )
            )
        outputs = set()
        for seed in (0, 1, 2, 3):
            env = dict(os.environ, PYTHONHASHSEED=str(seed))
            completed = subprocess.run(
                [sys.executable, "-c", _CHILD, str(root)],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            outputs.add(completed.stdout)
        assert len(outputs) == 1, "projection must not depend on PYTHONHASHSEED"

    def test_projection_lists_concepts_in_selection_order(self, world):
        for _ in range(5):
            world.ingest(
                Observation(
                    concepts=["hub", "x", "y", "z"],
                    relations=[("hub", n, "depends_on") for n in "xyz"],
                    task="t",
                    source="s",
                )
            )
        first = world.project(["hub"], task="t", max_concepts=3)
        second = world.project(["hub"], task="t", max_concepts=3)
        assert [c.id for c in first.concepts] == [c.id for c in second.concepts]
        assert first.concepts[0].name == "hub"


# ═══════════════════════════════════════════════════════════════════════
# 7. Cognitive clock: time is counted in observations
# ═══════════════════════════════════════════════════════════════════════


class TestCognitiveClock:
    def test_ingest_advances_clock_and_reflect_does_not(self, world):
        assert world.clock.tick == 0
        world.ingest(Observation(concepts=["a"], source="s"))
        world.ingest(Observation(concepts=["b"], source="s"))
        assert world.clock.tick == 2
        world.reflect()
        assert world.clock.tick == 2
        assert world.status().cognitive_tick == 2

    def test_clock_persists_across_restart(self, tmp_path):
        root = tmp_path / "clock"
        first = World(store_path=root)
        for _ in range(7):
            first.ingest(Observation(concepts=["a"], source="s"))
        second = World(store_path=root)
        assert second.clock.tick == 7
        node = second.concepts.resolve("a")
        assert node.last_activated_tick == 7
        assert node.created_tick == 1

    def test_records_are_stamped_with_ticks(self, world):
        world.ingest(Observation(concepts=["x"], source="s"))
        world.ingest(Observation(concepts=["y"], source="s"))
        world.ingest(
            Observation(concepts=["x", "y"], relations=[("x", "y", "depends_on")], source="s")
        )
        x = world.concepts.resolve("x")
        assert x.created_tick == 1
        assert x.last_activated_tick == 3
        edge = world.relations.all()[0]
        assert edge.discovered_tick == 3
        assert edge.last_reinforced_tick == 3

    def test_decay_is_driven_by_observations_not_calendar(self, world):
        world.ingest(Observation(concepts=["a"], source="s"))
        node = world.concepts.resolve("a")
        before = node.confidence
        # A week of wall-clock time with no observations: only drift.
        node.last_activated = datetime.now(timezone.utc) - timedelta(days=7)
        world._decay.decay_concepts()
        after_idle_week = node.confidence
        assert before * 0.5 < after_idle_week < before

        other = World(store_path=world._store._root.parent / "busy")
        other.ingest(Observation(concepts=["a"], source="s"))
        busy = other.concepts.resolve("a")
        _advance(other, 168)  # a week's worth of observations
        other._decay.decay_concepts()
        assert busy.confidence < after_idle_week

    def test_cognitive_elapsed_clamps_negative_components(self):
        from world0.schemas.clock import cognitive_elapsed

        now = datetime.now(timezone.utc)
        assert cognitive_elapsed(5, 9, now, now + timedelta(hours=3)) == 0.0
        assert cognitive_elapsed(9, 5, now, now) == 4.0
        assert cognitive_elapsed(9, 5, now, now - timedelta(hours=10)) == pytest.approx(5.0)

    def test_clock_cannot_move_backwards(self):
        from world0.schemas.clock import CognitiveClock

        clock = CognitiveClock(3)
        assert clock.advance(2) == 5
        with pytest.raises(ValueError):
            clock.advance(-1)


# ═══════════════════════════════════════════════════════════════════════
# 8. Hebbian learning survives a restart
# ═══════════════════════════════════════════════════════════════════════


class TestHebbianPersistence:
    def test_pending_pairs_survive_restart(self, tmp_path):
        root = tmp_path / "heb"
        first = World(store_path=root)
        first.ingest(Observation(concepts=["x", "y"], task="t", source="s"))
        assert first._hebbian.pending_pairs == 1

        second = World(store_path=root)
        assert second._hebbian.pending_pairs == 1
        result = second.ingest(Observation(concepts=["x", "y"], task="t", source="s"))
        assert result.hebbian_relations == ["x ↔ y"]
        assert second._hebbian.pending_pairs == 0

        third = World(store_path=root)
        assert third._hebbian.pending_pairs == 0

    def test_snapshot_roundtrip(self, world):
        world.ingest(Observation(concepts=["a", "b", "c"], task="t", source="s"))
        snapshot = world._hebbian.snapshot()
        assert len(snapshot) == 3
        world._hebbian.restore({})
        assert world._hebbian.pending_pairs == 0
        world._hebbian.restore(snapshot)
        assert world._hebbian.snapshot() == snapshot
        world._hebbian.restore({"bad": 1, "x|y": "nope", "p|q": 0})
        assert world._hebbian.pending_pairs == 0

    def test_pairs_follow_observation_salience_order(self, world):
        names = [f"n{i:02d}" for i in range(21)]  # 210 pairs > MAX_PAIRS
        world.ingest(Observation(concepts=names, task="t", source="s"))
        snapshot = world._hebbian.snapshot()
        assert len(snapshot) == MAX_PAIRS
        ids = {n: world.concepts.resolve(n).id for n in names}
        first = ids["n00"]
        assert all(first in key.split("|") for key in list(snapshot)[:20])
        last_pair = "|".join(sorted((ids["n19"], ids["n20"])))
        assert last_pair not in snapshot
