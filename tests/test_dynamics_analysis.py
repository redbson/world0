"""Behavioral tests for the cognitive-dynamics analysis.

Every test here corresponds to a finding in
``docs/world0-cognitive-dynamics-analysis.md`` and to an empirical probe
that *failed* on the previous implementation.  They pin down the
mathematical properties a continuously updating cognitive layer needs:

- decay is a function of elapsed time, not of how often ``reflect()`` runs
- evidence buys persistence: a concept used daily can mature, a one-off
  mention still fades, and nothing is immortal
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


def _advance(world: World, hours: float) -> None:
    """Simulate ``hours`` of wall-clock time passing for every record.

    Every timestamp the dynamics read is shifted into the past, including
    ``last_decayed_at`` — otherwise the decay reference would stay "now"
    and no interval would elapse.
    """
    delta = timedelta(hours=hours)
    for node in world.concepts.all():
        node.last_activated -= delta
        if node.last_decayed_at:
            node.last_decayed_at -= delta
    for edge in world.relations.all():
        edge.last_reinforced -= delta
        if edge.last_decayed_at:
            edge.last_decayed_at -= delta


def _tick(world: World) -> None:
    world._decay.decay_concepts()
    world._decay.decay_relations()
    world._lifecycle.evaluate()


def _simulate_cadence(
    world: World, name: str, *, gap_hours: float, days: int
) -> ConceptNode:
    """Activate ``name`` every ``gap_hours`` for ``days`` days, decaying
    and evaluating lifecycle between activations."""
    anchor = f"{name}_anchor"
    steps = int(days * 24 / gap_hours)
    for _ in range(steps):
        world.ingest(
            Observation(
                concepts=[name, anchor],
                relations=[(name, anchor, "depends_on")],
                task="routine work",
                source="sim",
            )
        )
        _advance(world, gap_hours)
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
        _advance(world, 0.5)
        world._decay.decay_concepts()  # inside grace → skipped
        assert node.confidence == before
        assert node.last_decayed_at is None
        _advance(world, 0.6)
        world._decay.decay_concepts()  # 1.1 h total now applied
        assert node.confidence < before


# ═══════════════════════════════════════════════════════════════════════
# 2. Evidence buys persistence — the maturity ladder is reachable
# ═══════════════════════════════════════════════════════════════════════


class TestEvidenceAnchoredDecay:
    def test_daily_used_concept_matures(self, world):
        # Before the fix a daily concept sat at confidence ≈ 0.06 forever.
        node = _simulate_cadence(world, "habit", gap_hours=24, days=90)
        assert node.maturity in (Maturity.ESTABLISHED, Maturity.CORE)
        assert node.confidence >= 0.6

    def test_weekly_used_concept_settles_as_developing(self, world):
        # Before the fix a weekly concept was FADING with confidence 0.03.
        node = _simulate_cadence(world, "weekly", gap_hours=168, days=182)
        assert node.maturity == Maturity.DEVELOPING
        assert node.confidence > 0.25

    def test_monthly_used_concept_survives_on_its_evidence_floor(self, world):
        node = _simulate_cadence(world, "monthly", gap_hours=720, days=365)
        assert node.maturity != Maturity.EMBRYONIC
        assert node.confidence > FADING_THRESHOLD

    def test_one_shot_concept_still_fades(self, world):
        world.ingest(Observation(concepts=["one_off"], source="s"))
        node = world.concepts.resolve("one_off")
        _advance(world, 72)
        world._decay.decay_concepts()
        assert node.maturity == Maturity.FADING
        assert node.confidence < FADING_THRESHOLD

    def test_burst_then_abandon_declines_slowly_but_surely(self, world):
        for _ in range(30):
            world.ingest(Observation(concepts=["burst"], task="t", source="s"))
        node = world.concepts.resolve("burst")
        node.maturity = Maturity.ESTABLISHED
        peak = node.confidence

        _advance(world, 24 * 30)  # one silent month
        world._decay.decay_concepts()
        after_month = node.confidence
        assert after_month < peak
        assert after_month > 0.3, "30 confirmations must survive a month"

        _advance(world, 24 * 365)  # a silent year
        world._decay.decay_concepts()
        after_year = node.confidence
        assert FADING_THRESHOLD < after_year < after_month

        _advance(world, 24 * 365 * 2)  # two more silent years
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
        _advance(world, 24 * 30)
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
# 7. Hebbian learning survives a restart
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
