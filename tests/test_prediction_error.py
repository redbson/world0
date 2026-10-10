"""Prediction error on ingest (docs/mc/04-prediction.md, indicator PP-1).

Baseline probe: ``IngestResult`` carried no signal of how expected an
observation was.  ``IngestResult.prediction`` now scores every observation
against learned co-occurrence *before* learning it: strong companions that
failed to appear (relative to the absences the model itself expects) and
well-known concepts meeting for the first time (weighted by how likely they
were to have met already).
"""

from __future__ import annotations

import random
import statistics

from world0 import Observation, World
from world0.dynamics.hebbian import PREDICTION_MIN_SUPPORT

DATA = ["pipeline", "etl job", "warehouse", "schema", "batch window"]
OPS = ["kubernetes", "helm chart", "rollout"]
ML = ["pipeline", "training run", "checkpoint", "gpu", "loss curve"]


def _two_clusters(path) -> World:
    w = World(store_path=path)
    for _ in range(20):
        w.ingest(Observation(concepts=DATA, source="s"))
        w.ingest(Observation(concepts=OPS, source="s"))
    return w


def _pe(w: World, concepts):
    return w.ingest(Observation(concepts=concepts, source="s")).prediction


class TestDepartures:
    def test_routine_observation_is_unsurprising(self, tmp_path):
        pe = _pe(_two_clusters(tmp_path / "w"), DATA)
        assert pe.surprise == 0.0 and not pe.missing and not pe.novel_pairs

    def test_missing_constant_companions(self, tmp_path):
        pe = _pe(_two_clusters(tmp_path / "w"), ["etl job"])
        assert pe.missing_ratio == 1.0
        assert {m[1] for m in pe.missing} == set(DATA) - {"etl job"}
        assert all(m[0] == "etl job" and m[2] == 1.0 for m in pe.missing)

    def test_novel_combination_of_known_concepts(self, tmp_path):
        w = _two_clusters(tmp_path / "w")
        pe = _pe(w, ["etl job", "rollout"])
        assert ("etl job", "rollout") in pe.novel_pairs and pe.novelty > 0.9
        # Scored before learning: the second time they are no longer new.
        assert _pe(w, ["etl job", "rollout"]).novel_pairs == []

    def test_unknown_concepts_make_no_predictions(self, tmp_path):
        w = World(store_path=tmp_path / "w")
        for _ in range(PREDICTION_MIN_SUPPORT - 2):
            w.ingest(Observation(concepts=["a", "b"], source="s"))
        pe = _pe(w, ["a", "c"])
        assert pe.surprise == 0.0


class TestCalibration:
    def _mean_surprise(self, path, structured: bool) -> float:
        rng = random.Random(7)
        w = World(store_path=path)
        topics = [[f"t{t}c{i}" for i in range(6)] for t in range(10)]
        pool = [c for t in topics for c in t]
        s = []
        for _ in range(400):
            obs = rng.sample(rng.choice(topics), 4) if structured else rng.sample(pool, 4)
            s.append(_pe(w, obs).surprise)
        return statistics.mean(s[200:])

    def test_loose_cluster_variability_is_not_surprise(self, tmp_path):
        structured = self._mean_surprise(tmp_path / "s", True)
        random_world = self._mean_surprise(tmp_path / "r", False)
        assert structured < 0.2 and structured < random_world

    def test_drift_raises_then_settles(self, tmp_path):
        rng = random.Random(7)
        w = World(store_path=tmp_path / "w")
        s = []
        for t in range(300):
            cluster = DATA if t < 150 else ML
            s.append(_pe(w, ["pipeline", *rng.sample(cluster[1:], 3)]).surprise)
        before, spike, after = (statistics.mean(s[a:b]) for a, b in ((100, 150), (150, 175), (250, 300)))
        assert spike > 0.3 and spike > 5 * before and after < spike / 3


class TestPersistence:
    def test_linked_statistics_survive_restart(self, tmp_path):
        path = tmp_path / "w"
        with _two_clusters(path) as w:
            before = w._hebbian.prediction_error([w.concepts.resolve("etl job").id])
        again = World(store_path=path)
        after = again._hebbian.prediction_error([again.concepts.resolve("etl job").id])
        assert after == before and after.missing_ratio == 1.0

    def test_forget_concept_drops_linked_counts(self, tmp_path):
        w = _two_clusters(tmp_path / "w")
        etl = w.concepts.resolve("etl job").id
        w._hebbian.forget_concept(etl)
        assert all(etl not in key for key in w._hebbian.stats_snapshot()["linked"])
