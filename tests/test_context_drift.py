"""Context under concept drift — the same concept used in two domains.

``pipeline`` is used in a data-engineering sense for 150 observations and
then in an ML sense for 150 more.  Afterwards the world holds both
neighbourhoods; which one a projection shows must follow the task
context (AGENTS.md Rule 4: context changes relevance), not only recency.

Probe (analysis doc §7.12): before the fix, ``task="data eng"`` returned
four data-engineering concepts plus ``checkpoint`` in second place — MMR
judged the data-engineering clique fully redundant with itself and
bought "diversity" from the off-task cluster.  Off-task candidates are
now treated as redundant with the task itself, so they enter only when
their raw relevance beats an on-task candidate's by more than the task
discount.
"""

from __future__ import annotations

import random

import pytest

from world0 import Observation, Perspective, World

DE = ["pipeline", "ETL", "warehouse", "Airflow", "Spark", "schema"]
ML = ["pipeline", "training", "model", "GPU", "dataset", "checkpoint"]


def _names(projection) -> list[str]:
    return [c.name for c in projection.concepts]


@pytest.fixture
def drifted(tmp_path):
    rng = random.Random(3)
    w = World(store_path=tmp_path / ".world0")
    for _ in range(150):
        extra = ["pipeline"] if rng.random() < 0.5 else []
        w.ingest(Observation(concepts=rng.sample(DE, 4) + extra, task="data eng", domain="data", source="s"))
    for _ in range(150):
        extra = ["pipeline"] if rng.random() < 0.5 else []
        w.ingest(Observation(concepts=rng.sample(ML, 4) + extra, task="ml", domain="ml", source="s"))
    return w


class TestContextUnderDrift:
    def test_task_selects_the_matching_neighbourhood(self, drifted):
        de = _names(drifted.project(["pipeline"], task="data eng", max_concepts=6))
        ml = _names(drifted.project(["pipeline"], task="ml", max_concepts=6))
        assert set(de[1:]) <= set(DE) - {"pipeline"}
        assert set(ml[1:]) <= set(ML) - {"pipeline"}
        assert len(de) == len(ml) == 6

    def test_without_task_recency_wins(self, drifted):
        recent = _names(drifted.project(["pipeline"], max_concepts=6))
        assert sum(n in ML for n in recent[1:]) >= 3

    def test_domain_perspective_agrees_with_task(self, drifted):
        p = Perspective(task="data eng", active_domains=["data"])
        de = _names(drifted.project(["pipeline"], perspective=p, max_concepts=6))
        assert set(de[1:]) <= set(DE) - {"pipeline"}

    def test_drifted_concept_keeps_both_domains(self, drifted):
        node = drifted.concepts.resolve("pipeline")
        assert node.domain_profile.get("data", 0) > 0
        assert node.domain_profile.get("ml", 0) > 0
        assert node.task_affinity("data eng") == 1.0
        assert node.task_affinity("ml") == 1.0
