"""Task grounding — a task changes the projection through the concepts it
names, not only through task labels recorded on past observations.

Probe (analysis doc §7.17): in a world built from observations without
``task`` labels, *no* task string changed the projection — not even
``"kubernetes rollout"``, which names two concepts of one cluster —
because task affinity came only from ``task_profile`` history.  Grounded
affinity (named concept 1.0, its direct neighbours 0.5) now gives such a
task a structural foothold; history still applies where it exists.
"""

from __future__ import annotations

import pytest

from world0 import Observation, World
from world0.context import (
    GROUNDING_NEIGHBOR_SHARE,
    ground_task,
    name_coverage,
)

OPS = ["deployment", "kubernetes", "helm chart", "container registry", "rollout", "load balancer"]
ML = ["deployment", "pytorch", "gpu cluster", "training run", "checkpoint", "loss curve"]


def _two_clusters(world: World, *, task_labels: bool) -> World:
    for _ in range(12):
        world.ingest(Observation(
            concepts=OPS,
            relations=[("deployment", "kubernetes", "depends_on")],
            task="ops" if task_labels else "",
            source="s",
        ))
        world.ingest(Observation(
            concepts=ML,
            relations=[("deployment", "pytorch", "depends_on")],
            task="ml" if task_labels else "",
            source="s",
        ))
    world.reflect()
    return world


def _names(world: World, task: str, k: int = 5) -> list[str]:
    return [c.name for c in world.project(["deployment"], task=task, max_concepts=k).concepts]


@pytest.fixture
def unlabelled(tmp_path):
    return _two_clusters(World(store_path=tmp_path / "w"), task_labels=False)


class TestGroundedProjection:
    def test_task_naming_ops_concepts_selects_ops_cluster(self, unlabelled):
        names = _names(unlabelled, "kubernetes rollout")
        assert set(names[1:]) <= set(OPS)
        assert {"kubernetes", "rollout"} <= set(names)

    def test_task_naming_ml_concepts_selects_ml_cluster(self, unlabelled):
        names = _names(unlabelled, "pytorch training run")
        assert set(names[1:]) <= set(ML)

    def test_same_world_different_tasks_different_views(self, unlabelled):
        assert set(_names(unlabelled, "kubernetes rollout")).isdisjoint(
            set(_names(unlabelled, "pytorch checkpoint")) - {"deployment"}
        )

    def test_task_naming_nothing_changes_nothing(self, unlabelled):
        assert _names(unlabelled, "quarterly planning") == _names(unlabelled, "")

    def test_task_naming_only_the_seed_changes_nothing(self, unlabelled):
        assert _names(unlabelled, "deployment") == _names(unlabelled, "")

    def test_history_still_applies(self, tmp_path):
        world = _two_clusters(World(store_path=tmp_path / "w"), task_labels=True)
        assert set(_names(world, "ops")[1:]) <= set(OPS)
        assert set(_names(world, "ml")[1:]) <= set(ML)

    def test_cjk_task_grounds_on_cjk_names(self, tmp_path):
        world = World(store_path=tmp_path / "w")
        a = ["部署", "容器", "镜像仓库", "负载均衡", "滚动发布"]
        b = ["部署", "模型", "训练任务", "显卡集群", "损失曲线"]
        for _ in range(12):
            world.ingest(Observation(concepts=a, source="s"))
            world.ingest(Observation(concepts=b, source="s"))
        names = [c.name for c in world.project(["部署"], task="容器滚动发布", max_concepts=4).concepts]
        assert set(names[1:]) <= set(a)


class TestNameCoverage:
    def test_word_level_not_substring(self):
        assert name_coverage("ml", ["html"]) == 0.0

    def test_full_and_partial(self):
        assert name_coverage("kubernetes rollout", ["kubernetes"]) == 1.0
        assert name_coverage("helm", ["helm chart"]) == 0.5
        assert name_coverage("kubernetes rollout", ["helm chart"]) == 0.0

    def test_alias_counts(self):
        assert name_coverage("deploy to k8s", ["kubernetes", "k8s"]) == 1.0

    def test_cjk_containment(self):
        assert name_coverage("容器滚动发布", ["容器"]) == 1.0
        assert name_coverage("容器", ["滚动发布"]) == 0.0

    def test_empty_task(self):
        assert name_coverage("", ["kubernetes"]) == 0.0


class TestGroundTask:
    def test_anchor_neighbour_and_partial(self, unlabelled):
        ids = {c.name: c.id for c in unlabelled.concepts.all()}
        cands = list(ids.values())
        nbrs = {cid: set(unlabelled.relations.neighbors(cid)) for cid in cands}
        g = ground_task("kubernetes helm", cands, unlabelled.concepts, nbrs)
        assert g[ids["kubernetes"]] == 1.0
        assert g[ids["helm chart"]] == 0.5  # partially named
        assert g[ids["deployment"]] == GROUNDING_NEIGHBOR_SHARE  # neighbour of anchor
        assert ids["pytorch"] not in g or g[ids["pytorch"]] == GROUNDING_NEIGHBOR_SHARE

    def test_no_task_no_grounding(self, unlabelled):
        cands = [c.id for c in unlabelled.concepts.all()]
        assert ground_task("", cands, unlabelled.concepts, {}) == {}
