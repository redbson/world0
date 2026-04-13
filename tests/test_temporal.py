"""Temporal dimension tests — validates that time freshness affects
activation propagation, projection selection, and overall cognitive
behavior.

Tests simulate time passage by manipulating last_activated and
last_reinforced timestamps rather than actually waiting.

Sections:
  1. Temporal Relevance Unit Tests — ConceptNode / RelationEdge methods
  2. Activation with Time Decay — fresh vs stale propagation
  3. Projection Temporal Preference — fresh concepts preferred in MMR
  4. Multi-Session Temporal Evolution — realistic time-passing scenario
  5. Temporal + Task Interaction — time and task affinity combined
  6. Edge Cases — zero-age, extreme age, clock boundaries
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from world0 import Observation, World
from world0.schemas.concept import ConceptNode, Maturity
from world0.schemas.relation import RelationEdge, RelationType


@pytest.fixture
def world(tmp_path):
    return World(store_path=tmp_path / ".world0")


# ═══════════════════════════════════════════════════════════════════════
# 1. Temporal Relevance Unit Tests
# ═══════════════════════════════════════════════════════════════════════

class TestTemporalRelevanceUnit:
    """Unit tests for temporal_relevance() on ConceptNode and RelationEdge."""

    def test_concept_just_activated_is_1(self):
        node = ConceptNode(name="fresh")
        node.last_activated = datetime.now(timezone.utc)
        assert node.temporal_relevance() == pytest.approx(1.0, abs=1e-6)

    def test_concept_half_life_halves(self):
        node = ConceptNode(name="aging")
        node.last_activated = datetime.now(timezone.utc) - timedelta(hours=168)
        tr = node.temporal_relevance(half_life_hours=168.0)
        assert abs(tr - 0.5) < 0.02, f"Expected ~0.5, got {tr:.4f}"

    def test_concept_two_half_lives(self):
        node = ConceptNode(name="old")
        node.last_activated = datetime.now(timezone.utc) - timedelta(hours=336)
        tr = node.temporal_relevance(half_life_hours=168.0)
        assert abs(tr - 0.25) < 0.02, f"Expected ~0.25, got {tr:.4f}"

    def test_concept_very_old_hits_floor(self):
        node = ConceptNode(name="ancient")
        node.last_activated = datetime.now(timezone.utc) - timedelta(days=365)
        tr = node.temporal_relevance(half_life_hours=168.0)
        assert tr == 0.1, f"Expected floor 0.1, got {tr:.4f}"

    def test_concept_temporal_monotonic_decrease(self):
        """Temporal relevance should decrease with age."""
        base = datetime.now(timezone.utc)
        prev = 1.0
        for hours in [0, 24, 72, 168, 336, 720]:
            node = ConceptNode(name="mono")
            node.last_activated = base - timedelta(hours=hours)
            tr = node.temporal_relevance()
            assert tr <= prev, f"Not monotonic at {hours}h: {tr} > {prev}"
            prev = tr

    def test_relation_just_reinforced_is_1(self):
        edge = RelationEdge(source_id="a", target_id="b")
        edge.last_reinforced = datetime.now(timezone.utc)
        assert edge.temporal_relevance() == pytest.approx(1.0, abs=1e-6)

    def test_relation_half_life_halves(self):
        edge = RelationEdge(source_id="a", target_id="b")
        edge.last_reinforced = datetime.now(timezone.utc) - timedelta(hours=72)
        edge.reinforcement_count = 0
        tr = edge.temporal_relevance(half_life_hours=72.0)
        assert abs(tr - 0.5) < 0.02, f"Expected ~0.5, got {tr:.4f}"

    def test_relation_reinforcement_extends_half_life(self):
        """More reinforced relations should have higher temporal relevance
        at the same age, because their effective half-life is longer."""
        base = datetime.now(timezone.utc) - timedelta(hours=72)

        # Not reinforced
        e0 = RelationEdge(source_id="a", target_id="b")
        e0.last_reinforced = base
        e0.reinforcement_count = 0

        # Heavily reinforced
        e10 = RelationEdge(source_id="a", target_id="b")
        e10.last_reinforced = base
        e10.reinforcement_count = 10

        tr0 = e0.temporal_relevance(half_life_hours=72.0)
        tr10 = e10.temporal_relevance(half_life_hours=72.0)

        assert tr10 > tr0, (
            f"Reinforced relation should stay fresher: "
            f"count=0 → {tr0:.4f}, count=10 → {tr10:.4f}"
        )

    def test_relation_very_old_hits_floor(self):
        edge = RelationEdge(source_id="a", target_id="b")
        edge.last_reinforced = datetime.now(timezone.utc) - timedelta(days=365)
        edge.reinforcement_count = 0
        tr = edge.temporal_relevance()
        assert tr == 0.15, f"Expected floor 0.15, got {tr:.4f}"


# ═══════════════════════════════════════════════════════════════════════
# 2. Activation with Time Decay
# ═══════════════════════════════════════════════════════════════════════

class TestActivationTimeDimension:
    """Verify that temporal freshness affects activation propagation."""

    def test_fresh_neighbor_scores_higher_than_stale(self, world):
        """A recently activated neighbor should receive a higher activation
        score than one activated long ago, all else being equal."""
        # Create two parallel paths from seed
        world.ingest(Observation(
            concepts=["seed", "fresh_target", "stale_target"],
            relations=[
                ("seed", "fresh_target", "depends_on"),
                ("seed", "stale_target", "depends_on"),
            ],
            task="test", source="bench",
        ))
        # Reinforce both equally
        for _ in range(10):
            world.ingest(Observation(
                concepts=["seed", "fresh_target", "stale_target"],
                relations=[
                    ("seed", "fresh_target", "depends_on"),
                    ("seed", "stale_target", "depends_on"),
                ],
                task="test", source="bench",
            ))

        # Age stale_target
        stale = world.concepts.resolve("stale_target")
        stale.last_activated = datetime.now(timezone.utc) - timedelta(days=30)

        # Also age the relation to stale_target
        seed = world.concepts.resolve("seed")
        for rel in world.relations.for_concept(seed.id):
            if rel.other_end(seed.id) == stale.id:
                rel.last_reinforced = datetime.now(timezone.utc) - timedelta(days=30)

        proj = world.project(["seed"], task="test")
        scores = proj.activation_scores

        fresh = world.concepts.resolve("fresh_target")
        fresh_score = scores.get(fresh.id, 0)
        stale_score = scores.get(stale.id, 0)

        assert fresh_score > stale_score, (
            f"Fresh ({fresh_score:.4f}) should score higher than "
            f"stale ({stale_score:.4f})"
        )

    def test_stale_concept_still_reachable(self, world):
        """Even a very old concept should still appear in projections
        thanks to the temporal floor (0.1)."""
        world.ingest(Observation(
            concepts=["root", "ancient"],
            relations=[("root", "ancient", "depends_on")],
            task="test", source="bench",
        ))
        for _ in range(10):
            world.ingest(Observation(
                concepts=["root", "ancient"],
                relations=[("root", "ancient", "depends_on")],
                task="test", source="bench",
            ))

        # Age ancient concept to 6 months ago
        ancient = world.concepts.resolve("ancient")
        ancient.last_activated = datetime.now(timezone.utc) - timedelta(days=180)

        proj = world.project(["root"], task="test")
        names = {c.name for c in proj.concepts}

        assert "ancient" in names, (
            f"Ancient concept should still be reachable. Got: {names}"
        )

    def test_recently_reactivated_concept_recovers(self, world):
        """A stale concept that gets reactivated should recover its
        temporal relevance immediately."""
        world.ingest(Observation(
            concepts=["node_a", "node_b"],
            relations=[("node_a", "node_b", "depends_on")],
            task="test", source="bench",
        ))
        for _ in range(5):
            world.ingest(Observation(
                concepts=["node_a", "node_b"],
                relations=[("node_a", "node_b", "depends_on")],
                task="test", source="bench",
            ))

        b = world.concepts.resolve("node_b")
        b.last_activated = datetime.now(timezone.utc) - timedelta(days=60)
        tr_before = b.temporal_relevance()

        # Reactivate
        world.ingest(Observation(
            concepts=["node_b"], task="reactivate", source="bench",
        ))

        b = world.concepts.resolve("node_b")
        tr_after = b.temporal_relevance()

        assert tr_after > tr_before, (
            f"Reactivation should restore freshness: "
            f"{tr_before:.4f} → {tr_after:.4f}"
        )
        assert tr_after > 0.9, (
            f"Just reactivated should be near 1.0: {tr_after:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 3. Projection Temporal Preference
# ═══════════════════════════════════════════════════════════════════════

class TestProjectionTemporalPreference:
    """MMR selection should prefer fresher concepts."""

    def test_fresh_concepts_ranked_higher(self, world):
        """Among equally relevant concepts, fresh ones should rank higher."""
        # Build a hub with many spokes
        spokes = [f"spoke_{i}" for i in range(8)]
        for _ in range(10):
            world.ingest(Observation(
                concepts=["hub"] + spokes,
                relations=[("hub", s, "supports") for s in spokes],
                task="test", source="bench",
            ))

        # Make half the spokes stale
        for i in range(4):
            node = world.concepts.resolve(f"spoke_{i}")
            node.last_activated = datetime.now(timezone.utc) - timedelta(days=60)

        proj = world.project(["hub"], task="test", max_concepts=6)
        top_names = [c.name for c in proj.top_concepts(6) if c.name != "hub"]

        # Fresh spokes (4-7) should tend to appear before stale spokes (0-3)
        fresh_spokes = {f"spoke_{i}" for i in range(4, 8)}
        stale_spokes = {f"spoke_{i}" for i in range(4)}

        fresh_in_top = len(set(top_names[:4]) & fresh_spokes)
        stale_in_top = len(set(top_names[:4]) & stale_spokes)

        assert fresh_in_top >= stale_in_top, (
            f"Fresh concepts should dominate top ranks. "
            f"Fresh in top 4: {fresh_in_top}, stale: {stale_in_top}. "
            f"Ranking: {top_names}"
        )

    def test_projection_shifts_with_time(self, world):
        """After aging domain A and refreshing domain B, projection
        from a bridge concept should shift toward B."""
        # Domain A
        for _ in range(10):
            world.ingest(Observation(
                concepts=["bridge", "domain_a_1", "domain_a_2"],
                relations=[
                    ("bridge", "domain_a_1", "supports"),
                    ("bridge", "domain_a_2", "supports"),
                ],
                task="domain_a", source="bench",
            ))
        # Domain B
        for _ in range(10):
            world.ingest(Observation(
                concepts=["bridge", "domain_b_1", "domain_b_2"],
                relations=[
                    ("bridge", "domain_b_1", "depends_on"),
                    ("bridge", "domain_b_2", "depends_on"),
                ],
                task="domain_b", source="bench",
            ))

        # Age domain A
        for name in ["domain_a_1", "domain_a_2"]:
            node = world.concepts.resolve(name)
            node.last_activated = datetime.now(timezone.utc) - timedelta(days=30)
        # Age relations to domain A
        bridge = world.concepts.resolve("bridge")
        for rel in world.relations.for_concept(bridge.id):
            other = rel.other_end(bridge.id)
            node = world.concepts.get(other)
            if node and node.name.startswith("domain_a"):
                rel.last_reinforced = datetime.now(timezone.utc) - timedelta(days=30)

        proj = world.project(["bridge"], task="", max_concepts=4)
        scores = proj.activation_scores

        # Domain B should have higher scores than domain A
        a1 = world.concepts.resolve("domain_a_1")
        b1 = world.concepts.resolve("domain_b_1")

        score_a = scores.get(a1.id, 0) if a1 else 0
        score_b = scores.get(b1.id, 0) if b1 else 0

        assert score_b > score_a, (
            f"Fresh domain B ({score_b:.4f}) should outscore "
            f"stale domain A ({score_a:.4f})"
        )


# ═══════════════════════════════════════════════════════════════════════
# 4. Multi-Session Temporal Evolution
# ═══════════════════════════════════════════════════════════════════════

class TestMultiSessionTemporalEvolution:
    """Simulate realistic time-passing across work sessions."""

    def test_recent_session_dominates_projection(self, world):
        """Concepts from the most recent session should dominate the
        projection for a shared seed."""
        # Session 1 (old): backend
        for _ in range(10):
            world.ingest(Observation(
                concepts=["python", "flask", "sqlalchemy"],
                relations=[("python", "flask", "supports")],
                task="backend", source="session_1",
            ))

        # Simulate 2 weeks passing
        for name in ["flask", "sqlalchemy"]:
            node = world.concepts.resolve(name)
            node.last_activated = datetime.now(timezone.utc) - timedelta(days=14)
        # Age relations
        python_node = world.concepts.resolve("python")
        for rel in world.relations.for_concept(python_node.id):
            other = rel.other_end(python_node.id)
            n = world.concepts.get(other)
            if n and n.name in ("flask", "sqlalchemy"):
                rel.last_reinforced = datetime.now(timezone.utc) - timedelta(days=14)

        # Session 2 (recent): ML
        for _ in range(10):
            world.ingest(Observation(
                concepts=["python", "pytorch", "numpy"],
                relations=[("python", "pytorch", "supports")],
                task="ML", source="session_2",
            ))

        proj = world.project(["python"], task="", max_concepts=5)
        scores = proj.activation_scores

        pytorch = world.concepts.resolve("pytorch")
        flask = world.concepts.resolve("flask")

        pytorch_score = scores.get(pytorch.id, 0) if pytorch else 0
        flask_score = scores.get(flask.id, 0) if flask else 0

        assert pytorch_score > flask_score, (
            f"Recent pytorch ({pytorch_score:.4f}) should outscore "
            f"old flask ({flask_score:.4f})"
        )

    def test_temporal_decay_across_reflect_cycles(self, world):
        """Temporal relevance should interact correctly with reflect decay."""
        for _ in range(10):
            world.ingest(Observation(
                concepts=["temporal_concept"],
                task="work", source="bench",
            ))

        node = world.concepts.resolve("temporal_concept")
        initial_tr = node.temporal_relevance()
        initial_conf = node.confidence

        # Simulate time + reflect
        node.last_activated = datetime.now(timezone.utc) - timedelta(hours=48)
        world.reflect()

        node = world.concepts.resolve("temporal_concept")
        if node:
            later_tr = node.temporal_relevance()
            later_conf = node.confidence

            assert later_tr < initial_tr, (
                f"Temporal relevance should decrease: {initial_tr:.4f} → {later_tr:.4f}"
            )
            assert later_conf < initial_conf, (
                f"Confidence should decay: {initial_conf:.4f} → {later_conf:.4f}"
            )


# ═══════════════════════════════════════════════════════════════════════
# 5. Temporal + Task Interaction
# ═══════════════════════════════════════════════════════════════════════

class TestTemporalTaskInteraction:
    """Time and task affinity should compound correctly."""

    def test_fresh_and_task_aligned_wins(self, world):
        """A concept that is both fresh and task-aligned should dominate
        one that is only task-aligned but stale."""
        # Build world with two task-aligned concepts
        for _ in range(10):
            world.ingest(Observation(
                concepts=["seed", "fresh_aligned", "stale_aligned"],
                relations=[
                    ("seed", "fresh_aligned", "depends_on"),
                    ("seed", "stale_aligned", "depends_on"),
                ],
                task="target_task", source="bench",
            ))

        # Age one of them
        stale = world.concepts.resolve("stale_aligned")
        stale.last_activated = datetime.now(timezone.utc) - timedelta(days=30)

        proj = world.project(["seed"], task="target_task")
        scores = proj.activation_scores

        fresh = world.concepts.resolve("fresh_aligned")
        fresh_score = scores.get(fresh.id, 0)
        stale_score = scores.get(stale.id, 0)

        assert fresh_score > stale_score, (
            f"Fresh+aligned ({fresh_score:.4f}) should beat "
            f"stale+aligned ({stale_score:.4f})"
        )

    def test_fresh_unaligned_vs_stale_aligned(self, world):
        """A fresh but unaligned concept may compete with a stale but
        aligned one — both factors should matter."""
        for _ in range(10):
            world.ingest(Observation(
                concepts=["hub", "aligned_old"],
                relations=[("hub", "aligned_old", "depends_on")],
                task="special_task", source="bench",
            ))
        for _ in range(10):
            world.ingest(Observation(
                concepts=["hub", "unaligned_new"],
                relations=[("hub", "unaligned_new", "depends_on")],
                task="other_task", source="bench",
            ))

        # Age the aligned concept
        aligned = world.concepts.resolve("aligned_old")
        aligned.last_activated = datetime.now(timezone.utc) - timedelta(days=14)

        proj = world.project(["hub"], task="special_task")
        scores = proj.activation_scores

        unaligned = world.concepts.resolve("unaligned_new")
        aligned_score = scores.get(aligned.id, 0)
        unaligned_score = scores.get(unaligned.id, 0)

        # Both should be reachable — temporal and task factors compete
        assert aligned_score > 0, "Stale but aligned should still be reachable"
        assert unaligned_score > 0, "Fresh but unaligned should still be reachable"


# ═══════════════════════════════════════════════════════════════════════
# 6. Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestTemporalEdgeCases:
    """Boundary conditions for temporal relevance."""

    def test_zero_half_life_returns_1(self):
        """Half-life of 0 should return 1.0 (no decay)."""
        node = ConceptNode(name="test")
        node.last_activated = datetime.now(timezone.utc) - timedelta(hours=100)
        assert node.temporal_relevance(half_life_hours=0.0) == 1.0

    def test_negative_half_life_returns_1(self):
        node = ConceptNode(name="test")
        node.last_activated = datetime.now(timezone.utc) - timedelta(hours=100)
        assert node.temporal_relevance(half_life_hours=-10.0) == 1.0

    def test_future_timestamp_returns_1(self):
        """A concept activated in the 'future' (clock skew) should return 1.0."""
        node = ConceptNode(name="future")
        node.last_activated = datetime.now(timezone.utc) + timedelta(hours=1)
        assert node.temporal_relevance() == 1.0

    def test_temporal_floor_guarantees_minimum(self):
        """Even at extreme age, temporal relevance never goes below floor."""
        node = ConceptNode(name="ancient")
        node.last_activated = datetime.now(timezone.utc) - timedelta(days=3650)
        tr = node.temporal_relevance()
        assert tr == 0.1

        edge = RelationEdge(source_id="a", target_id="b")
        edge.last_reinforced = datetime.now(timezone.utc) - timedelta(days=3650)
        edge.reinforcement_count = 0
        tr_rel = edge.temporal_relevance()
        assert tr_rel == 0.15

    def test_temporal_relevance_is_serializable(self, tmp_path):
        """Temporal relevance should work correctly after save/load."""
        store = tmp_path / ".w0"
        w1 = World(store_path=store)
        w1.ingest(Observation(concepts=["persist_test"], source="bench"))
        node = w1.concepts.resolve("persist_test")
        tr1 = node.temporal_relevance()
        w1.concepts.save_all()
        del w1

        w2 = World(store_path=store)
        node2 = w2.concepts.resolve("persist_test")
        tr2 = node2.temporal_relevance()

        # Should be very close (only microseconds passed during save/load)
        assert abs(tr1 - tr2) < 0.01


# ═══════════════════════════════════════════════════════════════════════
# 7. Quantitative Report
# ═══════════════════════════════════════════════════════════════════════

class TestTemporalReport:
    """Print a quantitative report of temporal dimension effects."""

    def test_temporal_dimension_report(self, world, capsys):
        """Measure temporal effects across different aging scenarios."""
        # Build world
        concepts_fresh = ["fresh_a", "fresh_b", "fresh_c"]
        concepts_medium = ["med_a", "med_b", "med_c"]
        concepts_stale = ["stale_a", "stale_b", "stale_c"]
        all_concepts = concepts_fresh + concepts_medium + concepts_stale

        for _ in range(10):
            world.ingest(Observation(
                concepts=["hub"] + all_concepts,
                relations=[("hub", c, "supports") for c in all_concepts],
                task="test", source="bench",
            ))

        # Age groups
        for name in concepts_medium:
            node = world.concepts.resolve(name)
            node.last_activated = datetime.now(timezone.utc) - timedelta(days=7)
        for name in concepts_stale:
            node = world.concepts.resolve(name)
            node.last_activated = datetime.now(timezone.utc) - timedelta(days=60)

        # Also age relations to match
        hub = world.concepts.resolve("hub")
        for rel in world.relations.for_concept(hub.id):
            other_id = rel.other_end(hub.id)
            other = world.concepts.get(other_id)
            if other and other.name in concepts_medium:
                rel.last_reinforced = datetime.now(timezone.utc) - timedelta(days=7)
            elif other and other.name in concepts_stale:
                rel.last_reinforced = datetime.now(timezone.utc) - timedelta(days=60)

        proj = world.project(["hub"], task="test", max_concepts=12)
        scores = proj.activation_scores
        proj_names = {c.name for c in proj.concepts}

        report = "\n"
        report += "╔════════════════════════════════════════════════════════════╗\n"
        report += "║         Temporal Dimension Effect Report                  ║\n"
        report += "╠════════════════════════════════════════════════════════════╣\n"
        report += "║ Concept         Age        TR      ActivScore  InProj    ║\n"
        report += "╠════════════════════════════════════════════════════════════╣\n"

        for group_name, group, age_label in [
            ("Fresh", concepts_fresh, "0 days"),
            ("Medium", concepts_medium, "7 days"),
            ("Stale", concepts_stale, "60 days"),
        ]:
            for name in group:
                node = world.concepts.resolve(name)
                tr = node.temporal_relevance()
                score = scores.get(node.id, 0)
                in_proj = "YES" if name in proj_names else "NO"
                report += (
                    f"║ {name:15s} {age_label:10s} {tr:>5.3f}   "
                    f"{score:>8.4f}    {in_proj:3s}      ║\n"
                )

        # Summary
        fresh_scores = [scores.get(world.concepts.resolve(n).id, 0) for n in concepts_fresh]
        med_scores = [scores.get(world.concepts.resolve(n).id, 0) for n in concepts_medium]
        stale_scores = [scores.get(world.concepts.resolve(n).id, 0) for n in concepts_stale]

        avg_fresh = sum(fresh_scores) / len(fresh_scores) if fresh_scores else 0
        avg_med = sum(med_scores) / len(med_scores) if med_scores else 0
        avg_stale = sum(stale_scores) / len(stale_scores) if stale_scores else 0

        report += "╠════════════════════════════════════════════════════════════╣\n"
        report += f"║ Avg score  Fresh: {avg_fresh:.4f}  Med: {avg_med:.4f}  Stale: {avg_stale:.4f}   ║\n"
        report += f"║ Ratio  Fresh/Med: {avg_fresh/avg_med:.2f}x    Fresh/Stale: {avg_fresh/max(avg_stale,0.0001):.2f}x        ║\n"
        report += "╚════════════════════════════════════════════════════════════╝\n"
        print(report)

        # Assertions
        assert avg_fresh > avg_med > avg_stale, (
            f"Expected fresh > medium > stale: "
            f"{avg_fresh:.4f} > {avg_med:.4f} > {avg_stale:.4f}"
        )
        assert avg_fresh / avg_med > 1.05, (
            f"Fresh should meaningfully outscore medium: ratio={avg_fresh/avg_med:.2f}"
        )
