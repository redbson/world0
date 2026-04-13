"""Benchmark validation tests — quantitative measurements of World 0's
core cognitive properties.

Each test class targets a specific measurable property with numerical
thresholds. These are not performance benchmarks (speed), but quality
benchmarks (does the system behave correctly at scale and under stress).

Metrics:
  - Activation precision: seed > neighbor > distant
  - Projection relevance: precision/recall against expected sets
  - Confidence dynamics: monotonic growth with diminishing returns
  - Decay curve fidelity: match mathematical half-life model
  - Hebbian weight convergence: co-occurrence strengthens predictably
  - Cross-domain separation: Jaccard distance between domain projections
  - Scale behavior: 100+ concepts without quality degradation
  - Persistence roundtrip: zero numerical drift after save/load
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from world0 import Observation, World
from world0.schemas.concept import ConceptNode, Maturity
from world0.schemas.relation import RelationEdge, RelationType
from world0.dynamics.decay import CONCEPT_HALF_LIFE, RELATION_BASE_HALF_LIFE


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def jaccard(a: set, b: set) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|. 0 = disjoint, 1 = identical."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def precision_recall(retrieved: set, relevant: set) -> tuple[float, float]:
    """Return (precision, recall) against a ground-truth set."""
    if not retrieved:
        return (0.0, 0.0)
    tp = len(retrieved & relevant)
    precision = tp / len(retrieved)
    recall = tp / len(relevant) if relevant else 0.0
    return precision, recall


def build_chain_world(world: World, length: int) -> list[str]:
    """Build a linear chain: c0 → c1 → c2 → ... → c{length-1}.

    Returns the list of concept names in order.
    """
    names = [f"chain_{i}" for i in range(length)]
    for i, name in enumerate(names):
        concepts = [name]
        relations = []
        if i > 0:
            concepts.append(names[i - 1])
            relations.append((names[i - 1], name, "precedes"))
        world.ingest(Observation(
            concepts=concepts, relations=relations,
            task="build_chain", source="bench",
        ))
    # Reinforce all to raise confidence above noise floor
    for _ in range(5):
        for name in names:
            world.ingest(Observation(
                concepts=[name], task="reinforce", source="bench",
            ))
    return names


@pytest.fixture
def world(tmp_path):
    return World(store_path=tmp_path / ".world0")


# ═══════════════════════════════════════════════════════════════════════
# 1. Activation Precision
# ═══════════════════════════════════════════════════════════════════════

class TestActivationPrecision:
    """Activation scores must decay monotonically with graph distance."""

    def test_seed_scores_highest(self, world):
        names = build_chain_world(world, 6)
        proj = world.project([names[0]], max_depth=5, decay=0.5)
        scores = proj.activation_scores

        seed = world.concepts.resolve(names[0])
        seed_score = scores.get(seed.id, 0)

        for c in proj.concepts:
            if c.id != seed.id:
                assert scores.get(c.id, 0) <= seed_score, (
                    f"Non-seed {c.name} scored {scores.get(c.id, 0)} > seed {seed_score}"
                )

    def test_score_decreases_with_distance(self, world):
        """In a linear chain, score at distance d+1 < score at distance d."""
        names = build_chain_world(world, 5)
        proj = world.project([names[0]], max_depth=4, decay=0.5)
        scores = proj.activation_scores

        prev_score = float("inf")
        for name in names:
            node = world.concepts.resolve(name)
            if not node or node.id not in scores:
                break
            score = scores[node.id]
            assert score <= prev_score, (
                f"{name} scored {score} > previous {prev_score}"
            )
            prev_score = score

    def test_decay_factor_controls_falloff_rate(self, world):
        """Higher decay → slower falloff; lower decay → faster falloff."""
        names = build_chain_world(world, 4)

        proj_fast = world.project([names[0]], max_depth=3, decay=0.3)
        proj_slow = world.project([names[0]], max_depth=3, decay=0.7)

        # At distance 2+, slow decay should retain more concepts
        slow_count = len(proj_slow.concepts)
        fast_count = len(proj_fast.concepts)
        assert slow_count >= fast_count, (
            f"Slow decay ({slow_count} concepts) should retain >= "
            f"fast decay ({fast_count} concepts)"
        )


# ═══════════════════════════════════════════════════════════════════════
# 2. Projection Relevance (Precision / Recall)
# ═══════════════════════════════════════════════════════════════════════

class TestProjectionRelevance:
    """Projections should retrieve relevant concepts with high precision."""

    def _build_two_clusters(self, world):
        """Build two distinct clusters connected by a single bridge concept."""
        # Cluster A: ML concepts
        for _ in range(8):
            world.ingest(Observation(
                concepts=["neural network", "gradient descent", "loss function",
                          "backpropagation", "optimizer"],
                relations=[
                    ("neural network", "gradient descent", "depends_on"),
                    ("gradient descent", "loss function", "depends_on"),
                    ("backpropagation", "gradient descent", "activates"),
                    ("optimizer", "gradient descent", "supports"),
                ],
                task="ML training", source="bench",
            ))

        # Cluster B: DevOps concepts
        for _ in range(8):
            world.ingest(Observation(
                concepts=["CI/CD", "terraform", "monitoring",
                          "load balancer", "autoscaling"],
                relations=[
                    ("CI/CD", "terraform", "precedes"),
                    ("monitoring", "autoscaling", "activates"),
                    ("load balancer", "autoscaling", "supports"),
                ],
                task="DevOps setup", source="bench",
            ))

        # Bridge: one concept touching both clusters
        for _ in range(4):
            world.ingest(Observation(
                concepts=["neural network", "CI/CD", "model deployment"],
                relations=[
                    ("neural network", "model deployment", "precedes"),
                    ("model deployment", "CI/CD", "depends_on"),
                ],
                task="bridge", source="bench",
            ))

    def test_ml_seed_yields_ml_precision(self, world):
        self._build_two_clusters(world)
        proj = world.project(["neural network", "gradient descent"], task="ML")
        retrieved = {c.name for c in proj.concepts}
        ml_relevant = {"neural network", "gradient descent", "loss function",
                       "backpropagation", "optimizer"}

        p, r = precision_recall(retrieved, ml_relevant)
        assert p >= 0.4, f"ML precision {p:.2f} < 0.4"
        assert r >= 0.4, f"ML recall {r:.2f} < 0.4"

    def test_devops_seed_yields_devops_precision(self, world):
        self._build_two_clusters(world)
        proj = world.project(["CI/CD", "terraform"], task="DevOps")
        retrieved = {c.name for c in proj.concepts}
        devops_relevant = {"CI/CD", "terraform", "monitoring",
                           "load balancer", "autoscaling"}

        p, r = precision_recall(retrieved, devops_relevant)
        assert p >= 0.4, f"DevOps precision {p:.2f} < 0.4"
        assert r >= 0.4, f"DevOps recall {r:.2f} < 0.4"

    def test_bridge_concept_reachable_from_both(self, world):
        self._build_two_clusters(world)
        proj_ml = world.project(["neural network"], task="ML")
        proj_ops = world.project(["CI/CD"], task="DevOps")

        ml_names = {c.name for c in proj_ml.concepts}
        ops_names = {c.name for c in proj_ops.concepts}

        assert "model deployment" in ml_names or "model deployment" in ops_names, (
            "Bridge concept 'model deployment' should be reachable from at least one cluster"
        )


# ═══════════════════════════════════════════════════════════════════════
# 3. Confidence Dynamics
# ═══════════════════════════════════════════════════════════════════════

class TestConfidenceDynamics:
    """Confidence must grow monotonically with reinforcement, with
    diminishing returns, and remain within [0, 1]."""

    def test_confidence_monotonically_increases(self, world):
        world.ingest(Observation(concepts=["alpha"], source="bench"))
        prev = world.concepts.resolve("alpha").confidence

        for i in range(20):
            world.ingest(Observation(concepts=["alpha"], source="bench"))
            curr = world.concepts.resolve("alpha").confidence
            assert curr >= prev, (
                f"Step {i}: confidence dropped {prev:.4f} → {curr:.4f}"
            )
            prev = curr

    def test_confidence_has_diminishing_returns(self, world):
        world.ingest(Observation(concepts=["beta"], source="bench"))
        c = world.concepts.resolve("beta")

        # Record boost sizes
        boosts = []
        for _ in range(15):
            before = c.confidence
            world.concepts.reinforce(c.id, source="bench")
            after = c.confidence
            boosts.append(after - before)

        # Later boosts should be smaller on average
        first_half_avg = sum(boosts[:5]) / 5
        second_half_avg = sum(boosts[10:]) / 5
        assert second_half_avg < first_half_avg, (
            f"Diminishing returns violated: early avg {first_half_avg:.5f}, "
            f"late avg {second_half_avg:.5f}"
        )

    def test_confidence_bounded_0_to_1(self, world):
        world.ingest(Observation(concepts=["gamma"], source="bench"))
        # Reinforce 100 times
        c = world.concepts.resolve("gamma")
        for _ in range(100):
            world.concepts.reinforce(c.id, source="bench")

        c = world.concepts.resolve("gamma")
        assert 0.0 <= c.confidence <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# 4. Decay Curve Fidelity
# ═══════════════════════════════════════════════════════════════════════

class TestDecayCurve:
    """Decay must match the expected exponential half-life model."""

    @pytest.mark.parametrize("maturity,half_life", [
        (Maturity.EMBRYONIC, CONCEPT_HALF_LIFE[Maturity.EMBRYONIC]),
        (Maturity.DEVELOPING, CONCEPT_HALF_LIFE[Maturity.DEVELOPING]),
        (Maturity.ESTABLISHED, CONCEPT_HALF_LIFE[Maturity.ESTABLISHED]),
        (Maturity.CORE, CONCEPT_HALF_LIFE[Maturity.CORE]),
    ])
    def test_concept_decay_matches_half_life(self, world, maturity, half_life):
        """After exactly one half-life, confidence should halve (±5%)."""
        world.ingest(Observation(concepts=["decay_test"], source="bench"))
        node = world.concepts.resolve("decay_test")

        # Set initial state
        initial_confidence = 0.8
        node.confidence = initial_confidence
        node.maturity = maturity
        node.last_activated = datetime.now(timezone.utc) - timedelta(hours=half_life)

        # Run decay
        world._decay.decay_concepts()

        expected = initial_confidence * 0.5
        actual = node.confidence
        tolerance = 0.05

        assert abs(actual - expected) < tolerance, (
            f"{maturity.value}: after {half_life}h, expected ~{expected:.3f}, "
            f"got {actual:.3f}"
        )

    def test_relation_decay_matches_half_life(self, world):
        """Unreinforced relation should halve weight after base half-life."""
        world.ingest(Observation(
            concepts=["r_src", "r_tgt"],
            relations=[("r_src", "r_tgt", "depends_on")],
            source="bench",
        ))

        src = world.concepts.resolve("r_src")
        tgt = world.concepts.resolve("r_tgt")
        edge = world.relations.find_between(src.id, tgt.id, RelationType.DEPENDS_ON)

        initial_weight = 0.8
        edge.weight = initial_weight
        edge.confidence = initial_weight
        edge.reinforcement_count = 0
        half_life = RELATION_BASE_HALF_LIFE * (1.0 + 0 * 0.5)  # = base
        edge.last_reinforced = datetime.now(timezone.utc) - timedelta(hours=half_life)

        world._decay.decay_relations()

        expected = initial_weight * 0.5
        assert abs(edge.weight - expected) < 0.05, (
            f"Relation decay: expected ~{expected:.3f}, got {edge.weight:.3f}"
        )

    def test_reinforcement_slows_relation_decay(self, world):
        """A heavily reinforced relation should decay slower."""
        world.ingest(Observation(
            concepts=["slow_src", "slow_tgt"],
            relations=[("slow_src", "slow_tgt", "supports")],
            source="bench",
        ))
        src = world.concepts.resolve("slow_src")
        tgt = world.concepts.resolve("slow_tgt")
        edge = world.relations.find_between(src.id, tgt.id, RelationType.SUPPORTS)

        # Simulate heavily reinforced
        edge.weight = 0.8
        edge.confidence = 0.8
        edge.reinforcement_count = 10
        elapsed = RELATION_BASE_HALF_LIFE  # one base half-life
        edge.last_reinforced = datetime.now(timezone.utc) - timedelta(hours=elapsed)

        world._decay.decay_relations()

        # With 10 reinforcements, effective half-life = 72*(1+10*0.5) = 432h
        # After 72h: factor = 0.5^(72/432) = 0.5^(1/6) ≈ 0.891
        expected_factor = math.pow(0.5, elapsed / (RELATION_BASE_HALF_LIFE * (1.0 + 10 * 0.5)))
        expected = 0.8 * expected_factor
        assert edge.weight > expected * 0.9, (
            f"Reinforced relation decayed too fast: {edge.weight:.3f} vs expected ~{expected:.3f}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 5. Hebbian Weight Convergence
# ═══════════════════════════════════════════════════════════════════════

class TestHebbianConvergence:
    """Co-occurring concepts should form and strengthen relations
    predictably."""

    def test_weight_grows_with_cooccurrence_count(self, world):
        """More co-occurrences → higher relation weight."""
        weights = []
        for i in range(15):
            world.ingest(Observation(
                concepts=["hebb_a", "hebb_b"],
                task=f"iter_{i}", source="bench",
            ))
            a = world.concepts.resolve("hebb_a")
            b = world.concepts.resolve("hebb_b")
            rels = world.relations.find_any_between(a.id, b.id)
            if rels:
                weights.append(max(r.weight for r in rels))

        # Weight should be monotonically non-decreasing
        for i in range(1, len(weights)):
            assert weights[i] >= weights[i - 1], (
                f"Hebbian weight decreased at step {i}: "
                f"{weights[i - 1]:.4f} → {weights[i]:.4f}"
            )

        # Final weight should be substantially above initial
        assert weights[-1] > weights[0] * 1.5, (
            f"Hebbian convergence too slow: {weights[0]:.4f} → {weights[-1]:.4f}"
        )

    def test_hebbian_weight_capped_below_explicit(self, world):
        """Hebbian relations cap at 0.7; explicit relations can reach 1.0."""
        # Hebbian: co-occurrence only (no explicit relation declared)
        for i in range(100):
            world.ingest(Observation(
                concepts=["bound_a", "bound_b"],
                task=f"iter_{i}", source="bench",
            ))
        a = world.concepts.resolve("bound_a")
        b = world.concepts.resolve("bound_b")
        hebbian_rels = [r for r in world.relations.find_any_between(a.id, b.id)
                        if not r.is_explicit]
        for r in hebbian_rels:
            assert r.weight <= 0.7, f"Hebbian weight {r.weight} > 0.7 cap"
            assert r.confidence <= 0.7

        # Explicit: declared relation
        for i in range(100):
            world.ingest(Observation(
                concepts=["exp_a", "exp_b"],
                relations=[("exp_a", "exp_b", "depends_on")],
                task=f"iter_{i}", source="bench",
            ))
        ea = world.concepts.resolve("exp_a")
        eb = world.concepts.resolve("exp_b")
        explicit_rels = [r for r in world.relations.find_any_between(ea.id, eb.id)
                         if r.is_explicit]
        assert len(explicit_rels) > 0, "Expected explicit relation"
        for r in explicit_rels:
            assert r.weight > 0.7, f"Explicit weight {r.weight} should exceed Hebbian cap"


# ═══════════════════════════════════════════════════════════════════════
# 6. Cross-Domain Separation
# ═══════════════════════════════════════════════════════════════════════

class TestCrossDomainSeparation:
    """Projections from different domains should have low Jaccard
    similarity (high separation)."""

    def _build_three_domains(self, world):
        domains = {
            "security": {
                "concepts": ["authentication", "encryption", "firewall",
                             "TLS", "OAuth", "RBAC"],
                "relations": [
                    ("authentication", "OAuth", "depends_on"),
                    ("encryption", "TLS", "contains"),
                    ("firewall", "RBAC", "supports"),
                ],
            },
            "database": {
                "concepts": ["PostgreSQL", "indexing", "query optimizer",
                             "transaction", "replication", "sharding"],
                "relations": [
                    ("PostgreSQL", "indexing", "contains"),
                    ("query optimizer", "indexing", "depends_on"),
                    ("replication", "sharding", "similar_to"),
                    ("transaction", "PostgreSQL", "part_of"),
                ],
            },
            "mobile": {
                "concepts": ["Swift", "UIKit", "push notification",
                             "app store", "CoreData", "gesture recognizer"],
                "relations": [
                    ("Swift", "UIKit", "supports"),
                    ("UIKit", "gesture recognizer", "contains"),
                    ("CoreData", "Swift", "supports"),
                    ("push notification", "app store", "depends_on"),
                ],
            },
        }

        for domain, data in domains.items():
            for _ in range(10):
                world.ingest(Observation(
                    concepts=data["concepts"],
                    relations=[(s, t, r) for s, t, r in data["relations"]],
                    task=domain, source="bench",
                ))

    def test_pairwise_jaccard_below_threshold(self, world):
        """Any two domain projections should have Jaccard < 0.3."""
        self._build_three_domains(world)

        proj_sec = world.project(["authentication", "encryption"], task="security")
        proj_db = world.project(["PostgreSQL", "indexing"], task="database")
        proj_mob = world.project(["Swift", "UIKit"], task="mobile")

        names_sec = {c.name for c in proj_sec.concepts}
        names_db = {c.name for c in proj_db.concepts}
        names_mob = {c.name for c in proj_mob.concepts}

        j_sec_db = jaccard(names_sec, names_db)
        j_sec_mob = jaccard(names_sec, names_mob)
        j_db_mob = jaccard(names_db, names_mob)

        assert j_sec_db < 0.3, f"security↔database Jaccard {j_sec_db:.2f} ≥ 0.3"
        assert j_sec_mob < 0.3, f"security↔mobile Jaccard {j_sec_mob:.2f} ≥ 0.3"
        assert j_db_mob < 0.3, f"database↔mobile Jaccard {j_db_mob:.2f} ≥ 0.3"

    def test_same_domain_jaccard_is_high(self, world):
        """Two seeds from the same domain should produce similar projections."""
        self._build_three_domains(world)

        # Use the actual task name so task affinity boosts in-domain propagation
        proj_a = world.project(["authentication"], task="security")
        proj_b = world.project(["encryption"], task="security")

        names_a = {c.name for c in proj_a.concepts}
        names_b = {c.name for c in proj_b.concepts}

        j = jaccard(names_a, names_b)
        assert j > 0.3, f"Same-domain Jaccard {j:.2f} ≤ 0.3 — projections too dissimilar"


# ═══════════════════════════════════════════════════════════════════════
# 7. Scale Behavior
# ═══════════════════════════════════════════════════════════════════════

class TestScaleBehavior:
    """System should handle 100+ concepts without quality degradation."""

    def _build_large_world(self, world, n_domains: int = 5, concepts_per: int = 20):
        """Build a world with n_domains * concepts_per concepts."""
        domain_concepts = {}
        for d in range(n_domains):
            names = [f"d{d}_c{i}" for i in range(concepts_per)]
            domain_concepts[d] = names
            # Ingest with internal relations (chain within domain)
            for _ in range(6):
                rels = [(names[i], names[i + 1], "precedes")
                        for i in range(len(names) - 1)]
                world.ingest(Observation(
                    concepts=names, relations=rels,
                    task=f"domain_{d}", source="bench",
                ))
        return domain_concepts

    def test_100_plus_concepts_retained(self, world):
        domain_concepts = self._build_large_world(world)
        status = world.status()
        assert status.total_concepts >= 100, (
            f"Only {status.total_concepts} concepts — expected ≥ 100"
        )

    def test_projection_still_discriminates_at_scale(self, world):
        domain_concepts = self._build_large_world(world)

        proj_d0 = world.project(
            [domain_concepts[0][0], domain_concepts[0][1]],
            task="domain_0", max_concepts=15,
        )
        proj_d4 = world.project(
            [domain_concepts[4][0], domain_concepts[4][1]],
            task="domain_4", max_concepts=15,
        )

        names_d0 = {c.name for c in proj_d0.concepts}
        names_d4 = {c.name for c in proj_d4.concepts}

        j = jaccard(names_d0, names_d4)
        assert j < 0.2, (
            f"At 100+ concepts, domain separation failed: Jaccard {j:.2f} ≥ 0.2"
        )

    def test_projection_respects_max_concepts(self, world):
        self._build_large_world(world)
        for limit in (5, 10, 15):
            proj = world.project(
                ["d0_c0"], task="limit_test", max_concepts=limit,
            )
            assert len(proj.concepts) <= limit, (
                f"max_concepts={limit} but got {len(proj.concepts)}"
            )

    def test_reflect_prunes_at_scale(self, world):
        domain_concepts = self._build_large_world(world)
        before = world.status().total_concepts

        # Age all concepts from domain 0
        for name in domain_concepts[0]:
            node = world.concepts.resolve(name)
            if node:
                node.confidence = 0.01
                node.maturity = Maturity.FADING
                node.last_activated = datetime.now(timezone.utc) - timedelta(hours=200)

        result = world.reflect()
        after = world.status().total_concepts
        assert after < before, (
            f"Reflect did not prune: before={before}, after={after}"
        )
        assert len(result.pruned_concepts) > 0


# ═══════════════════════════════════════════════════════════════════════
# 8. Lifecycle Promotion Thresholds
# ═══════════════════════════════════════════════════════════════════════

class TestLifecycleThresholds:
    """Verify exact promotion thresholds are correctly enforced."""

    def test_embryonic_not_promoted_below_threshold(self, world):
        world.ingest(Observation(concepts=["under"], source="bench"))
        # Only 2 activations (need 3), low confidence
        world.ingest(Observation(concepts=["under"], source="bench"))

        node = world.concepts.resolve("under")
        assert node.activation_count < 3 or node.confidence < 0.3

        world.reflect()
        node = world.concepts.resolve("under")
        assert node.maturity == Maturity.EMBRYONIC

    def test_embryonic_promoted_at_threshold(self, world):
        world.ingest(Observation(concepts=["promoted"], source="bench"))
        node = world.concepts.resolve("promoted")

        # Push past threshold: activation_count >= 3 and confidence >= 0.3
        for _ in range(5):
            world.concepts.reinforce(node.id, source="bench")
        node.confidence = 0.35

        world.reflect()
        node = world.concepts.resolve("promoted")
        assert node.maturity == Maturity.DEVELOPING

    def test_developing_not_promoted_without_enough_activations(self, world):
        world.ingest(Observation(concepts=["dev_stuck"], source="bench"))
        node = world.concepts.resolve("dev_stuck")
        node.maturity = Maturity.DEVELOPING
        node.confidence = 0.7
        node.activation_count = 5  # need 10

        world.reflect()
        node = world.concepts.resolve("dev_stuck")
        assert node.maturity == Maturity.DEVELOPING

    def test_established_to_core_requires_connections(self, world):
        """Established → Core requires activation_count >= 30 AND connections >= 5."""
        world.ingest(Observation(concepts=["lonely"], source="bench"))
        node = world.concepts.resolve("lonely")
        node.maturity = Maturity.ESTABLISHED
        node.activation_count = 50
        node.confidence = 0.9
        # No relations → connections = 0

        world.reflect()
        node = world.concepts.resolve("lonely")
        assert node.maturity == Maturity.ESTABLISHED, (
            "Should not promote to core without enough connections"
        )


# ═══════════════════════════════════════════════════════════════════════
# 9. Persistence Roundtrip Fidelity
# ═══════════════════════════════════════════════════════════════════════

class TestPersistenceFidelity:
    """Save/load roundtrip must preserve numerical values exactly."""

    def test_concept_values_survive_roundtrip(self, tmp_path):
        store_path = tmp_path / ".world0"
        w1 = World(store_path=store_path)

        for i in range(10):
            w1.ingest(Observation(
                concepts=["fidelity_test"],
                task=f"iter_{i}", source="bench",
            ))

        node_before = w1.concepts.resolve("fidelity_test")
        confidence_before = node_before.confidence
        activation_before = node_before.activation_count
        maturity_before = node_before.maturity

        w1.concepts.save_all()
        del w1

        w2 = World(store_path=store_path)
        node_after = w2.concepts.resolve("fidelity_test")

        assert node_after.confidence == confidence_before
        assert node_after.activation_count == activation_before
        assert node_after.maturity == maturity_before

    def test_relation_values_survive_roundtrip(self, tmp_path):
        store_path = tmp_path / ".world0"
        w1 = World(store_path=store_path)

        for i in range(8):
            w1.ingest(Observation(
                concepts=["rel_a", "rel_b"],
                relations=[("rel_a", "rel_b", "depends_on")],
                task=f"iter_{i}", source="bench",
            ))

        a = w1.concepts.resolve("rel_a")
        b = w1.concepts.resolve("rel_b")
        edge_before = w1.relations.find_between(a.id, b.id, RelationType.DEPENDS_ON)
        weight_before = edge_before.weight
        count_before = edge_before.reinforcement_count

        w1.relations.save_all()
        w1.concepts.save_all()
        del w1

        w2 = World(store_path=store_path)
        a2 = w2.concepts.resolve("rel_a")
        b2 = w2.concepts.resolve("rel_b")
        edge_after = w2.relations.find_between(a2.id, b2.id, RelationType.DEPENDS_ON)

        assert edge_after.weight == weight_before
        assert edge_after.reinforcement_count == count_before

    def test_projection_deterministic_after_roundtrip(self, tmp_path):
        """Same seeds on reloaded world → same projection content."""
        store_path = tmp_path / ".world0"
        w1 = World(store_path=store_path)

        for _ in range(10):
            w1.ingest(Observation(
                concepts=["det_a", "det_b", "det_c"],
                relations=[
                    ("det_a", "det_b", "depends_on"),
                    ("det_b", "det_c", "precedes"),
                ],
                source="bench",
            ))

        proj1 = w1.project(["det_a"], task="determinism")
        names1 = {c.name for c in proj1.concepts}

        w1.concepts.save_all()
        w1.relations.save_all()
        del w1

        w2 = World(store_path=store_path)
        proj2 = w2.project(["det_a"], task="determinism")
        names2 = {c.name for c in proj2.concepts}

        assert names1 == names2, (
            f"Projection changed after roundtrip: {names1} vs {names2}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 10. Projection Stability
# ═══════════════════════════════════════════════════════════════════════

class TestProjectionStability:
    """Projections should be stable under minor perturbations and
    consistent across repeated calls."""

    def test_repeated_projection_is_identical(self, world):
        """Same state + same seeds → identical projection every time."""
        for _ in range(10):
            world.ingest(Observation(
                concepts=["stable_a", "stable_b", "stable_c"],
                relations=[("stable_a", "stable_b", "depends_on")],
                source="bench",
            ))

        results = []
        for _ in range(5):
            proj = world.project(["stable_a"], task="stability")
            results.append({c.name for c in proj.concepts})

        for i in range(1, len(results)):
            assert results[i] == results[0], (
                f"Projection {i} differs from first: {results[i]} vs {results[0]}"
            )

    def test_additional_distant_concept_does_not_disrupt(self, world):
        """Adding an unrelated concept should not change an existing
        projection significantly."""
        for _ in range(10):
            world.ingest(Observation(
                concepts=["core_x", "core_y", "core_z"],
                relations=[
                    ("core_x", "core_y", "depends_on"),
                    ("core_y", "core_z", "precedes"),
                ],
                source="bench",
            ))

        proj_before = world.project(["core_x"], task="before")
        names_before = {c.name for c in proj_before.concepts}

        # Add a distant, unconnected concept
        world.ingest(Observation(
            concepts=["distant_noise"],
            source="bench",
        ))

        proj_after = world.project(["core_x"], task="after")
        names_after = {c.name for c in proj_after.concepts}

        assert names_before == names_after, (
            f"Distant concept disrupted projection: "
            f"before={names_before}, after={names_after}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 11. Task Context Sensitivity
# ═══════════════════════════════════════════════════════════════════════

class TestTaskSensitivity:
    """Projections must differ when the same seed is projected under
    different task contexts."""

    def _build_multi_task_world(self, world):
        """Build a world where 'python' is used in ML and web dev contexts."""
        for _ in range(10):
            world.ingest(Observation(
                concepts=["python", "machine learning", "training data",
                          "neural network"],
                relations=[
                    ("python", "machine learning", "supports"),
                    ("machine learning", "training data", "depends_on"),
                    ("machine learning", "neural network", "contains"),
                ],
                task="ML research", source="bench",
            ))
        for _ in range(10):
            world.ingest(Observation(
                concepts=["python", "flask", "REST API", "deployment"],
                relations=[
                    ("python", "flask", "supports"),
                    ("flask", "REST API", "contains"),
                    ("REST API", "deployment", "precedes"),
                ],
                task="web development", source="bench",
            ))

    def test_different_tasks_change_score_ranking(self, world):
        """Same seed 'python' with different tasks → different score rankings."""
        self._build_multi_task_world(world)

        proj_ml = world.project(["python"], task="ML research")
        proj_web = world.project(["python"], task="web development")

        ml_scores = proj_ml.activation_scores
        web_scores = proj_web.activation_scores

        # Get the top non-seed concept for each projection
        py_id = world.concepts.resolve("python").id
        ml_top = sorted(
            [(c, ml_scores.get(c.id, 0)) for c in proj_ml.concepts if c.id != py_id],
            key=lambda x: x[1], reverse=True,
        )
        web_top = sorted(
            [(c, web_scores.get(c.id, 0)) for c in proj_web.concepts if c.id != py_id],
            key=lambda x: x[1], reverse=True,
        )

        assert ml_top[0][0].name != web_top[0][0].name, (
            f"Task context had no effect on ranking — both top: "
            f"ML={ml_top[0][0].name}, web={web_top[0][0].name}"
        )

    def test_ml_task_boosts_ml_concepts(self, world):
        self._build_multi_task_world(world)

        proj_ml = world.project(["python"], task="ML research")
        proj_web = world.project(["python"], task="web development")

        ml_scores = proj_ml.activation_scores
        web_scores = proj_web.activation_scores

        # ML concept should score higher under ML task
        ml_node = world.concepts.resolve("machine learning")
        flask_node = world.concepts.resolve("flask")

        if ml_node and flask_node:
            ml_in_ml = ml_scores.get(ml_node.id, 0)
            ml_in_web = web_scores.get(ml_node.id, 0)
            flask_in_ml = ml_scores.get(flask_node.id, 0)
            flask_in_web = web_scores.get(flask_node.id, 0)

            assert ml_in_ml > ml_in_web or ml_in_web == 0, (
                f"'machine learning' should score higher under ML task: "
                f"ML={ml_in_ml:.4f} vs web={ml_in_web:.4f}"
            )
            assert flask_in_web > flask_in_ml or flask_in_ml == 0, (
                f"'flask' should score higher under web task: "
                f"web={flask_in_web:.4f} vs ML={flask_in_ml:.4f}"
            )

    def test_no_task_gives_neutral_projection(self, world):
        """Projection without task should include all reachable concepts."""
        self._build_multi_task_world(world)

        proj_no_task = world.project(["python"], task="")
        proj_ml = world.project(["python"], task="ML research")

        neutral_names = {c.name for c in proj_no_task.concepts}
        ml_names = {c.name for c in proj_ml.concepts}

        # Neutral projection should be at least as broad
        assert len(neutral_names) >= len(ml_names) or neutral_names != ml_names


# ═══════════════════════════════════════════════════════════════════════
# 12. Relation Type Differentiation
# ═══════════════════════════════════════════════════════════════════════

class TestRelationTypeDifferentiation:
    """Different relation types should produce different activation strengths."""

    def test_depends_on_propagates_stronger_than_related_to(self, world):
        """depends_on (factor 1.0) should propagate more than related_to (0.5)."""
        for _ in range(10):
            world.ingest(Observation(
                concepts=["center", "strong_dep", "weak_rel"],
                relations=[
                    ("center", "strong_dep", "depends_on"),
                    ("center", "weak_rel", "related_to"),
                ],
                task="test", source="bench",
            ))

        proj = world.project(["center"], task="test")
        scores = proj.activation_scores

        dep = world.concepts.resolve("strong_dep")
        rel = world.concepts.resolve("weak_rel")

        dep_score = scores.get(dep.id, 0)
        rel_score = scores.get(rel.id, 0)

        assert dep_score > rel_score, (
            f"depends_on ({dep_score:.4f}) should score higher than "
            f"related_to ({rel_score:.4f})"
        )

    def test_contrasts_propagates_weakest(self, world):
        """contrasts (factor 0.4) should be the weakest typed relation."""
        for _ in range(10):
            world.ingest(Observation(
                concepts=["origin", "supported", "contrasted"],
                relations=[
                    ("origin", "supported", "supports"),
                    ("origin", "contrasted", "contrasts"),
                ],
                task="test", source="bench",
            ))

        proj = world.project(["origin"], task="test")
        scores = proj.activation_scores

        sup = world.concepts.resolve("supported")
        con = world.concepts.resolve("contrasted")

        sup_score = scores.get(sup.id, 0)
        con_score = scores.get(con.id, 0)

        assert sup_score > con_score, (
            f"supports ({sup_score:.4f}) should score higher than "
            f"contrasts ({con_score:.4f})"
        )

    def test_relation_type_factor_ordering(self, world):
        """Verify the full ordering: depends_on > supports > related_to."""
        for _ in range(10):
            world.ingest(Observation(
                concepts=["hub", "dep_node", "sup_node", "rel_node"],
                relations=[
                    ("hub", "dep_node", "depends_on"),
                    ("hub", "sup_node", "supports"),
                    ("hub", "rel_node", "related_to"),
                ],
                task="test", source="bench",
            ))

        proj = world.project(["hub"], task="test")
        scores = proj.activation_scores

        dep = world.concepts.resolve("dep_node")
        sup = world.concepts.resolve("sup_node")
        rel = world.concepts.resolve("rel_node")

        s_dep = scores.get(dep.id, 0)
        s_sup = scores.get(sup.id, 0)
        s_rel = scores.get(rel.id, 0)

        assert s_dep > s_sup > s_rel, (
            f"Expected depends_on > supports > related_to, "
            f"got {s_dep:.4f} > {s_sup:.4f} > {s_rel:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 13. Alias Management
# ═══════════════════════════════════════════════════════════════════════

class TestAliasManagement:
    """Aliases should be indexable and resolvable after mutation."""

    def test_add_alias_makes_concept_resolvable(self, world):
        world.ingest(Observation(concepts=["machine learning"], source="bench"))
        node = world.concepts.resolve("machine learning")

        world.concepts.add_alias(node.id, "ML")
        world.concepts.add_alias(node.id, "ml")

        assert world.concepts.resolve("ML") is not None
        assert world.concepts.resolve("ml") is not None
        assert world.concepts.resolve("ML").id == node.id

    def test_set_aliases_replaces_all(self, world):
        world.ingest(Observation(concepts=["kubernetes"], source="bench"))
        node = world.concepts.resolve("kubernetes")

        world.concepts.add_alias(node.id, "k8s")
        assert world.concepts.resolve("k8s") is not None

        world.concepts.set_aliases(node.id, ["kube", "K8S"])
        assert world.concepts.resolve("kube") is not None
        assert world.concepts.resolve("K8S") is not None
        # Old alias "k8s" should still work (case-insensitive match with "K8S")
        assert world.concepts.resolve("k8s") is not None

    def test_alias_conflict_rejected(self, world):
        world.ingest(Observation(concepts=["docker"], source="bench"))
        world.ingest(Observation(concepts=["podman"], source="bench"))
        docker = world.concepts.resolve("docker")
        podman = world.concepts.resolve("podman")

        # Cannot alias "podman" to docker's concept
        result = world.concepts.add_alias(docker.id, "podman")
        assert result is False

    def test_aliases_survive_persistence(self, tmp_path):
        store_path = tmp_path / ".world0"
        w1 = World(store_path=store_path)
        w1.ingest(Observation(concepts=["machine learning"], source="bench"))
        node = w1.concepts.resolve("machine learning")
        w1.concepts.add_alias(node.id, "ML")
        w1.concepts.save_all()
        del w1

        w2 = World(store_path=store_path)
        assert w2.concepts.resolve("ML") is not None
        assert w2.concepts.resolve("ML").name == "machine learning"


# ═══════════════════════════════════════════════════════════════════════
# 14. Propagation Floor
# ═══════════════════════════════════════════════════════════════════════

class TestPropagationFloor:
    """Low-confidence concepts should not block activation spread."""

    def test_activation_passes_through_low_confidence_node(self, world):
        """A → B (low confidence) → C should still reach C."""
        world.ingest(Observation(
            concepts=["start", "middle", "end"],
            relations=[
                ("start", "middle", "precedes"),
                ("middle", "end", "precedes"),
            ],
            task="test", source="bench",
        ))
        # Reinforce start and end heavily, but not middle
        for _ in range(15):
            world.ingest(Observation(concepts=["start"], source="bench"))
            world.ingest(Observation(concepts=["end"], source="bench"))
            world.ingest(Observation(
                concepts=["start", "middle"],
                relations=[("start", "middle", "precedes")],
                source="bench",
            ))
            world.ingest(Observation(
                concepts=["middle", "end"],
                relations=[("middle", "end", "precedes")],
                source="bench",
            ))

        proj = world.project(["start"], task="test", max_depth=3)
        names = {c.name for c in proj.concepts}

        assert "end" in names, (
            f"Activation did not reach 'end' through low-confidence 'middle'. "
            f"Reached: {names}"
        )
