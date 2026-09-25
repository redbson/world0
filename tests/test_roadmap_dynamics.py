"""Behavioral tests for the roadmap items implemented after the dynamics
analysis (``docs/world0-cognitive-dynamics-analysis.md`` §7):

- §7.2 recurrence-based promotion (spaced repetition beats bursts)
- §7.3 explicit re-observation confirms a relation's semantic probability
- §7.4 perspective-level directional propagation
- §7.6 token-indexed synonym shortlist (same decisions, no full scan)
- §7.8 light reflect and ``auto_reflect_every`` continuous mode
- §7.1 salience = freshness ∨ evidence-backed persistence (single clock)
"""

from __future__ import annotations

import pytest

from tests._cognitive_benchmark import ranked_projection_names
from world0 import Observation, Perspective, World
from world0.schemas.concept import (
    RECURRENCE_WINDOW,
    ConceptNode,
    Maturity,
)
from world0.schemas.relation import RelationEdge


@pytest.fixture
def world(tmp_path):
    return World(store_path=tmp_path / ".world0")


def _tick(world: World) -> None:
    world._decay.decay_concepts()
    world._decay.decay_relations()
    world._lifecycle.evaluate()


# ═══════════════════════════════════════════════════════════════════════
# §7.2 Recurrence
# ═══════════════════════════════════════════════════════════════════════


class TestRecurrence:
    def test_burst_in_one_window_counts_once(self, world):
        # 20 observations all fall inside the first RECURRENCE_WINDOW ticks.
        for _ in range(20):
            world.ingest(Observation(concepts=["burst"], source="s"))
        node = world.concepts.resolve("burst")
        assert node.activation_count == 20
        assert node.recurrence_count == 1

    def test_spaced_mentions_count_per_window(self, world):
        for _ in range(5):
            world.ingest(Observation(concepts=["spaced"], source="s"))
            world.clock.advance(RECURRENCE_WINDOW)
        node = world.concepts.resolve("spaced")
        assert node.activation_count == 5
        assert node.recurrence_count == 5

    def test_sparse_but_regular_concept_reaches_established(self, world):
        """Re-observed every 168 observations: the confidence equilibrium
        (≈0.36) never crosses the 0.6 gate, but a year of recurrence does."""
        for _ in range(52):
            world.ingest(
                Observation(
                    concepts=["weekly", "anchor"],
                    relations=[("weekly", "anchor", "depends_on")],
                    source="s",
                )
            )
            world.clock.advance(167)
            _tick(world)
        node = world.concepts.resolve("weekly")
        assert node.recurrence_count >= 10
        assert node.maturity == Maturity.ESTABLISHED

    def test_burst_alone_does_not_use_the_recurrence_gate(self, world):
        node = ConceptNode(name="x", activation_count=30, recurrence_count=1, confidence=0.2)
        world.concepts._concepts[node.id] = node  # type: ignore[attr-defined]
        promoted, _ = world._lifecycle.evaluate()
        assert node.id not in promoted
        assert node.maturity == Maturity.EMBRYONIC

    def test_legacy_record_defaults(self):
        node = ConceptNode.model_validate({"name": "legacy"})
        assert node.recurrence_count == 0
        assert node.last_recurrence_window == -1

    def test_merge_keeps_max_recurrence(self, world):
        for _ in range(4):
            world.ingest(Observation(concepts=["k"], source="s"))
            world.clock.advance(RECURRENCE_WINDOW)
        world.ingest(Observation(concepts=["dup"], source="s"))
        assert world.merge("k", "dup")
        assert world.concepts.resolve("k").recurrence_count == 4


# ═══════════════════════════════════════════════════════════════════════
# §7.3 Relation confirmation
# ═══════════════════════════════════════════════════════════════════════


class TestRelationConfirmation:
    def _edge(self, world):
        return world.relations.all()[0]

    def test_explicit_restatement_raises_probability(self, world):
        obs = Observation(
            concepts=["p", "q"],
            relations=[("p", "q", "depends_on")],
            task="t",
            source="s",
        )
        world.ingest(obs)
        before = self._edge(world).probability
        for _ in range(5):
            world.ingest(obs)
        edge = self._edge(world)
        assert edge.probability > before
        assert edge.probability_observation_count == 5
        assert edge.reinforcement_count >= 5

    def test_hebbian_cooccurrence_does_not_touch_probability(self, world):
        world.ingest(
            Observation(concepts=["a", "b"], relations=[("a", "b", "depends_on")], source="s")
        )
        edge = self._edge(world)
        p0 = edge.probability
        # Co-occurrence without restating the relation: Hebbian only.
        for _ in range(5):
            world.ingest(Observation(concepts=["a", "b"], source="s"))
        edge = self._edge(world)
        assert edge.probability == pytest.approx(p0)
        assert edge.reinforcement_count >= 5

    def test_confirm_has_diminishing_returns_and_is_bounded(self):
        edge = RelationEdge(source_id="a", target_id="b", semantic_relation="dependence")
        steps = []
        for _ in range(200):
            before = edge.probability
            edge.confirm()
            steps.append(edge.probability - before)
        assert all(later <= earlier + 1e-12 for earlier, later in zip(steps, steps[1:]))
        assert edge.probability <= 1.0

    def test_adjust_strength_moves_probability_by_delta(self, world):
        world.ingest(
            Observation(concepts=["a", "b"], relations=[("a", "b", "depends_on")], source="s")
        )
        edge = self._edge(world)
        p0 = edge.probability
        world.relations.adjust_strength(edge.id, weight_delta=-0.1, confidence_delta=-0.1)
        assert edge.probability == pytest.approx(p0 - 0.1)
        assert edge.probability != edge.confidence


# ═══════════════════════════════════════════════════════════════════════
# §7.4 Directional propagation
# ═══════════════════════════════════════════════════════════════════════


class TestDirectionalPropagation:
    @pytest.fixture
    def chain(self, world):
        for _ in range(6):
            world.ingest(
                Observation(
                    concepts=["service", "database"],
                    relations=[("service", "database", "depends_on")],
                    task="t",
                    source="s",
                )
            )
        return world

    def test_default_perspective_is_undirected(self, chain):
        fwd = chain.project(["service"], task="t")
        bwd = chain.project(["database"], task="t")
        db = chain.concepts.resolve("database")
        svc = chain.concepts.resolve("service")
        assert fwd.activation_scores[db.id] > 0
        assert bwd.activation_scores[svc.id] > 0

    def test_backward_weight_suppresses_who_depends_on_me(self, chain):
        downstream_only = Perspective(
            name="what do I depend on", direction_weights={"backward": 0.1}
        )
        neutral = Perspective(name="neutral")
        svc = chain.concepts.resolve("service")
        db = chain.concepts.resolve("database")

        # From the dependant: forward traversal is untouched.
        forward_neutral = chain.project(["service"], perspective=neutral)
        forward_biased = chain.project(["service"], perspective=downstream_only)
        assert forward_biased.activation_scores[db.id] == pytest.approx(
            forward_neutral.activation_scores[db.id]
        )

        # From the dependency: backward traversal is damped.
        backward_neutral = chain.project(["database"], perspective=neutral)
        backward_biased = chain.project(["database"], perspective=downstream_only)
        assert backward_biased.activation_scores[svc.id] < backward_neutral.activation_scores[svc.id]

    def test_direction_weight_defaults_are_neutral(self):
        p = Perspective()
        assert p.weight_for_direction("forward") == 1.0
        assert p.weight_for_direction("backward") == 1.0
        q = Perspective(direction_weights={"forward": 1.5})
        assert q.weight_for_direction("forward") == 1.5
        assert q.weight_for_direction("backward") == 1.0


# ═══════════════════════════════════════════════════════════════════════
# §7.6 Indexed synonym shortlist
# ═══════════════════════════════════════════════════════════════════════


class TestSynonymShortlist:
    def test_alias_synonym_still_resolves_to_existing_concept(self, world):
        first, is_new = world.concepts.get_or_create(
            "PostgreSQL",
            kind="database",
            sense="relational database engine",
            description="open source relational database",
        )
        assert is_new
        second, is_new = world.concepts.get_or_create(
            "Postgres",
            aliases=["PostgreSQL"],
            kind="database",
            sense="relational database engine",
            description="open source relational database",
        )
        assert not is_new
        assert second.id == first.id

    def test_sense_only_overlap_is_found(self, world):
        first, _ = world.concepts.get_or_create(
            "Apple", kind="company", sense="consumer electronics technology company cupertino"
        )
        second, is_new = world.concepts.get_or_create(
            "Apple Inc",
            aliases=["Apple"],
            kind="company",
            sense="consumer electronics technology company cupertino",
        )
        assert not is_new
        assert second.id == first.id

    def test_distinct_senses_stay_distinct(self, world):
        company, _ = world.concepts.get_or_create(
            "Apple", kind="company", sense="technology company"
        )
        fruit, is_new = world.concepts.get_or_create(
            "Apple", kind="fruit", sense="edible pome fruit"
        )
        assert is_new
        assert fruit.id != company.id

    def test_untokenizable_probe_falls_back_safely(self, world):
        a, _ = world.concepts.get_or_create("认知", kind="概念", sense="理解")
        b, is_new = world.concepts.get_or_create("认知系统", kind="概念", sense="理解")
        assert a is not None and b is not None
        assert isinstance(is_new, bool)

    def test_shortlist_ignores_unrelated_concepts(self, world):
        for i in range(50):
            world.concepts.get_or_create(f"noise{i}", kind="thing", sense=f"unrelated sense {i}")
        target, _ = world.concepts.get_or_create(
            "Kubernetes", kind="platform", sense="container orchestration platform"
        )
        found, is_new = world.concepts.get_or_create(
            "K8s",
            aliases=["Kubernetes"],
            kind="platform",
            sense="container orchestration platform",
        )
        assert not is_new
        assert found.id == target.id


# ═══════════════════════════════════════════════════════════════════════
# §7.8 Light reflect / continuous mode
# ═══════════════════════════════════════════════════════════════════════


class TestContinuousMode:
    def test_light_reflect_decays_but_skips_structure_passes(self, world):
        for _ in range(3):
            world.ingest(
                Observation(
                    concepts=["a", "b", "c"],
                    relations=[("a", "b", "depends_on"), ("b", "c", "depends_on")],
                    source="s",
                )
            )
        node = world.concepts.resolve("a")
        before = node.confidence
        world.clock.advance(48)
        result = world.reflect(light=True)
        assert node.confidence < before
        assert result.new_communities == []
        assert result.color_sources == []
        assert world.status().last_reflect is None  # only a full reflect stamps it

    def test_auto_reflect_every_consolidates_without_explicit_calls(self, tmp_path):
        world = World(store_path=tmp_path / "auto", auto_reflect_every=10)
        world.ingest(Observation(concepts=["early"], source="s"))
        early = world.concepts.resolve("early")
        initial = early.confidence
        for i in range(30):
            world.ingest(Observation(concepts=[f"other{i}"], source="s"))
        assert world.clock.tick == 31
        assert early.last_decayed_tick == 30
        assert early.confidence < initial

    def test_auto_reflect_disabled_by_default(self, world):
        world.ingest(Observation(concepts=["early"], source="s"))
        early = world.concepts.resolve("early")
        for i in range(30):
            world.ingest(Observation(concepts=[f"other{i}"], source="s"))
        assert early.last_decayed_tick is None


# ═══════════════════════════════════════════════════════════════════════
# §7.5 Relative activation cut / §7.1 evidence & salience accessors
# ═══════════════════════════════════════════════════════════════════════


class TestRelativeCutAndEvidence:
    def test_weak_seed_keeps_its_horizon(self, world):
        """A once-observed (confidence ≈ 0.2) seed used to lose everything
        beyond one hop to the absolute 0.01 cut; the relative cut keeps the
        same chain length a confident seed gets."""
        names = [f"w{i}" for i in range(5)]
        for a, b in zip(names, names[1:]):
            world.ingest(Observation(concepts=[a, b], relations=[(a, b, "depends_on")], source="s"))
        ids = {n: world.concepts.resolve(n).id for n in names}
        act = world._activation.activate([ids["w0"]], max_depth=4, decay=0.5, record=False)
        assert world.concepts.resolve("w0").confidence < 0.35
        assert all(ids[n] in act for n in names), sorted(n for n in names if ids[n] not in act)
        scores = [act[ids[n]] for n in names]
        assert all(a > b for a, b in zip(scores, scores[1:]))
        proj = world.project(["w0"], max_depth=4, max_concepts=10)
        assert {c.name for c in proj.concepts} == set(names)

    def test_relative_cut_never_tightens_the_absolute_one(self, world):
        for _ in range(10):
            world.ingest(Observation(concepts=["s", "t"], relations=[("s", "t", "depends_on")], source="x"))
        s = world.concepts.resolve("s")
        act = world._activation.activate([s.id], max_depth=1, decay=0.5, record=False, min_activation=0.5)
        # Strong seed (≈1.0): cut = min(0.5, 0.02) → weak neighbours are still
        # accepted relative to the seed, never rejected below the absolute.
        assert len(act) >= 1

    def test_evidence_and_salience_are_independent(self, world):
        for _ in range(12):
            world.ingest(Observation(concepts=["e"], source="s"))
        node = world.concepts.resolve("e")
        evidence_now = node.evidence()
        salience_now = node.salience(now_tick=world.clock.tick)
        world.clock.advance(500)
        assert node.evidence() == pytest.approx(evidence_now)  # time does not touch evidence
        assert node.salience(now_tick=world.clock.tick) < salience_now
        world.concepts.weaken(node.id)
        assert node.evidence() < evidence_now  # disconfirmation does
        assert 0.0 < node.evidence() < 1.0
        assert ConceptNode(name="never").evidence() == 0.0

    def test_render_exposes_evidence_next_to_confidence(self, world):
        for _ in range(3):
            world.ingest(Observation(concepts=["r", "q"], relations=[("r", "q", "depends_on")], source="s"))
        text = world.project(["r"]).render()
        assert "confidence: " in text and "evidence: " in text


class TestSynonymShortlistScaling:
    def test_common_tokens_do_not_defeat_the_shortlist(self, world):
        """Every concept shares the words "system data component"; the rarity
        filter still finds the true synonym and still refuses the others."""
        for i in range(60):
            world.concepts.get_or_create(
                f"component {i}", kind="module", sense=f"system data component variant{i}"
            )
        target, _ = world.concepts.get_or_create(
            "Scheduler", kind="module", sense="system data component that orders jobs"
        )
        found, is_new = world.concepts.get_or_create(
            "Job Scheduler",
            aliases=["Scheduler"],
            kind="module",
            sense="system data component that orders jobs",
        )
        assert not is_new and found.id == target.id
        other, is_new = world.concepts.get_or_create(
            "Cache", kind="module", sense="system data component that memoizes results"
        )
        assert is_new and other.id != target.id

    def test_signature_cache_tracks_edits(self, world):
        node, _ = world.concepts.get_or_create("Alpha", kind="k", sense="first greek letter symbol")
        world.concepts._node_signature(node)  # type: ignore[attr-defined]
        world.concepts.update_description(node.id, "the very first greek letter")
        labels, tokens, sense = world.concepts._node_signature(node)  # type: ignore[attr-defined]
        assert "very" in tokens
        world.concepts.add_alias(node.id, "α-letter")
        labels, _, _ = world.concepts._node_signature(node)  # type: ignore[attr-defined]
        assert "α letter" in labels or "α-letter".lower() in {l.replace(" ", "-") for l in labels}


class TestNumericIdentityTokens:
    def test_generation_number_keeps_concepts_distinct(self, world):
        a, _ = world.concepts.get_or_create(
            "GPT 4", kind="model", sense="large language model generation 4 by openai"
        )
        b, is_new = world.concepts.get_or_create(
            "GPT 5", kind="model", sense="large language model generation 5 by openai"
        )
        assert is_new and b.id != a.id

    def test_single_digit_tokens_survive_tokenization(self):
        from world0.schemas.concept import tokenize_signature

        assert tokenize_signature("GPT 4 model") == {"gpt", "4", "model"}
        assert tokenize_signature("a b c 7") == {"7"}  # letters < 2 chars still dropped

    def test_true_synonym_with_numbers_still_merges(self, world):
        a, _ = world.concepts.get_or_create(
            "Python 3", kind="language", sense="python programming language version 3"
        )
        b, is_new = world.concepts.get_or_create(
            "Python3", aliases=["Python 3"], kind="language", sense="python programming language version 3"
        )
        assert not is_new and b.id == a.id


class TestProjectionDiversity:
    """A redundancy-sensitive scenario: six near-identical siblings hanging
    off the same anchors compete with a short chain of distinct concepts.
    Pure relevance ranking fills the projection with siblings; MMR must
    reserve room for the other region."""

    @pytest.fixture
    def sibling_world(self, world):
        siblings = [f"siblingA{i}" for i in range(6)]
        for _ in range(8):
            world.ingest(
                Observation(
                    concepts=["hub", "anchorA1", "anchorA2"] + siblings,
                    relations=[("hub", "anchorA1", "depends_on"), ("hub", "anchorA2", "depends_on")]
                    + [(s, "anchorA1", "depends_on") for s in siblings]
                    + [(s, "anchorA2", "depends_on") for s in siblings]
                    + [("hub", s, "supports") for s in siblings],
                    task="t",
                    source="s",
                )
            )
            world.ingest(
                Observation(
                    concepts=["hub", "b1", "b2", "b3"],
                    relations=[("hub", "b1", "depends_on"), ("b1", "b2", "depends_on"), ("b2", "b3", "depends_on")],
                    task="t",
                    source="s",
                )
            )
        return world

    def test_projection_covers_both_regions(self, sibling_world):
        names = {c.name for c in sibling_world.project(["hub"], task="t", max_concepts=6, max_depth=3).concepts}
        siblings = {n for n in names if n.startswith("siblingA")}
        chain = {n for n in names if n in {"b1", "b2", "b3"}}
        assert "hub" in names
        assert len(chain) >= 2, names
        assert len(siblings) <= 3, names

    def test_pure_relevance_would_have_filled_with_siblings(self, sibling_world):
        import world0.projection.engine as pe

        original = pe.MMR_LAMBDA
        try:
            pe.MMR_LAMBDA = 0.0
            names = {c.name for c in sibling_world.project(["hub"], task="t", max_concepts=6, max_depth=3).concepts}
        finally:
            pe.MMR_LAMBDA = original
        assert sum(n.startswith("siblingA") for n in names) >= 4


# ═══════════════════════════════════════════════════════════════════════
# §7.1 Salience: freshness ∨ evidence-backed persistence
# ═══════════════════════════════════════════════════════════════════════


class TestSaliencePersistence:
    """Time must be charged once per concept during activation.

    Before: a dormant neighbor paid for its age through the decayed
    ``confidence`` *and* the freshness term, so a dependency confirmed
    fifty times scored ~0.2× a same-observation one-off mention and never
    made a slot-limited projection.  Now readiness reads the stronger of
    confidence and evidence, and salience keeps an evidence-backed floor
    that forgets on the era scale.
    """

    @staticmethod
    def _dormant_veteran(world: World, dormant: int, rookies: int = 6):
        for _ in range(50):
            world.ingest(
                Observation(
                    concepts=["Seed", "Veteran"],
                    relations=[("Seed", "Veteran", "depends_on")],
                    source="s",
                )
            )
            world.clock.advance(23)
        world.clock.advance(dormant)
        world.reflect(light=True)
        for i in range(rookies):
            world.ingest(
                Observation(
                    concepts=["Seed", f"Rookie{i}"],
                    relations=[("Seed", f"Rookie{i}", "depends_on")],
                    source="s",
                )
            )
        seed = world.concepts.resolve("Seed").id
        scores = world._activation.activate([seed], record=False)
        veteran = scores[world.concepts.resolve("Veteran").id]
        best_rookie = max(
            scores[world.concepts.resolve(f"Rookie{i}").id] for i in range(rookies)
        )
        return veteran, best_rookie

    def test_confirmed_dependency_holds_a_slot_against_fresh_one_offs(self, world):
        veteran, rookie = self._dormant_veteran(world, dormant=500)
        assert veteran > rookie
        ranked = ranked_projection_names(world.project(["Seed"], max_concepts=4))
        assert "Veteran" in ranked

    def test_fresh_context_still_wins_after_long_dormancy(self, world):
        """Context changes relevance: persistence is a floor, not immunity."""
        veteran, rookie = self._dormant_veteran(world, dormant=3000)
        assert veteran < rookie

    def test_disconfirmation_lowers_propagation_through_dormant_neighbor(self, world):
        before, _ = self._dormant_veteran(world, dormant=500)
        node = world.concepts.resolve("Veteran")
        for _ in range(40):
            node.weaken(source="s")
        seed = world.concepts.resolve("Seed").id
        after = world._activation.activate([seed], record=False)[node.id]
        assert after < before

    def test_one_off_does_not_persist(self):
        node = ConceptNode(name="noise", activation_count=1, last_activated_tick=0)
        assert node.salience(now_tick=2000) == pytest.approx(0.1)
        assert node.salience(now_tick=2000) == node.temporal_relevance(now_tick=2000)

    def test_persistence_forgets_on_the_era_scale(self):
        node = ConceptNode(name="old", activation_count=50, last_activated_tick=0)
        assert node.salience(now_tick=500) > 0.5
        assert node.salience(now_tick=5000) < node.salience(now_tick=500)
        assert node.salience(now_tick=30_000) == pytest.approx(0.1)

    def test_salience_is_bounded_by_freshness_and_one(self):
        node = ConceptNode(name="x", activation_count=20, last_activated_tick=0)
        for t in (0, 10, 100, 1000, 5000, 20_000):
            s = node.salience(now_tick=t)
            assert node.temporal_relevance(now_tick=t) <= s <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# §7.11 Hebbian discovery gated on association strength
# ═══════════════════════════════════════════════════════════════════════


class TestHebbianAssociationGate:
    """Co-occurring twice is not a relation when both concepts are
    mentioned all the time.  With the count gate alone, 60 concepts
    observed six at a time linked 85 % of all pairs after 400 random
    observations; the Jaccard gate keeps that at ~9 % while every pair
    drawn from one topic is still linked (scripts/sweep_hebbian.py)."""

    POOL = [f"c{i}" for i in range(60)]

    def _generic_edges(self, world: World) -> int:
        return sum(1 for r in world.relations.all() if r.semantic_relation == "generic_relation")

    def test_random_co_mention_does_not_form_a_clique(self, world):
        import random

        rng = random.Random(7)
        for _ in range(400):
            world.ingest(Observation(concepts=rng.sample(self.POOL, 6), source="s"))
        total_pairs = 60 * 59 // 2
        assert self._generic_edges(world) < 0.15 * total_pairs

    def test_topic_mates_are_still_linked(self, world):
        import random

        rng = random.Random(7)
        topics = [self.POOL[i * 10 : (i + 1) * 10] for i in range(6)]
        for _ in range(400):
            world.ingest(Observation(concepts=rng.sample(rng.choice(topics), 6), source="s"))
        topic = topics[0]
        linked = 0
        for i, a in enumerate(topic):
            for b in topic[i + 1 :]:
                ia, ib = world.concepts.resolve(a).id, world.concepts.resolve(b).id
                linked += bool(world.relations.find_any_between(ia, ib))
        assert linked >= 0.95 * (10 * 9 // 2)

    def test_always_together_pair_links_at_threshold(self, world):
        world.ingest(Observation(concepts=["x", "y"], source="s"))
        result = world.ingest(Observation(concepts=["x", "y"], source="s"))
        assert result.hebbian_relations == ["x ↔ y"]

    def test_hub_is_not_linked_to_a_passing_acquaintance(self, world):
        # "hub" is in every observation; "rare" shares only two of forty.
        for i in range(40):
            others = [f"o{i}a", f"o{i}b"]
            if i in (10, 30):
                others.append("rare")
            world.ingest(Observation(concepts=["hub", *others], source="s"))
        hub, rare = world.concepts.resolve("hub"), world.concepts.resolve("rare")
        assert world.relations.find_any_between(hub.id, rare.id) == []
        assert world._hebbian.association(hub.id, rare.id) < 0.2

    def test_association_statistics_survive_restart(self, tmp_path):
        root = tmp_path / "heb"
        first = World(store_path=root)
        for i in range(5):
            first.ingest(Observation(concepts=["hub", f"o{i}"], source="s"))
        hub = first.concepts.resolve("hub").id
        assert first._hebbian.mentions(hub) == 5
        assert first._hebbian.observations == 5
        second = World(store_path=root)
        assert second._hebbian.mentions(hub) == 5
        assert second._hebbian.observations == 5

    def test_legacy_state_without_statistics_restores(self, world):
        world._hebbian.restore_stats(None)
        world._hebbian.restore_stats({"observations": "x", "mentions": {"a": "b", "c": 2}})
        assert world._hebbian.observations == 0
        assert world._hebbian.mentions("c") == 2
        assert world._hebbian.mentions("a") == 0
        # and the count-only behaviour holds until statistics accumulate
        world.ingest(Observation(concepts=["p", "q"], source="s"))
        assert world.ingest(Observation(concepts=["p", "q"], source="s")).hebbian_relations == ["p ↔ q"]

    def test_forget_concept_drops_mentions(self, world):
        world.ingest(Observation(concepts=["a", "b"], source="s"))
        a = world.concepts.resolve("a").id
        assert world._hebbian.mentions(a) == 1
        world._hebbian.forget_concept(a)
        assert world._hebbian.mentions(a) == 0


class TestHebbianRevalidation:
    """Generic edges linked on thin early statistics are re-judged by
    reflect once the statistics are meaningful (§7.11)."""

    POOL = [f"c{i}" for i in range(60)]

    def _generic(self, world: World):
        return [r for r in world.relations.all() if r.semantic_relation == "generic_relation"]

    def test_reflect_removes_stale_chance_edges(self, world):
        import random

        rng = random.Random(7)
        backbone = [("c1", "c2"), ("c3", "c4"), ("c5", "c6")]
        for _ in range(400):
            concepts = rng.sample(self.POOL, 6)
            rels = [(a, b, "depends_on") for a, b in backbone if a in concepts and b in concepts]
            world.ingest(Observation(concepts=concepts, relations=rels, source="s"))
        before_ids = {r.id for r in self._generic(world)}
        assert before_ids
        result = world.reflect(light=True)
        after_ids = {r.id for r in self._generic(world)}
        removed = before_ids - after_ids
        # Every removed generic edge was either revalidated away or decayed
        # below the prune threshold in the same cycle — nothing else.
        assert removed == set(result.stale_relations) | (set(result.pruned_relations) & before_ids)
        assert len(result.stale_relations) > 0.8 * len(before_ids)
        assert len(after_ids) < 0.2 * len(before_ids)
        # explicit relations are untouched
        explicit = [r for r in world.relations.all() if r.is_explicit]
        assert all(r.semantic_relation == "dependence" for r in explicit)
        assert explicit

    def test_thin_statistics_are_not_judged(self, world):
        world.ingest(Observation(concepts=["x", "y"], source="s"))
        world.ingest(Observation(concepts=["x", "y"], source="s"))
        assert len(self._generic(world)) == 1
        result = world.reflect(light=True)
        assert result.stale_relations == []
        assert len(self._generic(world)) == 1

    def test_well_associated_edges_survive(self, world):
        for _ in range(15):
            world.ingest(Observation(concepts=["a", "b"], source="s"))
            world.ingest(Observation(concepts=["c", "d"], source="s"))
        assert len(self._generic(world)) == 2
        result = world.reflect()
        assert result.stale_relations == []
        assert len(self._generic(world)) == 2

    def test_upgraded_edge_is_protected(self, world):
        """A generic edge later re-stated as a typed relation is no longer
        generic and is never revalidated away."""
        world.ingest(Observation(concepts=["p", "q"], source="s"))
        world.ingest(Observation(concepts=["p", "q"], source="s"))
        world.ingest(Observation(concepts=["p", "q"], relations=[("p", "q", "depends_on")], source="s"))
        for i in range(30):  # p and q now appear apart, association collapses
            world.ingest(Observation(concepts=["p", f"m{i}"], source="s"))
            world.ingest(Observation(concepts=["q", f"n{i}"], source="s"))
        world.reflect(light=True)
        p_, q_ = world.concepts.resolve("p"), world.concepts.resolve("q")
        edges = world.relations.find_any_between(p_.id, q_.id)
        assert edges and edges[0].semantic_relation == "dependence"
