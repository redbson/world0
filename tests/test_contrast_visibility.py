"""Round 28: a stated contrast brings its partner into view; relation aliases say what was stated.

Negative-axis claims used to *only* inhibit: "A conflicts with B" stated
once left B out of A's projection, so the view could not show the one
thing the Agent was told about the pair (analysis doc §7.30).  A negative
edge now gives its partner *visibility* — it appears at the strength of
the stated contrast, as a terminal: activation does not spread onward
from it — while the inhibition channel still suppresses excitation the
partner receives through other paths: ``net = max(visibility,
excitation − inhibition)``.

Aliases: ``contrasts`` is the new, weaker ``contrast`` relation (not
``conflict``); ``part_of`` is ``inclusion`` seen from the part (stored with
the whole as source, like ``precedes``); ``mutual_understanding`` is
``recursive_co_modeling`` (not ``equivalence``); the bare axis word
``negative`` is ``contrast``, the weakest negative claim.
"""

from __future__ import annotations

import pytest

from world0 import Observation, World
from world0.schemas.relation import (
    RELATION_PHRASES,
    SEMANTIC_RELATION_SPECS,
    RelationEdge,
    RelationType,
    normalize_semantic_relation,
    orient_relation,
    relation_phrase,
    semantic_relation_spec,
)


def _claim_lines(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        if line.startswith("- ") and "(belief" in line:
            out.append(line[2:].split(" (belief")[0])
        elif line and not line.startswith(("- ", "#", "Context for", "Also relevant", "Definitions")):
            break
    return out


@pytest.fixture
def world(tmp_path):
    return World(store_path=tmp_path / ".world0")


class TestContrastVisibility:
    def test_a_stated_conflict_brings_its_partner_into_view(self, world):
        for _ in range(4):
            world.ingest(Observation(concepts=["lock", "async io"],
                                     relations=[("lock", "async io", "conflict")], task="t", source="s"))
        proj = world.project(["lock"], task="t")
        names = {c.name for c in proj.concepts}
        assert "async io" in names
        assert "lock conflicts with async io" in _claim_lines(proj.render())

    def test_the_partner_is_a_terminal(self, world):
        """The contrast partner's own neighbourhood is not pulled in through it."""
        # one pair per observation: co-mentioning all four would also link
        # lock to the far concepts by co-occurrence
        for _ in range(4):
            world.ingest(Observation(concepts=["lock", "async io"],
                                     relations=[("lock", "async io", "conflict")], task="t", source="s"))
            world.ingest(Observation(concepts=["async io", "event loop"],
                                     relations=[("async io", "event loop", "depends_on")], task="t", source="s"))
            world.ingest(Observation(concepts=["event loop", "callback"],
                                     relations=[("event loop", "callback", "depends_on")], task="t", source="s"))
        proj = world.project(["lock"], task="t", max_depth=3)
        names = {c.name for c in proj.concepts}
        assert "async io" in names
        assert not ({"event loop", "callback"} & names), names

    def test_visibility_is_below_the_seed_and_below_a_dependence(self, world):
        for _ in range(6):
            world.ingest(Observation(concepts=["origin", "needed", "contrasted"],
                                     relations=[("origin", "needed", "depends_on"),
                                                ("origin", "contrasted", "conflict")],
                                     task="t", source="s"))
        proj = world.project(["origin"], task="t")
        s = proj.activation_scores
        origin = world.concepts.resolve("origin").id
        needed = world.concepts.resolve("needed").id
        contrasted = world.concepts.resolve("contrasted").id
        assert s[origin] > s[needed] > s[contrasted] > 0

    def test_a_stated_contrast_outranks_a_weak_indirect_excitation(self, world):
        """A direct contrast is knowledge about the pair: it shows the partner at the contrast's strength."""
        for _ in range(6):  # one pair per observation: no co-occurrence edge root–target
            world.ingest(Observation(concepts=["root", "mid"],
                                     relations=[("root", "mid", "depends_on")], task="t", source="s"))
            world.ingest(Observation(concepts=["mid", "target"],
                                     relations=[("mid", "target", "depends_on")], task="t", source="s"))
        before = world.project(["root"], task="t", max_depth=2).activation_scores
        target = world.concepts.resolve("target").id
        for _ in range(6):
            world.ingest(Observation(concepts=["root", "target"],
                                     relations=[("root", "target", "conflict")], task="t", source="s"))
        proj = world.project(["root"], task="t", max_depth=2)
        after = proj.activation_scores
        root = world.concepts.resolve("root").id
        assert before[target] < after[target] < after[root]
        assert "root conflicts with target" in _claim_lines(proj.render())

    def test_inhibition_still_suppresses_excitation_from_other_paths(self, tmp_path):
        """Contrasted *and* depended upon: the contrast lowers the partner's score (it does not add).

        Control: the same stream whose last observation co-mentions the pair
        without stating the contrast (co-mention alone reinforces the edge).
        """
        def build(with_contrast: bool):
            w = World(store_path=tmp_path / ("c" if with_contrast else "n"))
            for _ in range(8):
                w.ingest(Observation(concepts=["root", "target"],
                                     relations=[("root", "target", "depends_on")], task="t", source="s"))
            w.ingest(Observation(concepts=["root", "target"],
                                 relations=[("root", "target", "conflict")] if with_contrast else [],
                                 task="t", source="s"))
            scores = w.project(["root"], task="t").activation_scores
            return scores[w.concepts.resolve("target").id]
        control, contrasted = build(False), build(True)
        assert 0 < contrasted < control

    def test_a_contrast_partner_can_still_be_excited_through_a_positive_path(self, world):
        for _ in range(6):
            world.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "conflict")], task="t", source="s"))
            world.ingest(Observation(concepts=["a", "c"], relations=[("a", "c", "depends_on")], task="t", source="s"))
            world.ingest(Observation(concepts=["c", "b"], relations=[("c", "b", "depends_on")], task="t", source="s"))
        proj = world.project(["a"], task="t", max_depth=2)
        names = {c.name for c in proj.concepts}
        assert {"a", "b", "c"} <= names


class TestContrastRelation:
    def test_contrast_is_a_weaker_negative_claim_than_conflict(self):
        spec = semantic_relation_spec("contrast")
        conflict = semantic_relation_spec("conflict")
        assert spec.axis is RelationType.NEGATIVE
        assert spec.structural_strength < conflict.structural_strength
        assert spec.propagation_strength < conflict.propagation_strength
        assert "contrast" in RELATION_PHRASES and set(RELATION_PHRASES) == set(SEMANTIC_RELATION_SPECS)

    def test_contrasts_alias_no_longer_reads_as_conflict(self):
        assert normalize_semantic_relation("contrasts") == "contrast"
        assert normalize_semantic_relation("negative") == "contrast"
        assert normalize_semantic_relation("repulsion") == "contrast"
        assert relation_phrase("contrasts") == "contrasts with"

    def test_contrast_is_symmetric_and_opposes_a_positive_claim(self):
        edge = RelationEdge(source_id="a", target_id="b", relation_type="negative",
                            semantic_relation="contrast", is_explicit=True)
        assert not edge.is_directed or edge.connects("b", "a", directed=True)
        assert edge.opposes(RelationType.POSITIVE, "dependence")

    def test_mutual_understanding_is_co_modeling_not_equivalence(self):
        assert normalize_semantic_relation("mutual_understanding") == "recursive_co_modeling"


class TestPartOf:
    def test_part_of_is_inclusion_seen_from_the_part(self):
        assert normalize_semantic_relation("part_of") == "inclusion"
        assert orient_relation("wheel", "car", "part_of") == ("car", "wheel", "inclusion")

    def test_a_part_of_claim_renders_as_the_whole_containing_the_part(self, world):
        world.ingest(Observation(concepts=["wheel", "car"], relations=[("wheel", "car", "part_of")]))
        text = world.project(["wheel", "car"], max_concepts=3).render()
        assert "car contains wheel" in _claim_lines(text)
