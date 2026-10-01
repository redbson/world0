"""A task-conditioned projection is not filled with another task's concepts.

Round 27 (analysis doc §7.29).  ``max_concepts`` is a ceiling, not a
target: a candidate whose *known* context is another task (a task profile
whose affinity for the current task is below ``CONTEXT_MATCH``) and whose
activation sits in the floor band (kept by the activation engine for
horizon completeness, not on the strength of its evidence) is filler and
is never selected.  Off-task concepts the seeds reach strongly still enter
on their merit; without a task, or when no candidate is in the task's
context, nothing is filler.
"""

from __future__ import annotations

import pytest

from world0 import Observation, World
from world0.dynamics.coefficients import PROPAGATION_MIN_RATIO


def _names(projection) -> set[str]:
    return {c.name for c in projection.concepts}


@pytest.fixture
def two_domains(tmp_path):
    """Two chains that share one polysemous bridge concept.

    alpha work:  a0 - bridge - a1 - a2 - a3 - a4
    beta work:   b0 - bridge - b1 - b2 - b3 - b4
    """
    w = World(store_path=tmp_path / ".world0")
    for _ in range(6):
        w.ingest(Observation(
            concepts=["a0", "bridge", "a1", "a2", "a3", "a4"],
            relations=[("a0", "bridge", "depends_on"), ("bridge", "a1", "depends_on"),
                       ("a1", "a2", "depends_on"), ("a2", "a3", "depends_on"), ("a3", "a4", "depends_on")],
            task="alpha work", source="s"))
        w.ingest(Observation(
            concepts=["b0", "bridge", "b1", "b2", "b3", "b4"],
            relations=[("b0", "bridge", "depends_on"), ("bridge", "b1", "depends_on"),
                       ("b1", "b2", "depends_on"), ("b2", "b3", "depends_on"), ("b3", "b4", "depends_on")],
            task="beta work", source="s"))
    return w


class TestOffTaskFiller:
    def test_the_other_domain_does_not_fill_the_view(self, two_domains):
        """Under "alpha work" the far beta chain is filler; the alpha chain fills the view."""
        proj = two_domains.project(["a0"], task="alpha work", max_concepts=15, max_depth=4)
        names = _names(proj)
        assert {"a0", "bridge", "a1", "a2", "a3"} <= names
        # b2..b4 are reached only through the bridge, in the floor band, and
        # their known context is beta work: never selected as filler.
        assert not ({"b2", "b3", "b4"} & names), names
        assert len(proj.concepts) < 15

    def test_a_strongly_reached_off_task_neighbour_still_enters(self, two_domains):
        """The bridge's direct beta neighbour is reached above the floor band and may enter on merit."""
        proj = two_domains.project(["bridge"], task="alpha work", max_concepts=15, max_depth=3)
        names = _names(proj)
        assert {"bridge", "a1", "a0"} <= names
        peak = max(proj.activation_scores.values())
        strong_beta = {c.name for c in proj.concepts
                       if c.name.startswith("b") and proj.activation_scores[c.id] >= PROPAGATION_MIN_RATIO * peak}
        weak_beta = {c.name for c in proj.concepts
                     if c.name.startswith("b") and proj.activation_scores[c.id] < PROPAGATION_MIN_RATIO * peak}
        assert strong_beta, "a direct neighbour of the seed is not filler"
        assert not weak_beta, weak_beta

    def test_without_a_task_the_view_is_filled_as_before(self, two_domains):
        proj = two_domains.project(["a0"], max_concepts=15, max_depth=4)
        assert len(proj.concepts) >= 9  # both chains are reachable and nothing is off-task

    def test_a_task_nobody_was_in_fills_as_before(self, two_domains):
        """No candidate is in a never-seen task's context: nothing is filler (wrong-label robustness)."""
        proj = two_domains.project(["a0"], task="gamma work", max_concepts=15, max_depth=4)
        names = _names(proj)
        assert {"b2", "b3"} & names, names

    def test_concepts_without_a_task_profile_are_neutral(self, tmp_path):
        """A concept never activated under any task has no known context and is not filler."""
        w = World(store_path=tmp_path / ".world0")
        for _ in range(6):
            w.ingest(Observation(concepts=["x0", "x1"], relations=[("x0", "x1", "depends_on")],
                                 task="alpha work", source="s"))
            # the far chain was observed without any task label
            w.ingest(Observation(concepts=["x1", "x2", "x3", "x4"],
                                 relations=[("x1", "x2", "depends_on"), ("x2", "x3", "depends_on"),
                                            ("x3", "x4", "depends_on")], source="s"))
        proj = w.project(["x0"], task="alpha work", max_concepts=15, max_depth=4)
        assert {"x0", "x1", "x2", "x3", "x4"} <= _names(proj)

    def test_seeds_are_never_filler(self, two_domains):
        proj = two_domains.project(["a0", "b4"], task="alpha work", max_concepts=15, max_depth=2)
        assert {"a0", "b4"} <= _names(proj)
