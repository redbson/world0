"""Floor-band candidates fill a projection only after the candidates above the band.

Round 28b (analysis doc §7.31).  A candidate the activation engine lifted
into the floor band (``< PROPAGATION_MIN_RATIO × peak``) carries no
evidential strength to trade against diversity — the band keeps only its
order — so MMR runs over the candidates above the band and the band fills
whatever is left, in activation order.  Before, the redundancy term
dominated the band's tiny relevance differences and a far, "diverse"
third-hop concept displaced a strongly reached second-hop one.
"""

from __future__ import annotations

from world0 import Observation, World
from world0.dynamics.coefficients import PROPAGATION_MIN_RATIO


def _world(tmp_path) -> World:
    w = World(store_path=tmp_path / ".world0")
    # A tight, redundant cluster around the seed: three well-evidenced
    # neighbours that all also depend on each other.
    for _ in range(6):
        for a, b in (("hub", "n1"), ("hub", "n2"), ("hub", "n3"),
                     ("n1", "n2"), ("n2", "n3"), ("n1", "n3")):
            w.ingest(Observation(concepts=[a, b], relations=[(a, b, "depends_on")], source="s"))
    # A far chain stated once: its tail sits in the floor band and shares no
    # neighbour with the cluster, so the redundancy term favours it.
    for a, b in (("hub", "c1"), ("c1", "c2"), ("c2", "c3"), ("c3", "c4")):
        w.ingest(Observation(concepts=[a, b], relations=[(a, b, "depends_on")], source="s"))
    return w


def _bands(w: World, seed: str, depth: int) -> tuple[set[str], set[str]]:
    sid = w.concepts.resolve(seed).id
    acts = w._activation.activate([sid], max_depth=depth, record=False)
    peak = max(acts.values())
    above = {w.concepts.get(c).name for c, a in acts.items() if a >= PROPAGATION_MIN_RATIO * peak}
    band = {w.concepts.get(c).name for c, a in acts.items() if a < PROPAGATION_MIN_RATIO * peak}
    return above, band


def test_the_band_never_displaces_a_candidate_above_it(tmp_path):
    w = _world(tmp_path)
    above, band = _bands(w, "hub", depth=4)
    assert band, "the fixture must put something in the floor band"
    for k in range(2, len(above)):
        proj = w.project(["hub"], max_concepts=k, max_depth=4)
        names = {c.name for c in proj.concepts}
        assert len(names) == k
        assert not (names & band), (k, names & band, above)
        assert names <= above


def test_the_band_fills_what_is_left_in_activation_order(tmp_path):
    w = _world(tmp_path)
    above, band = _bands(w, "hub", depth=4)
    proj = w.project(["hub"], max_concepts=len(above) + 1, max_depth=4)
    names = {c.name for c in proj.concepts}
    assert above <= names
    picked = names & band
    assert len(picked) == 1
    sid = w.concepts.resolve("hub").id
    acts = w._activation.activate([sid], max_depth=4, record=False)
    best_band = max(band, key=lambda n: acts[w.concepts.resolve(n).id])
    assert picked == {best_band}
