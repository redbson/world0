"""Probe World 0 against consciousness-science indicator properties.

Indicators follow Butlin, Long et al. (2023), "Consciousness in Artificial
Intelligence: Insights from the Science of Consciousness".  Each check
asks whether a *functional* property is observable in World 0's behaviour;
none of this measures or implies experience.

Run from the repository root:  python docs/mc/probes/baseline_indicators.py
"""

from __future__ import annotations

import tempfile

from world0 import Observation, World

OPS = ["deployment", "kubernetes", "helm chart", "container registry", "rollout", "load balancer"]
ML = ["deployment", "pytorch", "gpu cluster", "training run", "checkpoint", "loss curve"]


def build() -> World:
    w = World(store_path=tempfile.mkdtemp() + "/w")
    for _ in range(12):
        w.ingest(Observation(concepts=OPS, relations=[("deployment", "kubernetes", "depends_on")], source="s"))
        w.ingest(Observation(concepts=ML, relations=[("deployment", "pytorch", "depends_on")], source="s"))
    for _ in range(6):
        w.ingest(Observation(concepts=["pytorch", "gpu cluster"], relations=[("pytorch", "gpu cluster", "enables")], source="s"))
    for _ in range(4):
        w.ingest(Observation(concepts=["pytorch", "gpu cluster"], relations=[("pytorch", "gpu cluster", "conflict")], source="s"))
    w.ingest(Observation(concepts=["quantum annealer", "pytorch"], relations=[("pytorch", "quantum annealer", "related_to")], source="s"))
    w.reflect()
    return w


def names(p) -> list[str]:
    return [c.name for c in p.concepts]


def main() -> None:
    w = build()

    print("GWT-4 state-dependent attention — does the previous focus change the next view?")
    fresh = names(w.project(["deployment"], max_concepts=5))
    w.project(["kubernetes", "rollout"], max_concepts=5)
    w.project(["kubernetes", "helm chart"], max_concepts=5)
    after = names(w.project(["deployment"], max_concepts=5))
    print("  fresh               :", fresh)
    print("  after ops focus     :", after, "(identical)" if fresh == after else "(changed)")

    print("\nGWT-2 limited capacity / ignition — activation profile of one projection")
    p = w.project(["deployment"], max_concepts=15)
    print("  ", [round(v, 3) for v in sorted(p.activation_scores.values(), reverse=True)])

    print("\nHOT-2 metacognitive monitoring — reliability and contested claims")
    p = w.project(["pytorch"], max_concepts=10)
    by_id = {c.id: c.name for c in p.concepts}
    epistemic = getattr(p, "epistemic", None)
    if epistemic is None:
        print("  (no epistemic status on Projection)")
    else:
        print("  reliability:", {by_id[k]: v for k, v in epistemic.reliability.items()})
        for claim in epistemic.contested:
            print(f"  {claim.status}: {[(s, b) for _, s, b in claim.claims]} margin={claim.margin}")
    text = p.render()
    print("  render has epistemic section:", "### Epistemic Status" in text)

    print("\nAST-1 attention schema — does the view record why each concept is in focus?")
    print("  Projection fields:", list(type(p).model_fields))

    print("\nPP-1 prediction error — ingest 'pytorch' without its usual companions")
    res = w.ingest(Observation(concepts=["pytorch", "kubernetes"], source="s"))
    print("  IngestResult fields:", list(type(res).model_fields))


if __name__ == "__main__":
    main()
