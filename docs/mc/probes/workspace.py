"""Round 17 probe: sustained focus (GWT-4), ignition (GWT-2), attention schema (AST-1).

Run from the repository root:  python docs/mc/probes/workspace.py
"""

from __future__ import annotations

import tempfile

from world0 import Observation, World

OPS = ["deployment", "kubernetes", "helm chart", "container registry", "rollout", "load balancer"]
ML = ["deployment", "pytorch", "gpu cluster", "training run", "checkpoint", "loss curve"]
OTHER = ["sourdough", "starter", "hydration", "proofing", "crumb"]


def build(sustained: bool) -> World:
    w = World(store_path=tempfile.mkdtemp() + "/w", sustained_attention=sustained)
    for _ in range(12):
        w.ingest(Observation(concepts=OPS, relations=[("deployment", "kubernetes", "depends_on")], source="s"))
        w.ingest(Observation(concepts=ML, relations=[("deployment", "pytorch", "depends_on")], source="s"))
        w.ingest(Observation(concepts=OTHER, source="s"))
    w.reflect()
    return w


def split(names: list[str]) -> str:
    rest = names[1:]
    return f"ops={sum(n in OPS for n in rest)} ml={sum(n in ML for n in rest)}"


def view(w: World, seeds, task: str = "", k: int = 5) -> list[str]:
    return [c.name for c in w.project(seeds, task=task, max_concepts=k).concepts]


def main() -> None:
    print("1. state-dependent attention: bridge concept after an Ops line of attention")
    for sustained in (False, True):
        w = build(sustained)
        base = view(w, ["deployment"])
        view(w, ["kubernetes", "rollout"])
        view(w, ["helm chart", "load balancer"])
        after = view(w, ["deployment"])
        print(f"   sustained={sustained!s:5}  cold {split(base)}  after ops focus {split(after)}  {after}")

    print("\n2. same, after an ML line of attention")
    w = build(True)
    view(w, ["pytorch", "checkpoint"])
    view(w, ["training run", "loss curve"])
    print("  ", split(view(w, ["deployment"])), "| focus size", len(w.focus))

    print("\n3. no perseveration: unrelated seeds after the Ops focus")
    w = build(True)
    view(w, ["kubernetes", "rollout"])
    cold = view(build(False), ["sourdough"])
    print("   with focus:", view(w, ["sourdough"]), "| cold:", cold)

    print("\n4. retention: Ops focus, then k unrelated views, then deployment")
    for k in range(0, 5):
        w = build(True)
        view(w, ["kubernetes", "rollout"])
        view(w, ["helm chart", "load balancer"])
        for _ in range(k):
            view(w, ["sourdough"])
        print(f"   k={k}: {split(view(w, ['deployment']))}  focus={sorted(w.focus.items())[:3]}…")

    print("\n5. release on task switch")
    w = build(True)
    view(w, ["kubernetes", "rollout"], task="cluster ops")
    print("   focus before:", len(w.focus), "task", w.focus.task)
    print("   deployment under 'model training':", split(view(w, ["deployment"], task="model training")))
    print("   stateless same task              :", split(view(build(False), ["deployment"], task="model training")))

    print("\n6. attention schema of one view")
    w = build(True)
    view(w, ["kubernetes", "rollout"])
    p = w.project(["deployment"], task="helm upgrade", max_concepts=5)
    names = {c.id: c.name for c in p.concepts}
    for cid, t in p.attention.items():
        print(f"   {names[cid]:18} kind={t.kind:7} via={names.get(t.via, t.via)!s:12} rel={t.relation:16}"
              f" named={t.task_named} hist={t.task_history} sustained={t.sustained} in_focus={t.in_focus} ignited={t.ignited}")
    text = p.render()
    print(text[text.index("### Why These Concepts"):] if "### Why These Concepts" in text else "(no attention section)")


if __name__ == "__main__":
    main()
