"""Agent-context ablation harness — does the projection earn its keep?

The decisive question for World 0 is not whether its internal metrics
look good, but whether a task-conditioned projection beats the obvious
alternatives an agent builder would reach for instead.  This harness
builds the same context six ways and scores each on what reaches the
agent:

  none                 no external context (floor)
  raw_history          every observation, concatenated
  lexical_recall       top-k signature-similar concepts to the task
  graph_neighbors      seeds + unranked 1-hop neighborhood
  projection           World.project(...).render()
  projection_counter   projection with counter-signals exposed

Offline metrics (no LLM required):

  tokens      ≈ len(context) / 4 — what the variant costs per prompt
  coverage    expected-concept recall — did the right concepts arrive
  violations  forbidden concepts present WITHOUT a ⚠ warning — a
              forbidden concept may appear only alongside its
              counter-signal; raw dumps fail this by construction
  efficiency  coverage per 1k estimated tokens

An optional LLM judge (World's provider) can score answer quality on
top; the offline metrics are deterministic and run in CI.

Usage:
    python scripts/eval_agent_ablation.py
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from world0 import Observation, World  # noqa: E402
from world0.schemas.context import Perspective  # noqa: E402

LEXICAL_RECALL_K = 8


# ── Fixture world ────────────────────────────────────────────────────


@dataclass
class AblationTask:
    """One evaluation task with ground-truth expectations."""

    task: str
    seeds: list[str]
    # Concepts the context must surface for the agent to succeed.
    expected: list[str]
    # Concepts excluded by constraints — they may appear in a context
    # only alongside an explicit ⚠ warning, never silently.
    forbidden: list[str] = field(default_factory=list)


def build_fixture_world(store_path: Path) -> tuple[World, list[str]]:
    """A small distributed-storage design world with real constraints.

    Returns the world plus the raw observation log (for raw_history).
    """
    w = World(store_path=store_path)
    observations = [
        Observation(
            concepts=[
                "write-ahead log", "crash recovery", "durability",
            ],
            relations=[
                ("write-ahead log", "crash recovery", "enables"),
                ("write-ahead log", "durability", "enables"),
            ],
            descriptions={
                "write-ahead log": "append-only log written before applying mutations",
                "crash recovery": "restoring consistent state after failure",
            },
            task="design storage engine", source="design notes",
        ),
        Observation(
            concepts=[
                "sync replication", "durability", "low write latency",
                "async replication",
            ],
            relations=[
                ("sync replication", "durability", "enables"),
                ("sync replication", "low write latency", "violates_constraint"),
                ("sync replication", "async replication", "conflict"),
            ],
            descriptions={
                "sync replication": "acknowledge writes only after replicas confirm",
                "async replication": "acknowledge locally, replicate in background",
            },
            task="design replication", source="design notes",
        ),
        Observation(
            concepts=["quorum consensus", "sync replication", "availability"],
            relations=[
                ("quorum consensus", "sync replication", "dependence"),
            ],
            task="design replication", source="design notes",
        ),
        Observation(
            concepts=["compaction", "write amplification", "lsm tree"],
            relations=[
                ("lsm tree", "compaction", "dependence"),
                ("compaction", "write amplification", "mutual_reinforcement"),
            ],
            task="design storage engine", source="design notes",
        ),
    ]
    log: list[str] = []
    for obs in observations:
        w.ingest(obs)
        log.append(
            f"[{obs.task}] concepts: {', '.join(obs.concepts)}; "
            + "; ".join(f"{s} {r} {t}" for s, t, r in obs.relations)
        )
    # Reinforce the replication cluster so maturity/affinity differ.
    for _ in range(3):
        w.ingest(Observation(
            concepts=["sync replication", "durability", "quorum consensus"],
            task="design replication", source="design notes",
        ))
    return w, log


FIXTURE_TASKS = [
    AblationTask(
        task="design replication for strong durability",
        seeds=["sync replication"],
        expected=["sync replication", "durability", "quorum consensus"],
        forbidden=["low write latency"],
    ),
    AblationTask(
        task="design storage engine write path",
        seeds=["write-ahead log"],
        expected=["write-ahead log", "crash recovery", "durability"],
    ),
]


# ── Context variant builders ─────────────────────────────────────────


def ctx_none(world: World, log: list[str], task: AblationTask) -> str:
    return ""


def ctx_raw_history(world: World, log: list[str], task: AblationTask) -> str:
    return "\n".join(log)


def ctx_lexical_recall(world: World, log: list[str], task: AblationTask) -> str:
    hits = world.concepts.find_similar(
        task.task, min_similarity=0.05, limit=LEXICAL_RECALL_K
    )
    lines = [
        f"- {node.name}: {node.description}".rstrip(": ")
        for node, _ in hits
    ]
    return "\n".join(lines)


def ctx_graph_neighbors(world: World, log: list[str], task: AblationTask) -> str:
    """Seeds + unranked 1-hop neighborhood — the HippoRAG-refuted baseline."""
    names: list[str] = []
    for seed in task.seeds:
        node = world.concepts.resolve(seed)
        if node is None:
            continue
        names.append(node.name)
        for neighbor_id in world.relations.neighbors(node.id):
            neighbor = world.concepts.get(neighbor_id)
            if neighbor and neighbor.name not in names:
                names.append(neighbor.name)
    return "\n".join(f"- {n}" for n in names)


def ctx_projection(world: World, log: list[str], task: AblationTask) -> str:
    return world.project(task.seeds, task=task.task).render()


def ctx_projection_counter(
    world: World, log: list[str], task: AblationTask
) -> str:
    lens = Perspective(
        name="ablation",
        task=task.task,
        negative_visibility={"conflict": "expose"},
    )
    return world.project(task.seeds, perspective=lens).render()


VARIANTS = {
    "none": ctx_none,
    "raw_history": ctx_raw_history,
    "lexical_recall": ctx_lexical_recall,
    "graph_neighbors": ctx_graph_neighbors,
    "projection": ctx_projection,
    "projection_counter": ctx_projection_counter,
}


# ── Metrics ──────────────────────────────────────────────────────────


def score_variant(context: str, task: AblationTask) -> dict:
    text = context.lower()
    expected_hits = sum(1 for c in task.expected if c.lower() in text)
    coverage = expected_hits / len(task.expected) if task.expected else 1.0

    # A forbidden concept may appear only on a ⚠-marked warning line.
    violations = 0
    for concept in task.forbidden:
        needle = concept.lower()
        offending = [
            line
            for line in text.splitlines()
            if needle in line and "⚠" not in line
        ]
        if offending:
            violations += 1
    violation_rate = (
        violations / len(task.forbidden) if task.forbidden else 0.0
    )

    tokens = max(1, len(context) // 4)
    return {
        "tokens": tokens,
        "coverage": coverage,
        "violation_rate": violation_rate,
        "efficiency": coverage / tokens * 1000,
    }


def run_ablation(world: World, log: list[str], tasks: list[AblationTask]) -> dict:
    """Score every variant over every task; returns variant → averaged metrics."""
    results: dict[str, dict] = {}
    for name, builder in VARIANTS.items():
        per_task = [
            score_variant(builder(world, log, task), task) for task in tasks
        ]
        results[name] = {
            key: sum(m[key] for m in per_task) / len(per_task)
            for key in per_task[0]
        }
    return results


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        world, log = build_fixture_world(Path(tmp) / ".world0")
        results = run_ablation(world, log, FIXTURE_TASKS)

    print("# Agent-context ablation\n")
    print("| variant | tokens | coverage | violation_rate | coverage/1k tok |")
    print("|---|---|---|---|---|")
    for name, m in results.items():
        print(
            f"| {name} | {m['tokens']:.0f} | {m['coverage']:.2f} "
            f"| {m['violation_rate']:.2f} | {m['efficiency']:.2f} |"
        )
    print(
        "\nReading: projection variants should match or beat raw_history "
        "coverage at a fraction of the tokens, and projection_counter "
        "must hold violation_rate at 0 (forbidden concepts appear only "
        "behind ⚠ warnings)."
    )


if __name__ == "__main__":
    main()
