#!/usr/bin/env python
"""World 0 — projection-quality evaluation harness.

Runs the REAL activation + projection pipeline over a fixed,
deterministic, LLM-free world and scores each projection against a gold
concept set with offline retrieval metrics (seed-resolution rate, P@k,
R@k, NDCG@k, noise rate, typed-relation ratio, inhibition correctness).

An optional ``--judge <model>`` adds an LLM rubric score (coverage /
relevance / compactness, /9) per the ``agent.projection_judge.system``
prompt; it is credential-aware and skipped automatically when the
backend's API key is missing, so the offline path always runs anywhere.

Examples
--------
    python scripts/eval_projection_matrix.py
    python scripts/eval_projection_matrix.py --perspective debug --md report.md
    python scripts/eval_projection_matrix.py --judge gpt-5.4-nano
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Make `world0` importable when run from a source checkout.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)

from world0 import Observation, World  # noqa: E402
from world0.metrics.retrieval import (  # noqa: E402
    ndcg_at_k,
    precision_recall,
)
from world0.schemas.relation import RelationType  # noqa: E402

NOW = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)

_ML = [
    "model serving", "PyTorch", "training pipeline",
    "neural network", "gradient descent", "optimizer",
]
_OPS = [
    "model serving", "FastAPI", "deployment",
    "monitoring", "latency", "autoscaling",
]


@dataclass
class Case:
    name: str
    seeds: list[str]
    task: str
    relevant: set[str]
    # Concepts that, if present, are noise for this task (other domain).
    distractors: set[str] = field(default_factory=set)
    # Concepts the world should actively suppress (none here by default).
    should_suppress: set[str] = field(default_factory=set)


CASES = [
    Case(
        name="ml_from_serving",
        seeds=["model serving"],
        task="ml training",
        relevant=set(_ML),
        distractors=set(_OPS) - {"model serving"},
    ),
    Case(
        name="ops_from_serving",
        seeds=["model serving"],
        task="ops reliability",
        relevant=set(_OPS),
        distractors=set(_ML) - {"model serving"},
    ),
    Case(
        name="typo_seed",
        seeds=["model servng"],  # typo → fuzzy resolution
        task="ml training",
        relevant=set(_ML),
    ),
]


def build_world(store_path: str) -> World:
    w = World(store_path=store_path)
    observations = [
        Observation(
            concepts=_ML,
            relations=[
                ("model serving", "PyTorch", "depends_on"),
                ("PyTorch", "training pipeline", "supports"),
                ("training pipeline", "neural network", "contains"),
                ("neural network", "gradient descent", "depends_on"),
                ("optimizer", "gradient descent", "supports"),
            ],
            task="ml training",
            source="eval",
        ),
        Observation(
            concepts=_OPS,
            relations=[
                ("model serving", "FastAPI", "depends_on"),
                ("model serving", "deployment", "depends_on"),
                ("deployment", "monitoring", "contains"),
                ("monitoring", "latency", "activates"),
                ("autoscaling", "latency", "supports"),
            ],
            task="ops reliability",
            source="eval",
        ),
    ]
    for _ in range(10):
        for obs in observations:
            w.ingest(obs)
    return w


def evaluate_case(world: World, case: Case, perspective: str | None) -> dict:
    proj = world.project(
        case.seeds,
        task=case.task,
        max_concepts=8,
        perspective=perspective,
        now=NOW,
    )
    names = [c.name for c in proj.concepts]
    name_set = set(names)
    ranked = sorted(
        names,
        key=lambda n: proj.activation_scores.get(
            next(c.id for c in proj.concepts if c.name == n), 0.0
        ),
        reverse=True,
    )

    seed_resolved = sum(
        1
        for v in proj.seed_resolution.values()
        if v != "unresolved"
    )
    seed_rate = (
        seed_resolved / len(proj.seed_resolution)
        if proj.seed_resolution
        else 0.0
    )

    p5, _ = precision_recall(set(ranked[:5]), case.relevant)
    p15, r15 = precision_recall(name_set, case.relevant)
    noise = (
        len(name_set & case.distractors) / len(name_set)
        if name_set and case.distractors
        else 0.0
    )
    typed = [
        r for r in proj.relations
        if r.semantic_relation != "generic_relation"
    ]
    typed_ratio = len(typed) / len(proj.relations) if proj.relations else 1.0
    suppress_ok = not (name_set & case.should_suppress)

    return {
        "case": case.name,
        "seed_rate": seed_rate,
        "p@5": p5,
        "p@8": p15,
        "r@8": r15,
        "ndcg@8": ndcg_at_k(ranked, case.relevant, k=8),
        "noise_rate": noise,
        "typed_ratio": typed_ratio,
        "inhibition_ok": suppress_ok,
        "_projection": proj,
        "_task": case.task,
    }


def _try_judge(model: str):
    """Return a callable(task, rendered) -> /9 score, or None if no creds."""
    try:
        from world0.llm.openai import OpenAIProvider  # noqa: E402
        from world0.prompts import PromptRegistry  # noqa: E402
    except Exception:
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    provider = OpenAIProvider(model=model)
    registry = PromptRegistry()

    def judge(task: str, rendered: str) -> float:
        import json

        system = registry.render("agent.projection_judge.system")
        user = f"## Task\n{task}\n\n## Projection\n{rendered}"
        try:
            raw = provider.complete_json(system, user)
            data = json.loads(raw)
            return (
                float(data.get("coverage", 0))
                + float(data.get("relevance", 0))
                + float(data.get("compactness", 0))
            )
        except Exception:
            return -1.0

    return judge


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perspective", default="", help="named profile to apply")
    ap.add_argument("--judge", default="", help="LLM model for rubric scoring")
    ap.add_argument("--md", default="", help="write a Markdown report here")
    args = ap.parse_args()

    perspective = args.perspective or None
    judge = _try_judge(args.judge) if args.judge else None
    if args.judge and judge is None:
        print(f"(judge {args.judge!r} unavailable — running offline only)\n")

    with tempfile.TemporaryDirectory() as tmp:
        world = build_world(os.path.join(tmp, ".world0"))
        rows = [evaluate_case(world, c, perspective) for c in CASES]

    if judge:
        for row in rows:
            row["judge/9"] = judge(
                row["_task"], row["_projection"].render()
            )

    _print(rows, perspective, bool(judge))
    if args.md:
        _write_md(args.md, rows, perspective, bool(judge))
        print(f"\nWrote {args.md}")


_METRIC_KEYS = [
    "seed_rate", "p@5", "p@8", "r@8", "ndcg@8",
    "noise_rate", "typed_ratio",
]


def _print(rows, perspective, judged) -> None:
    print("=== Projection-quality eval ===")
    print(f"perspective: {perspective or '(none)'}\n")
    head = f"{'case':<18}" + "".join(f"{k:>11}" for k in _METRIC_KEYS)
    head += f"{'inhib':>7}"
    if judged:
        head += f"{'judge/9':>9}"
    print(head)
    print("-" * len(head))
    for r in rows:
        line = f"{r['case']:<18}" + "".join(
            f"{r[k]:>11.3f}" for k in _METRIC_KEYS
        )
        line += f"{'ok' if r['inhibition_ok'] else 'BAD':>7}"
        if judged:
            line += f"{r.get('judge/9', -1):>9.1f}"
        print(line)


_METRIC_LEGEND = (
    "- **seed_rate** — fraction of seed labels that resolved (incl. fuzzy)\n"
    "- **p@5 / p@8** — precision of the top-5 / full selection vs the gold set\n"
    "- **r@8** — recall of the gold set within the projection\n"
    "- **ndcg@8** — ranking quality (relevant concepts ranked high)\n"
    "- **noise_rate** — fraction of the projection that is distractor noise\n"
    "- **typed_ratio** — share of relations that are typed (not generic_relation)\n"
    "- **inhibition** — `ok` when no should-be-suppressed concept leaked in"
)


def _write_md(path, rows, perspective, judged) -> None:
    cols = ["case", *_METRIC_KEYS, "inhibition"]
    if judged:
        cols.append("judge/9")
    lines = [
        "# Projection-quality eval baseline",
        "",
        "Offline retrieval metrics for World 0 projections over a fixed, "
        "deterministic, LLM-free world (see `scripts/eval_projection_matrix.py`). "
        "Regenerate with:",
        "",
        "```bash",
        "python scripts/eval_projection_matrix.py --md docs/projection-eval-baseline.md",
        "```",
        "",
        f"Perspective: `{perspective or '(none)'}`",
        "",
        _METRIC_LEGEND,
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for r in rows:
        cells = [r["case"]]
        cells += [f"{r[k]:.3f}" for k in _METRIC_KEYS]
        cells.append("ok" if r["inhibition_ok"] else "BAD")
        if judged:
            cells.append(f"{r.get('judge/9', -1):.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
