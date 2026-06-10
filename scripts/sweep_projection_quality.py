#!/usr/bin/env python
"""World 0 — projection-quality sensitivity sweep.

Runs the REAL activation + projection pipeline over a fixed,
deterministic, LLM-free world for every cell of a coefficient grid, and
reports structural retrieval metrics (P@k, NDCG@k, cross-domain Jaccard
distance) plus the delta against the default-config baseline.

This is a *sensitivity report*, not an auto-tuner: it tells you how the
output moves when you nudge a constant, so tuning decisions stay human
and explainable.  No network, no API keys, fully reproducible (time is
pinned via ``now``).

Examples
--------
    python scripts/sweep_projection_quality.py --quick
    python scripts/sweep_projection_quality.py \
        --param activation.propagation_min_ratio=0.01,0.03,0.06 \
        --param projection.mmr_lambda=0.1,0.3,0.5 --md report.md
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timezone

# Make `world0` importable when run from a source checkout.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)

from world0 import Observation, World  # noqa: E402
from world0.dynamics.coefficients import (  # noqa: E402
    ActivationConfig,
    ProjectionConfig,
)
from world0.metrics.retrieval import (  # noqa: E402
    ndcg_at_k,
    precision_recall,
    selection_diversity,
)

# Pinned clock so temporal relevance is identical across runs.
NOW = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)


# ── Fixed evaluation corpus (two well-separated domains + a bridge) ───

# Projection budget deliberately below the concept count so MMR
# selection genuinely has to choose — that is what makes mmr_lambda and
# propagation_min_ratio observable.
MAX_CONCEPTS = 6

# A deep relevant chain from the hub — reaching its tail depends on
# propagation_min_ratio (the floor that keeps distant signal alive).
_SERVE_CHAIN = [
    "model serving", "inference engine", "request batching",
    "kv cache", "quantization", "latency reduction",
]

# A redundant near-hub cluster: every member's only neighbor is the hub,
# so their neighbor sets are identical → maximally redundant.  Pure
# relevance (low mmr_lambda) grabs several; diversity (high mmr_lambda)
# keeps one and frees slots for the chain.
_HUB_SATELLITES = [
    "serving config", "serving flags", "serving env", "serving manifest",
]

# A distractor domain sharing the hub (ops) and a pure-noise domain
# (frontend) unconnected to serving — both should stay out of a
# serving-optimization projection.
_OPS = ["deployment", "monitoring", "alerting", "dashboards", "autoscaling"]
_FRONTEND = ["React", "component model", "state management", "CSS styling"]


def build_world(store_path: str) -> World:
    """Deterministic multi-domain world built from recorded observations.

    Structure: hub + deep relevant chain + redundant satellite cluster +
    an ops distractor domain + an unconnected frontend noise domain.
    """
    w = World(store_path=store_path)
    chain_relations = [
        (a, b, "depends_on")
        for a, b in zip(_SERVE_CHAIN, _SERVE_CHAIN[1:])
    ]
    observations = [
        Observation(
            concepts=_SERVE_CHAIN,
            relations=chain_relations,
            task="serving optimization",
            source="sweep",
        ),
        Observation(
            concepts=["model serving", *_HUB_SATELLITES],
            relations=[
                ("model serving", s, "related_to") for s in _HUB_SATELLITES
            ],
            task="serving config",
            source="sweep",
        ),
        Observation(
            concepts=["model serving", *_OPS],
            relations=[
                ("model serving", "deployment", "depends_on"),
                ("deployment", "monitoring", "contains"),
                ("monitoring", "alerting", "enables"),
                ("monitoring", "dashboards", "enables"),
                ("autoscaling", "deployment", "supports"),
            ],
            task="ops reliability",
            source="sweep",
        ),
        Observation(
            concepts=_FRONTEND,
            relations=[
                ("React", "component model", "supports"),
                ("component model", "state management", "enables"),
                ("React", "CSS styling", "related_to"),
            ],
            task="frontend",
            source="sweep",
        ),
    ]
    for _ in range(10):
        for obs in observations:
            w.ingest(obs)
    return w


# Each case: seed(s), the curated gold set, the task label, and the
# distractor set whose presence is noise.
CASES = [
    {
        "name": "serving_chain",
        "seeds": ["model serving"],
        "task": "serving optimization",
        "relevant": set(_SERVE_CHAIN),
        "distractors": set(_HUB_SATELLITES) | set(_OPS) | set(_FRONTEND),
    },
    {
        "name": "ops_from_hub",
        "seeds": ["model serving"],
        "task": "ops reliability",
        "relevant": {"model serving", *_OPS},
        "distractors": set(_SERVE_CHAIN[1:]) | set(_HUB_SATELLITES) | set(_FRONTEND),
    },
]


# ── Metric computation for one config ─────────────────────────────────

def evaluate(
    act_cfg: ActivationConfig, proj_cfg: ProjectionConfig
) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        world = build_world(os.path.join(tmp, ".world0"))
        # Rebuild engines with the candidate configs (same data on disk).
        scored = World(
            store_path=os.path.join(tmp, ".world0"),
            activation_config=act_cfg,
            projection_config=proj_cfg,
        )

        per_case_p, per_case_ndcg = [], []
        domain_name_sets = []
        for case in CASES:
            proj = scored.project(
                case["seeds"], task=case["task"],
                max_concepts=MAX_CONCEPTS, max_depth=4, now=NOW,
            )
            names = [c.name for c in proj.concepts]
            ranked = sorted(
                names,
                key=lambda n: proj.activation_scores.get(
                    next(c.id for c in proj.concepts if c.name == n), 0.0
                ),
                reverse=True,
            )
            precision, _ = precision_recall(set(names), case["relevant"])
            per_case_p.append(precision)
            per_case_ndcg.append(
                ndcg_at_k(ranked, case["relevant"], k=MAX_CONCEPTS)
            )
            domain_name_sets.append(set(names))

        # Cross-domain separation: the two projections should be distinct.
        cross_jaccard_distance = 1.0
        if len(domain_name_sets) == 2:
            a, b = domain_name_sets
            union = a | b
            cross_jaccard_distance = (
                1.0 - len(a & b) / len(union) if union else 1.0
            )

        return {
            "precision": sum(per_case_p) / len(per_case_p),
            "ndcg": sum(per_case_ndcg) / len(per_case_ndcg),
            "cross_domain_dist": cross_jaccard_distance,
        }


# ── Grid parsing ──────────────────────────────────────────────────────

def parse_params(specs: list[str]) -> dict[str, list[float]]:
    """``activation.x=1,2,3`` → {'activation.x': [1.0, 2.0, 3.0]}."""
    grid: dict[str, list[float]] = {}
    for spec in specs:
        key, _, raw = spec.partition("=")
        grid[key.strip()] = [float(v) for v in raw.split(",") if v.strip()]
    return grid


def make_configs(
    overrides: dict[str, float]
) -> tuple[ActivationConfig, ProjectionConfig]:
    act = ActivationConfig()
    proj = ProjectionConfig()
    act_kw, proj_kw = {}, {}
    for key, value in overrides.items():
        section, _, field_name = key.partition(".")
        if section == "activation":
            act_kw[field_name] = value
        elif section == "projection":
            proj_kw[field_name] = value
        else:
            raise SystemExit(f"unknown param section: {section!r}")
    if act_kw:
        act = replace(act, **act_kw)
    if proj_kw:
        proj = replace(proj, **proj_kw)
    return act, proj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--param",
        action="append",
        default=[],
        help="section.field=v1,v2,...  (e.g. projection.mmr_lambda=0.1,0.3)",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="run a small built-in 2x2 grid",
    )
    ap.add_argument("--md", default="", help="write a Markdown report here")
    args = ap.parse_args()

    if args.quick and not args.param:
        args.param = [
            "projection.mmr_lambda=0.1,0.3,0.5",
            "activation.propagation_min_ratio=0.01,0.03,0.06",
        ]

    grid = parse_params(args.param)
    baseline = evaluate(ActivationConfig(), ProjectionConfig())

    keys = list(grid)
    combos = [
        dict(zip(keys, values))
        for values in itertools.product(*(grid[k] for k in keys))
    ] if keys else [{}]

    rows = []
    for overrides in combos:
        act, proj = make_configs(overrides)
        metrics = evaluate(act, proj)
        rows.append((overrides, metrics))

    _print_report(baseline, rows, keys)
    if args.md:
        _write_md(args.md, baseline, rows, keys)
        print(f"\nWrote {args.md}")


def _fmt_overrides(overrides: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:g}" for k, v in overrides.items()) or "(baseline)"


def _print_report(baseline, rows, keys) -> None:
    k = MAX_CONCEPTS
    print("\n=== Projection-quality sweep ===")
    print(
        f"baseline: P@{k}={baseline['precision']:.3f} "
        f"NDCG@{k}={baseline['ndcg']:.3f} "
        f"crossΔ={baseline['cross_domain_dist']:.3f}\n"
    )
    header = (
        f"{'config':<52} {'P@'+str(k):>7} {'NDCG@'+str(k):>8} {'crossΔ':>8}"
    )
    print(header)
    print("-" * len(header))
    for overrides, m in rows:
        dp = m["precision"] - baseline["precision"]
        dn = m["ndcg"] - baseline["ndcg"]
        print(
            f"{_fmt_overrides(overrides):<52} "
            f"{m['precision']:>7.3f} {m['ndcg']:>8.3f} "
            f"{m['cross_domain_dist']:>8.3f}   "
            f"(ΔP {dp:+.3f}, ΔNDCG {dn:+.3f})"
        )


def _write_md(path, baseline, rows, keys) -> None:
    k = MAX_CONCEPTS
    swept = ", ".join(f"`{key}`" for key in keys) or "(none)"
    lines = [
        "# Projection-quality coefficient sweep",
        "",
        "Sensitivity of projection quality to activation/projection "
        "coefficients, over a fixed deterministic world with a deep "
        "relevant chain, a redundant near-hub cluster, and distractor "
        "domains (see `scripts/sweep_projection_quality.py`). This is a "
        "*sensitivity report*, not an auto-tuner — it shows how the output "
        "moves so tuning stays a human, explainable decision.",
        "",
        f"Swept: {swept}. Budget: top-{k} concepts. "
        f"Baseline (shipped defaults) — P@{k} {baseline['precision']:.3f}, "
        f"NDCG@{k} {baseline['ndcg']:.3f}, "
        f"cross-domain distance {baseline['cross_domain_dist']:.3f}.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python scripts/sweep_projection_quality.py \\",
        "    --param projection.mmr_lambda=0.0,0.1,0.3,0.5,0.7,0.9 \\",
        "    --param activation.propagation_min_ratio=0.03,0.1,0.3 \\",
        "    --md docs/projection-sweep-baseline.md",
        "```",
        "",
        f"| config | P@{k} | NDCG@{k} | cross-domain Δ | ΔP | ΔNDCG |",
        "|---|---|---|---|---|---|",
    ]
    for overrides, m in rows:
        dp = m["precision"] - baseline["precision"]
        dn = m["ndcg"] - baseline["ndcg"]
        lines.append(
            f"| {_fmt_overrides(overrides)} | {m['precision']:.3f} | "
            f"{m['ndcg']:.3f} | {m['cross_domain_dist']:.3f} | "
            f"{dp:+.3f} | {dn:+.3f} |"
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
