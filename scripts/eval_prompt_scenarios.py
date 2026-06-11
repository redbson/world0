#!/usr/bin/env python
"""Extraction-prompt scenario evaluation — the headroom rubric.

The original matrix rubric (eval_extraction_matrix.py) is saturated:
gpt-5.4-nano scores 9.0/9 on the production prompt, so it cannot
measure further prompt improvements.  This harness probes the
dimensions that rubric does not:

- negative-axis capture     constraints/exclusions stated in prose must
                            become negative relations (counter-signals
                            and inhibition depend on them)
- direction accuracy        across a multi-edge dependency chain, not a
                            single case
- relation recall           gold (source, target, axis) triples found
- relation precision        forbidden pairs that must NOT be linked
                            (over-extraction probe)
- narrative focus           concept quality in conversational prose
- selectivity               dense text must not explode concept count

Each scenario carries gold annotations; metrics have real headroom.

Usage:
    python scripts/eval_prompt_scenarios.py [--model gpt-5.4-nano]
        [--prompts baseline,neg,dir,combined] [--runs 2] [--md out.md]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from eval_extraction_matrix import (  # noqa: E402
    GENERIC_NOISE,
    ModelSpec,
    _parser,
)

from world0 import World  # noqa: E402
from world0.extraction.extractor import ConceptExtractor  # noqa: E402
from world0.prompts import PromptRegistry  # noqa: E402
from world0.schemas.relation import RelationType  # noqa: E402

# ── Scenario corpus with gold annotations ────────────────────────────


@dataclass
class Scenario:
    id: str
    text: str
    task: str
    # Each gold concept is a tuple of acceptable surface alternatives.
    gold_concepts: list[tuple[str, ...]] = field(default_factory=list)
    # (source_alts, target_alts, axis, directed) — axis in
    # {"positive","negative","parallel"}; directed=True means the gold
    # direction is source→target and a flipped edge counts as wrong.
    gold_relations: list[tuple[tuple[str, ...], tuple[str, ...], str, bool]] = (
        field(default_factory=list)
    )
    # Pairs that must NOT be linked by any edge (over-extraction probe).
    forbidden_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = field(
        default_factory=list
    )
    max_concepts: int | None = None


SCENARIOS: list[Scenario] = [
    Scenario(
        "constraint_capture",
        (
            "Synchronous replication guarantees durability, but it violates "
            "our sub-millisecond write-latency budget. The strict-consistency "
            "mode and the low-latency mode are mutually exclusive: an "
            "operator can enable one or the other, never both."
        ),
        "record the replication design constraints",
        gold_concepts=[
            ("synchronous replication",),
            ("durability",),
            ("latency budget", "write-latency budget", "write latency"),
            ("strict-consistency mode", "strict consistency"),
            ("low-latency mode", "low latency mode"),
        ],
        gold_relations=[
            (("synchronous replication",), ("durability",), "positive", True),
            (
                ("synchronous replication",),
                ("latency budget", "write-latency", "write latency"),
                "negative",
                True,
            ),
            (
                ("strict-consistency", "strict consistency"),
                ("low-latency mode", "low latency"),
                "negative",
                False,
            ),
        ],
    ),
    Scenario(
        "zh_constraint",
        (
            "开启全量日志会拖慢写入路径，违反了我们的性能预算。"
            "审计模式与高吞吐模式互斥，集群同一时刻只能启用其中一个。"
            "增量快照依赖预写日志来保证一致性。"
        ),
        "记录日志系统的设计约束",
        gold_concepts=[
            ("全量日志",),
            ("性能预算",),
            ("审计模式",),
            ("高吞吐模式", "高吞吐"),
            ("增量快照",),
            ("预写日志",),
        ],
        gold_relations=[
            (("全量日志",), ("性能预算",), "negative", True),
            (("审计模式",), ("高吞吐模式", "高吞吐"), "negative", False),
            (("增量快照",), ("预写日志",), "positive", True),
        ],
    ),
    Scenario(
        "direction_chain",
        (
            "The checkout service depends on the payment gateway, and the "
            "payment gateway depends on the fraud-scoring model. Reliable "
            "fraud scoring enables instant refunds for trusted customers."
        ),
        "map the payment dependency chain",
        gold_concepts=[
            ("checkout service",),
            ("payment gateway",),
            ("fraud-scoring model", "fraud scoring"),
            ("instant refunds", "instant refund"),
        ],
        gold_relations=[
            (("checkout service",), ("payment gateway",), "positive", True),
            (
                ("payment gateway",),
                ("fraud-scoring", "fraud scoring"),
                "positive",
                True,
            ),
            (
                ("fraud scoring", "fraud-scoring"),
                ("instant refund",),
                "positive",
                True,
            ),
        ],
    ),
    Scenario(
        "meeting_notes",
        (
            "Notes from Tuesday sync: Dana walked us through the incident. "
            "Long story short, the retry storm overwhelmed the rate limiter, "
            "so we agreed to add exponential backoff to the client SDK. "
            "Priya will own the rollout. We also decided feature flags will "
            "gate the new backoff behavior so we can disable it quickly. "
            "Lunch orders were late again, unrelated."
        ),
        "capture the incident follow-ups",
        gold_concepts=[
            ("retry storm",),
            ("rate limiter",),
            ("exponential backoff",),
            ("feature flag", "feature flags"),
        ],
        gold_relations=[
            (("retry storm",), ("rate limiter",), "negative", True),
            (
                ("feature flag", "feature flags"),
                ("exponential backoff", "backoff"),
                "positive",
                False,
            ),
        ],
        forbidden_pairs=[
            (("lunch", "lunch orders"), ("rate limiter",)),
            (("lunch", "lunch orders"), ("retry storm",)),
        ],
    ),
    Scenario(
        "implicit_overreach",
        (
            "The data team uses Spark for batch jobs. The mobile team ships "
            "a React Native app. Both teams attend the weekly platform "
            "review."
        ),
        "note the team tooling",
        gold_concepts=[
            ("spark",),
            ("react native", "react native app"),
            ("platform review", "weekly platform review"),
        ],
        gold_relations=[],
        # Spark and React Native merely co-occur in prose; a semantic
        # edge between them is over-extraction.
        forbidden_pairs=[(("spark",), ("react native",))],
    ),
    Scenario(
        "dense_text",
        (
            "The ingestion pipeline tails the change-data-capture stream "
            "from PostgreSQL, normalizes records, and writes them to Kafka. "
            "A Flink job consumes Kafka, joins the click stream, and "
            "materializes features into the online feature store. The "
            "training pipeline reads the offline store nightly, retrains "
            "the ranking model, and publishes it to the model registry, "
            "from which the serving layer pulls new versions. Monitoring "
            "covers lag, schema drift, and feature skew."
        ),
        "document the ML data platform",
        gold_concepts=[
            ("ingestion pipeline",),
            ("kafka",),
            ("feature store",),
            ("ranking model",),
            ("model registry",),
        ],
        gold_relations=[
            (
                ("training pipeline", "training"),
                ("ranking model",),
                "positive",
                False,
            ),
        ],
        max_concepts=15,
    ),
]


# ── Prompt variants ──────────────────────────────────────────────────

BASELINE = PromptRegistry().render("extraction.concepts_relations.system")

NEG_SECTION = """\

## Constraint and opposition capture (critical)
Negative-axis relations are first-class output, not an afterthought. When
the text states that something violates a budget, limit, or constraint;
that two modes/options are mutually exclusive ("never both",
"one or the other", "互斥", "只能启用其中一个"); that an assumption was
wrong; or that one thing degrades/overwhelms another — you MUST emit the
matching negative relation (violates_constraint, exclusion, disjointness,
conflict, instability) between the specific concepts involved. A constraint
stated in prose but not extracted is a lost cognitive boundary.
Example: "Synchronous mode guarantees durability but violates the latency
budget" → {"source": "synchronous mode", "target": "latency budget",
"type": "violates_constraint"} (and the positive durability relation too).
"""

DIR_SECTION = """\

## Direction discipline
Before emitting each relation, verify it reads correctly as
"<source> <label> <target>":
- dependence: source depends on target ("A depends on B" → source=A, target=B)
- enables: source enables target ("caching enables low latency" → source=caching)
- membership / inclusion: source belongs to / is contained in target
- violates_constraint: source violates target
- functional_map: source maps to target
Mention order in the sentence is NOT direction. If your relation reads
backwards under the label's definition, swap source and target before output.
"""

CAP_SECTION = """\

## Selectivity under density
Dense text does not license more concepts. Hard limit: never output more
than 15 concepts. When the text mentions more candidate units, keep only
the most structurally important ones — relation hubs and task-critical
units — and fold the rest into descriptions or aliases of kept concepts.
Count your concepts before responding; if over 15, remove the least
connected ones.
"""

_ANCHOR = "## Output format"


def _inject(base: str, *sections: str) -> str:
    return base.replace(_ANCHOR, "".join(sections) + "\n" + _ANCHOR)


PROMPTS: dict[str, str] = {
    "baseline": BASELINE,
    "neg": _inject(BASELINE, NEG_SECTION),
    "dir": _inject(BASELINE, DIR_SECTION),
    "combined": _inject(BASELINE, NEG_SECTION, DIR_SECTION),
    "refined": _inject(BASELINE, NEG_SECTION, DIR_SECTION, CAP_SECTION),
}


# ── Scoring ──────────────────────────────────────────────────────────


def _find(w: World, alts: tuple[str, ...]):
    for c in w.concepts.all():
        hay = " ".join([c.name, *c.aliases]).lower()
        if any(a.lower() in hay for a in alts):
            return c
    return None


def _edges_between(w: World, a, b):
    if a is None or b is None:
        return []
    return w.relations.find_any_between(a.id, b.id)


def score_scenario(w: World, sc: Scenario) -> dict:
    out = {
        "gold_c": len(sc.gold_concepts), "hit_c": 0,
        "gold_r": len(sc.gold_relations), "hit_r": 0,
        "dir_total": 0, "dir_ok": 0,
        "neg_total": 0, "neg_hit": 0,
        "forbidden": len(sc.forbidden_pairs), "false_links": 0,
        "noise": 0, "over_cap": 0,
    }
    for alts in sc.gold_concepts:
        if _find(w, alts):
            out["hit_c"] += 1
    axis_map = {
        "positive": RelationType.POSITIVE,
        "negative": RelationType.NEGATIVE,
        "parallel": RelationType.PARALLEL,
    }
    for src_alts, tgt_alts, axis, directed in sc.gold_relations:
        is_neg = axis == "negative"
        if is_neg:
            out["neg_total"] += 1
        src = _find(w, src_alts)
        tgt = _find(w, tgt_alts)
        edges = [
            e
            for e in _edges_between(w, src, tgt)
            if e.relation_type == axis_map[axis]
        ]
        if not edges:
            continue
        out["hit_r"] += 1
        if is_neg:
            out["neg_hit"] += 1
        if directed:
            out["dir_total"] += 1
            if any(
                e.source_id == src.id and e.target_id == tgt.id
                for e in edges
            ):
                out["dir_ok"] += 1
    for a_alts, b_alts in sc.forbidden_pairs:
        if _edges_between(w, _find(w, a_alts), _find(w, b_alts)):
            out["false_links"] += 1
    out["noise"] = sum(
        1
        for c in w.concepts.all()
        if c.name.strip().lower() in GENERIC_NOISE
    )
    if sc.max_concepts is not None and len(w.concepts) > sc.max_concepts:
        out["over_cap"] = 1
    return out


def run_prompt(spec: ModelSpec, prompt: str, runs: int) -> dict:
    totals: list[dict] = []
    for _ in range(runs):
        acc = {
            "gold_c": 0, "hit_c": 0, "gold_r": 0, "hit_r": 0,
            "dir_total": 0, "dir_ok": 0, "neg_total": 0, "neg_hit": 0,
            "forbidden": 0, "false_links": 0, "noise": 0,
            "over_cap": 0, "fail": 0,
        }
        for sc in SCENARIOS:
            user = ConceptExtractor._build_user_prompt(
                sc.text, task=sc.task, source=sc.id
            )
            try:
                raw = spec.build().complete_json(prompt, user)
                obs = _parser._parse_response(raw, task=sc.task, source=sc.id)
            except Exception:
                acc["fail"] += 1
                continue
            with tempfile.TemporaryDirectory() as d:
                w = World(store_path=d)
                try:
                    w.ingest(obs)
                except Exception:
                    acc["fail"] += 1
                    continue
                s = score_scenario(w, sc)
                for k, v in s.items():
                    acc[k] += v
        totals.append(acc)

    def rate(hit: str, total: str) -> float:
        h = sum(t[hit] for t in totals)
        n = sum(t[total] for t in totals)
        return h / n if n else 1.0

    metrics = {
        "concept_recall": rate("hit_c", "gold_c"),
        "relation_recall": rate("hit_r", "gold_r"),
        "neg_recall": rate("neg_hit", "neg_total"),
        "direction_acc": rate("dir_ok", "dir_total"),
        "false_link_rate": rate("false_links", "forbidden"),
        "noise": sum(t["noise"] for t in totals),
        "over_cap": sum(t["over_cap"] for t in totals),
        "fail": sum(t["fail"] for t in totals),
    }
    metrics["composite"] = (
        metrics["concept_recall"]
        + metrics["relation_recall"]
        + metrics["neg_recall"]
        + metrics["direction_acc"]
        + (1.0 - metrics["false_link_rate"])
    ) / 5.0
    return metrics


COLS = [
    "composite", "concept_recall", "relation_recall", "neg_recall",
    "direction_acc", "false_link_rate", "noise", "over_cap", "fail",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-nano")
    ap.add_argument("--prompts", default="baseline,neg,dir,combined")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--md", default="")
    args = ap.parse_args()

    spec = ModelSpec(args.model, "azure")
    if not spec.available():
        print("Azure credentials missing; aborting.")
        return 1

    names = [p.strip() for p in args.prompts.split(",") if p.strip()]
    lines = [
        f"# Extraction-prompt scenario eval — {args.model}, "
        f"{len(SCENARIOS)} scenarios × {args.runs} runs",
        "",
        "| prompt | " + " | ".join(COLS) + " |",
        "|" + "---|" * (len(COLS) + 1),
    ]
    for name in names:
        m = run_prompt(spec, PROMPTS[name], args.runs)
        row = f"| {name} | " + " | ".join(
            f"{m[c]:.2f}" if isinstance(m[c], float) else str(m[c])
            for c in COLS
        ) + " |"
        lines.append(row)
        print(row, flush=True)

    report = "\n".join(lines)
    print("\n" + report)
    if args.md:
        Path(args.md).write_text(report + "\n")
        print(f"\nwrote {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
