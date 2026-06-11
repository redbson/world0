"""Real-corpus, real-LLM evaluation of the relation optimizations.

Builds a world from prose via ``ingest_text`` with the configured LLM
(Azure OpenAI, deployment from --model), so every LLM semantic path
fires for real: extraction, Hebbian relation typing, similarity
judging, and reflect-time generic refinement.  Then runs the six-way
context ablation with an LLM answer + judge loop on top of the offline
metrics.

Usage:
    python scripts/eval_real_corpus_llm.py [--model gpt-5.4-nano] [--no-judge]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_agent_ablation import AblationTask, VARIANTS, score_variant  # noqa: E402

from world0 import World  # noqa: E402
from world0.llm.azure_openai import AzureOpenAIProvider  # noqa: E402

# ── Corpus: six prose chunks about a knowledge-tooling domain ────────
# Written as natural documentation prose (not concept lists) so the
# extractor does real work.  One chunk states an explicit constraint to
# seed negative-axis edges.

CORPUS = [
    (
        "An agent memory layer records episodic events: what happened, "
        "when, and in which session. Episodic memory enables recall of "
        "past interactions, but it does not by itself organize meaning — "
        "retrieving raw transcripts returns text, not understanding.",
        "memory layer design",
    ),
    (
        "A cognitive concept layer organizes meaning instead of storing "
        "events. Concepts are stable semantic units connected by typed "
        "relations; spreading activation over those relations selects "
        "which concepts matter for the current task. The concept layer "
        "depends on typed relations to route activation correctly.",
        "concept layer design",
    ),
    (
        "Projection turns the activated neighborhood into a compact, "
        "task-relevant view that is injected into the agent prompt. "
        "Projection quality depends on spreading activation, and good "
        "projections enable focused agent reasoning while reducing token "
        "cost.",
        "projection design",
    ),
    (
        "Storing full documents inside concept records violates the "
        "boundary between cognition and content: long-form facts belong "
        "in a source library, with concepts holding only pointers. "
        "Document storage inside the concept layer conflicts with "
        "projection compactness.",
        "boundary constraints",
    ),
    (
        "Relation decay removes stale structure: edges lose weight over "
        "time unless reinforced by reuse, and pruning deletes edges whose "
        "weight falls below threshold. Decay enables the graph to forget, "
        "which keeps projections current.",
        "lifecycle design",
    ),
    (
        "A vector store retrieves passages by embedding similarity. "
        "Vector retrieval is fast and recall-oriented, but similarity "
        "alone cannot express typed structure — pure embedding search "
        "conflicts with explainable concept selection.",
        "retrieval tradeoffs",
    ),
]

TASKS = [
    AblationTask(
        task="explain how the system builds a compact task-relevant context for an agent",
        seeds=["projection"],
        expected=["projection", "spreading activation", "typed relation"],
        forbidden=["document storage"],
    ),
    AblationTask(
        task="how does the graph forget stale structure over time",
        seeds=["relation decay"],
        expected=["decay", "pruning", "reinforce"],
    ),
]

ANSWER_SYSTEM = (
    "You are an assistant answering a design question. Use ONLY the "
    "provided context; if the context is insufficient, say so briefly. "
    'Return JSON only: {"answer": "<3-5 sentence answer>"}'
)

JUDGE_SYSTEM = (
    "You are a strict grader. Given a question, a reference fact list, "
    "and an answer, return JSON only: "
    '{"correctness": 1-5, "groundedness": 1-5}. '
    "correctness = does the answer state the reference facts; "
    "groundedness = does it avoid inventing facts not in the context."
)

REFERENCE_FACTS = {
    TASKS[0].task: (
        "Spreading activation over typed relations selects relevant "
        "concepts; projection compresses them into a compact view "
        "injected into the prompt; document storage in concepts is "
        "explicitly a boundary violation."
    ),
    TASKS[1].task: (
        "Edges lose weight over time unless reinforced by reuse; "
        "pruning deletes edges below a weight threshold; forgetting "
        "keeps projections current."
    ),
}


def build_world(llm, store: Path) -> World:
    w = World(store_path=store, llm=llm)
    for text, task in CORPUS:
        r = w.ingest_text(text, task=task, source="docs")
        print(
            f"  [{task}] +{len(r.new_concepts)} concepts, "
            f"+{len(r.new_relations)} relations, "
            f"hebbian {len(r.hebbian_relations)}, "
            f"similarity {len(r.similarity_relations)}"
        )
        for label in r.similarity_relations:
            print(f"      sim: {label}")
    return w


def relation_report(w: World) -> dict:
    edges = w.relations.all()
    generic = [e for e in edges if e.semantic_relation == "generic_relation"]
    negative = [e for e in edges if e.relation_type.value == "negative"]
    hebbian_typed = [
        e
        for e in edges
        if not e.is_explicit
        and e.semantic_relation
        not in ("generic_relation", "similarity_kernel")
    ]
    return {
        "total": len(edges),
        "typed_ratio": (
            round(1 - len(generic) / len(edges), 3) if edges else 1.0
        ),
        "generic": len(generic),
        "negative_axis": len(negative),
        "hebbian_llm_typed": len(hebbian_typed),
    }


def llm_judge_variant(llm, context: str, task: AblationTask) -> dict:
    question = task.task
    user = f"## Context\n{context or '(none)'}\n\n## Question\n{question}"
    raw_answer = llm.complete_json(ANSWER_SYSTEM, user)
    try:
        answer = json.loads(
            raw_answer[raw_answer.index("{") : raw_answer.rindex("}") + 1]
        ).get("answer", raw_answer)
    except Exception:
        answer = raw_answer
    judge_user = json.dumps(
        {
            "question": question,
            "reference_facts": REFERENCE_FACTS[question],
            "answer": answer,
        },
        ensure_ascii=False,
    )
    raw = llm.complete_json(JUDGE_SYSTEM, judge_user)
    try:
        scores = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
        return {
            "correctness": float(scores.get("correctness", 0)),
            "groundedness": float(scores.get("groundedness", 0)),
        }
    except Exception:
        return {"correctness": 0.0, "groundedness": 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    llm = AzureOpenAIProvider(model=args.model)

    with tempfile.TemporaryDirectory() as tmp:
        print(f"## Building world from {len(CORPUS)} prose chunks "
              f"(model: {args.model})")
        w = build_world(llm, Path(tmp) / ".world0")

        print("\n## Relation structure after ingest")
        before = relation_report(w)
        print(json.dumps(before, indent=2))

        print("\n## reflect() — generic refinement")
        rr = w.reflect()
        print(f"  retyped: {rr.retyped_relations}")
        print(f"  co_attention: {rr.co_attention_relations}")
        after = relation_report(w)
        print(json.dumps(after, indent=2))

        # Sample typed relations for qualitative inspection.
        print("\n## Sample relations")
        for e in w.relations.all()[:12]:
            s = w.concepts.get(e.source_id)
            t = w.concepts.get(e.target_id)
            if s and t:
                tag = "explicit" if e.is_explicit else "auto"
                print(
                    f"  {s.name} → {e.semantic_relation} → {t.name} "
                    f"[{e.relation_type.value}, {tag}]"
                )

        # Build the observation log for raw_history from corpus prose.
        log = [text for text, _ in CORPUS]

        # Diagnosis: what did each task's projection actually select?
        print("\n## Projection diagnosis per task")
        for task in TASKS:
            p = w.project(task.seeds, task=task.task)
            print(f"  task: {task.task[:50]}")
            print(f"    seed_resolution: {p.seed_resolution}")
            print(f"    selected: {[c.name for c in p.concepts]}")
            print(f"    counter_signals: "
                  f"{[(s.source_name, s.semantic_relation, s.target_name) for s in p.counter_signals]}")

        # Compact-render projection variants — the style meant for
        # prompt injection under token budgets.
        def ctx_projection_compact(world, log, task):
            return world.project(task.seeds, task=task.task).render("compact")

        def ctx_projection_counter_compact(world, log, task):
            from world0.schemas.context import Perspective
            lens = Perspective(
                name="ablation", task=task.task,
                negative_visibility={"conflict": "expose"},
            )
            return world.project(task.seeds, perspective=lens).render("compact")

        variants = dict(VARIANTS)
        variants["projection_compact"] = ctx_projection_compact
        variants["projection_counter_compact"] = ctx_projection_counter_compact

        print("\n## Six-way ablation (offline metrics"
              + (")" if args.no_judge else " + LLM judge)"))
        header = "| variant | tokens | coverage | violations |"
        if not args.no_judge:
            header += " correctness | groundedness |"
        print(header)
        print("|" + "---|" * (header.count("|") - 1))
        for name, builder in variants.items():
            per_task_offline = []
            per_task_judge = []
            for task in TASKS:
                ctx = builder(w, log, task)
                per_task_offline.append(score_variant(ctx, task))
                if not args.no_judge:
                    per_task_judge.append(llm_judge_variant(llm, ctx, task))
            avg = {
                k: sum(m[k] for m in per_task_offline) / len(per_task_offline)
                for k in per_task_offline[0]
            }
            row = (
                f"| {name} | {avg['tokens']:.0f} | {avg['coverage']:.2f} "
                f"| {avg['violation_rate']:.2f} |"
            )
            if per_task_judge:
                corr = sum(j["correctness"] for j in per_task_judge) / len(
                    per_task_judge
                )
                grnd = sum(j["groundedness"] for j in per_task_judge) / len(
                    per_task_judge
                )
                row += f" {corr:.1f} | {grnd:.1f} |"
            print(row)


if __name__ == "__main__":
    main()
