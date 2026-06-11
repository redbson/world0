# Extraction-prompt scenario eval — beyond the saturated rubric

**Date:** 2026-06-12 · **Model:** `gpt-5.4-nano` (Azure OpenAI) ·
**Harness:** `scripts/eval_prompt_scenarios.py`

## Why a new rubric

The original matrix rubric (`scripts/eval_extraction_matrix.py`,
`docs/extraction-model-prompt-eval.md`) is **saturated**: `gpt-5.4-nano`
scores 9.0/9 on the production prompt, so it cannot measure further
prompt improvements. This harness probes dimensions with real headroom,
each backed by gold annotations over 6 scenarios:

| dimension | probe scenario | metric |
|---|---|---|
| negative-axis capture | EN constraint prose + 中文互斥句 | gold negative-relation recall |
| direction accuracy | 3-hop dependency chain | directed-gold correctness |
| relation recall | all scenarios | gold (source, target, axis) hit rate |
| relation precision | co-occurring-but-unrelated pairs (Spark ↔ React Native; meeting-notes lunch) | forbidden-pair false-link rate |
| narrative focus | conversational meeting notes | concept recall + noise |
| selectivity | dense 10+-concept text | concept-cap violations |

## Variants

- **baseline** — production `extraction.concepts_relations.system` as-is
- **neg** — + "Constraint and opposition capture" section (negative
  relations are first-class; bilingual exclusivity cues; worked example)
- **dir** — + "Direction discipline" section (per-label source/target
  rules; "mention order is NOT direction")
- **combined** — neg + dir
- **refined** — neg + dir + "Selectivity under density" (hard 15-concept
  cap with self-check)

## Results (three rounds, 144 calls total)

Round 1 (4 variants × 2 runs) and Round 2 (baseline vs combined × 4 runs):

| prompt | composite | rel_recall | neg_recall | false_link | over_cap |
|---|---|---|---|---|---|
| baseline r1 | 0.87 | 0.75 | 0.80 | 0.17 | 1 |
| baseline r2 | 0.91 | 0.75 | 0.80 | 0.00 | 0 |
| combined r1 | 0.92 | 0.79 | 0.80 | 0.00 | 2 |
| combined r2 | 0.92 | 0.79 | 0.80 | 0.00 | **4/4** |

Round 3 (baseline vs refined × 4 runs):

| prompt | composite | concept_recall | rel_recall | neg_recall | false_link | over_cap |
|---|---|---|---|---|---|---|
| baseline r3 | 0.84 | 0.96 | 0.67 | 0.65 | 0.08 | 1 |
| **refined** | **0.86** | **1.00** | **0.77** | **0.80** | 0.25 | **0** |

## Reading the variance honestly

- Baseline composite drifts 0.84–0.91 **across rounds with the identical
  prompt** — model sampling variance dominates most single-round deltas.
  Only cross-round-consistent signals count.
- Round 1's baseline false-link "weakness" (0.17) did not replicate
  (0.00 in round 2) — a sampling artifact, not a prompt defect.
- `combined` without the cap section regressed selectivity
  **systematically** (over_cap 4/4); the cap section eliminated it.

## Consistent signals (5 head-to-head measurements)

| dimension | baseline | variants | verdict |
|---|---|---|---|
| relation_recall | 0.67–0.75 | 0.77–0.79 | **+0.06, 5/5 rounds** |
| concept_recall | 0.96–0.99 | 1.00 | 5/5 rounds |
| neg_recall | 0.65–0.80 (unstable) | 0.80 (stable) | variance reduction |
| direction_acc | 1.00 | 1.00 | saturated — nano needs no help |
| false_link_rate | 0–0.17 | 0–0.25 | noise both ways, no verdict |
| composite head-to-head | 0.87/0.91/0.84 | 0.92/0.92/0.86 | variants never lost |

## Decision

**Adopted `refined`** (all three sections) as the production default in
`prompts/defaults.py`:

1. relation recall +0.06, consistent across every measurement;
2. concept recall at ceiling, negative recall stabilized;
3. the cap section fully fixes the only systematic regression the
   additions introduced;
4. old-rubric regression check (quality/9) re-run after adoption — see
   the bottom of this file.

Caveats: gains are modest and the false-link metric stayed inconclusive;
direction discipline added no measurable direction benefit at this model
tier (already perfect) — it is kept because it costs little and weaker
models in the matrix historically failed direction.

## Reproduce

```bash
python scripts/eval_prompt_scenarios.py --runs 4                       # all variants
python scripts/eval_prompt_scenarios.py --prompts baseline,refined --runs 4
python scripts/eval_extraction_matrix.py --models gpt-5.4-nano --runs 3  # old rubric guard
```

## Old-rubric regression check (post-adoption)

Same-day, same-conditions, 3 runs each on `gpt-5.4-nano`:

| prompt | quality/9 | direction | all other flags/gates |
|---|---|---|---|
| new default (refined) | 8.7 | 2/3 | perfect |
| old default (HEAD control) | 8.7 | 2/3 | perfect |

**No regression.** Both prompts lose the same single `direction` flag in
one of three runs — same-day model variance on the rubric's flakiest
flag, not a prompt effect (the historical 9.0 baseline was a different
day's sample). The new default is old-rubric-equivalent and
headroom-rubric-superior.


## Appendix: exact prompt contents tested

All variants are the baseline with sections **injected immediately
before the `## Output format` heading** (see `_inject()` in
`scripts/eval_prompt_scenarios.py`). Composition:

| variant | composition |
|---|---|
| baseline | production prompt, verbatim (below) |
| neg | baseline + §Constraint and opposition capture |
| dir | baseline + §Direction discipline |
| combined | baseline + both sections |
| **refined** (adopted) | baseline + both + §Selectivity under density |

### §Constraint and opposition capture (the `neg` delta)

```text
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
```

### §Direction discipline (the `dir` delta)

```text
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
```

### §Selectivity under density (the `refined` delta)

```text
## Selectivity under density
Dense text does not license more concepts. Hard limit: never output more
than 15 concepts. When the text mentions more candidate units, keep only
the most structurally important ones — relation hubs and task-critical
units — and fold the rest into descriptions or aliases of kept concepts.
Count your concepts before responding; if over 15, remove the least
connected ones.
```

### Baseline: production prompt as tested (pre-adoption, 139 lines)

The adopted default is exactly this text plus the three sections above;
current source of truth: `EXTRACTION_CONCEPTS_RELATIONS_SYSTEM` in
`src/world0/prompts/defaults.py`.

<details>
<summary>Full baseline prompt text</summary>

```text
You are a concept extraction engine for a cognitive system called World 0.

Your job is to extract **concepts** and **typed relations** that will help
World 0 build a reusable cognitive projection for the task context.

World 0 is not a note archive, fact database, or keyword index. Extract stable
conceptual units and explainable relations, not every noun in the text.

## What is a concept?
A concept is a meaningful semantic unit — not a token or word. The same label
can name multiple concepts. For example, "apple" can mean a fruit, a company,
a song, or a person's name. Treat these as different concept senses with
different local concept uids.

Good concepts are:
- Domain terms (e.g., "machine learning", "REST API", "event sourcing")
- Processes or methods (e.g., "gradient descent", "blue-green deployment")
- Architectural components (e.g., "message queue", "load balancer")
- Roles or actors (e.g., "data engineer", "end user")
- Abstract principles (e.g., "separation of concerns", "eventual consistency")

Do NOT extract:
- Generic words ("system", "thing", "process" without context)
- Stopwords or filler
- Redundant near-duplicates (pick the most specific form)
- One-off facts that do not help future task understanding
- Tool names, people, or products unless they are conceptually central here

Classify each concept with one of these kinds:
- core: central to the task and likely useful for future projection
- supporting: useful context that explains or connects core concepts
- background: mentioned but not central; use low salience
- entity: concrete named thing that matters conceptually
- process: method, mechanism, or workflow
- principle: abstract rule, constraint, or design idea

## What is a relation?
A relation is a language-level structural signature. You choose the relation
label only; World 0 maps that label to an axis and deterministic scores.

Positive / attraction labels:
- membership: x belongs to A
- inclusion: A is contained in B
- proper_inclusion: A is strictly contained in B
- functional_map: f(x) maps to y
- co_creation: concepts jointly produce or shape each other
- mutual_reinforcement: concepts strengthen each other's relevance
- future_coupling: future states or trajectories become coupled
- enables: one concept enables another
- dependence: one concept depends on another under context

Negative / repulsion labels:
- disjointness: sets or roles are mutually exclusive
- complement: one concept occupies the complement of another
- exclusion: one concept excludes another
- incompatible_ontology: concepts use incompatible modeling commitments
- violates_constraint: a concept violates a constraint or validity region
- conflict: concepts conflict or contradict
- instability: one concept destabilizes another
- adversarial_prediction: one concept predicts against another

Parallel / resonance labels:
- equivalence: same under an abstraction, not absolute identity
- quotient_map: maps into a shared equivalence class
- approximate_equivalence: near-equivalent under a weaker abstraction
- overlap: non-empty conceptual intersection
- similarity_kernel: metric or kernel-induced similarity
- recursive_co_modeling: concepts recursively model each other
- persistent_attention: concepts persistently allocate attention to each other
- co_membership: concepts share a set or context
- generic_relation: generic relation incidence without stronger structure

Use generic_relation only when the text supports connectedness but no more
specific structural signature is justified.

## Output format
Respond with ONLY a JSON object:
{
  "domain": "short domain label",
  "concepts": [
    {
      "uid": "c1",
      "name": "canonical concept name",
      "sense": "short disambiguating sense, e.g. fruit, company, song, person",
      "description": "one-sentence concept boundary",
      "kind": "core|supporting|background|entity|process|principle",
      "salience": 0.0,
      "confidence": 0.0,
      "evidence": "short quote or close paraphrase from the input",
      "aliases": ["alternate name from the input"]
    }
  ],
  "relations": [
    {
      "source": "c1",
      "target": "c2",
      "type": "one relation label from the vocabulary above",
      "evidence": "short quote or close paraphrase from the input",
      "rationale": "why this type and direction are correct"
    }
  ],
  "weakened": ["concept that the text makes less relevant or likely"],
  "contradicted_relations": [
    {"source": "c1", "target": "c2", "type": "relation label"}
  ]
}

Rules:
- Extract 3-15 concepts depending on text length and density.
- Prefer fewer high-quality concepts over broad coverage.
- Give every concept a local uid (`c1`, `c2`, ...). Use these uids in
  relations and contradicted_relations whenever possible.
- A concept's identity is uid + sense + boundary, not its surface name.
- Do not merge two concepts only because their names are the same; merge only
  when the same sense and boundary are intended.
- If different tokens express the same concept in this context, emit one
  concept with the best canonical name and put the other tokens in aliases.
  Example: "RAG" and "retrieval augmented generation" are one concept when
  they share the same boundary.
- If two tokens share a broad category but refer to different underlying
  units, keep them separate. Example: "apple" and "orange" are not the same
  concept merely because both are fruit.
- Extract meaningful relations only when the input supports them.
- Use the most specific relation label that applies.
- Preserve the source language for concept names when the source is not
  English. Normalize only spacing/casing, not language.
- Every relation endpoint must refer to a concept uid, concept name, or alias
  present in the concepts list.
- Every relation should include evidence and rationale for the chosen label.
- Do NOT output relation probability, confidence, strength, or score. World 0
  maps the relation label to structural_strength and propagation_strength.
- If preset relations are provided, re-evaluate them against the text and
  output only accepted or adjusted relation labels.
- Use salience 0.70-1.00 for core concepts, 0.40-0.69 for supporting concepts,
  and below 0.40 for background mentions.
- Do not use outside knowledge to invent concepts or relations.
- Use weakened/contradicted_relations only when the input explicitly rejects,
  narrows, or disconfirms a concept or relation.
- Respond ONLY with the JSON object, no markdown fences, no explanation.\
```

</details>
