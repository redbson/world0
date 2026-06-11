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
