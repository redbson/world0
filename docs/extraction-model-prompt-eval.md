# Extraction Quality — Model × Prompt Evaluation

**Date:** 2026-06-09
**Scope:** Which LLM and which prompt produce the best concept/relation
extraction for World 0's `ingest_text` pipeline.
**Headline:** On the production extraction prompt, structural quality is
largely **model-agnostic** — so the recommended default is the cheapest
top-tier model: **`gpt-5.4-nano`** (via Azure OpenAI).

Reproduce any of this with
[`scripts/eval_extraction_matrix.py`](../scripts/eval_extraction_matrix.py).

---

## 1. Method

Every cell runs the **real** pipeline end to end:

```
text → <prompt> → <model> (real API) → ConceptExtractor parse → World.ingest → ConceptNode/RelationEdge
```

- **Corpus (7 cases, bilingual)** — each probes one quality property:
  `tech_arch` (typed dependencies + direction), `synonyms_acronyms`
  (RAG dedup), `relation_direction` (cache→latency), `generic_noise`
  (filter `system/data/result/…`), `domain_disambiguation` (Apple
  company vs fruit), `contradiction` (negative evidence), `chinese`
  (preserve CJK concept names).
- **Two prompts** — `default` = the shipped
  `extraction.concepts_relations.system` (rich schema: sense / aliases /
  kind / evidence + rules); `naive` = a flat
  `{concepts, relations}` instruction with no schema.
- **Quality rubric — `quality/9`** = the 5 behavioural checks (RAG dedup,
  Apple split, relation direction, contradiction surfaced, Chinese
  preserved) + 4 hygiene points (0 generic-noise concepts,
  `generic_relation_ratio < 0.1`, `endpoint_drop_rate < 0.05`, 0 failed
  cases). Higher is better.
- **Runs:** 3 per cell, aggregated. Flags reported as *passes / 3*;
  numeric metrics as the mean. This smooths the two single-run-sensitive
  flags (`direction`, `chinese`).
- **Backends:** Azure OpenAI (GPT tiers, DeepSeek-V4, grok-4.3),
  Anthropic (opus-4.8, sonnet-4.6), Zhipu GLM (glm-5.1). `claude-opus-4-8`
  rejects the `temperature` parameter and is wired through a
  no-temperature adapter.

---

## 2. Result — Prompt axis (real GPT, single run)

Same model, only the prompt changes. The prompt is the dominant lever.

| metric | naive | default | better |
|---|:--:|:--:|---|
| generic_relation_ratio | **0.88–0.96** | **0.00** | low |
| RAG dedup | ❌ | ✅ | yes |
| Apple sense split | ❌ | ✅ | yes |
| contradiction surfaced | ❌ | ✅ | yes |
| noise concepts | higher | 0–2 | low |
| accepted concepts | inflated (≈2×) | focused | quality > quantity |

A naive prompt collapses every relation to an untyped `generic_relation`
and loses dedup / sense-splitting / contradiction **regardless of model**.
The shipped `default` prompt fixes all of this. **Prompt quality
dominates model choice.**

---

## 3. Result — Cross-model (default prompt, 3 runs aggregated)

Flags = passes/3; numbers = mean.

| metric | gpt-5.4 | 5.4-mini | **5.4-nano** | DeepSeek-Pro | DeepSeek-Flash | grok-4.3 | glm-5.1 | opus-4.8 | sonnet-4.6 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **quality/9** | **9.0** | 8.0 | **9.0** | **9.0** | 8.0 | **9.0** | 8.7 | **9.0** | 6.7 |
| concepts# | 44.7 | 48.7 | 42.7 | 41.3 | 40.0 | 31.7 | 43.3 | 37.0 | 36.7 |
| noise | 0 | 2.3 | 0 | 0 | 1.3 | 0 | 0.3 | 0 | 1.0 |
| drop_rate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| generic_rel | .04 | 0 | 0 | .02 | 0 | .01 | .01 | 0 | 0 |
| RAG dedup | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| Apple split | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| direction | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | **2/3** |
| contradiction | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| chinese | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** | 3/3 | **0/3** |
| failed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Ranking (mean quality/9):**
`gpt-5.4-nano = DeepSeek-V4-Pro = grok-4.3 = claude-opus-4-8 = gpt-5.4 (9.0)`
> `glm-5.1 (8.67)` > `DeepSeek-V4-Flash = gpt-5.4-mini (8.0)`
> `claude-sonnet-4-6 (6.67)`

---

## 4. Findings

1. **On the production prompt, extraction quality is essentially
   model-agnostic.** RAG dedup, Apple split, contradiction, and
   `drop_rate=0` are perfect across all 9 models. Differences are confined
   to a couple of generic-noise concepts and two sensitive flags.
2. **3 runs corrected two single-run artifacts:**
   - `glm-5.1`'s earlier "Chinese ❌" was **noise** — now 3/3. GLM is fine
     for Chinese.
   - `claude-sonnet-4-6`'s "Chinese ❌" is a **stable, reproducible
     failure** (0/3): it translates Chinese input into English concept
     names instead of preserving the source language. Its `direction`
     is also flaky (2/3).
3. **Bigger ≠ better here.** `gpt-5.4-nano` ties the perfect tier with the
   most expensive models. The extraction task is structured parsing that
   the prompt already constrains; model headroom buys nothing.
4. **DeepSeek-V4-Flash (8.0)** passes every behavioural flag but averages
   ~1.3 noise concepts; **DeepSeek-V4-Pro (9.0, 0 noise)** is strictly
   better at comparable cost.

---

## 5. Recommendation

**Default extraction model: `gpt-5.4-nano` (Azure OpenAI).**
Top-tier quality (9.0/9), zero noise, all five behavioural flags stable
across runs, Chinese preserved, lowest cost/latency in the GPT family,
single Azure key+endpoint.

Set it:

```bash
pkm model set extraction --provider azure-openai --model gpt-5.4-nano
# or copy the template into your store:
cp models.example.json ~/.pkm_world/models.json
```

Required environment:

```bash
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
export AZURE_OPENAI_KEY="<key>"     # or AZURE_OPENAI_API_KEY
```

**Alternatives**

- `azure-openai/DeepSeek-V4-Pro` — 9.0, strong Chinese, low cost; the best
  Chinese-heavy alternative.
- `glm-5.1` — 8.67, independent key, Chinese OK; good when using existing
  GLM quota.
- `gpt-5.4` / `grok-4.3` — 9.0; only worth the extra cost if you need the
  bigger backbone for other operations.

**Avoid for this (bilingual) project**

- `claude-sonnet-4-6` — does not preserve Chinese concept names (0/3).
- `claude-opus-4-8` — perfect quality but most expensive and needs special
  no-temperature handling; poor value for extraction.
- `gpt-5.4-mini` / `DeepSeek-V4-Flash` — dominated by `gpt-5.4-nano` /
  `DeepSeek-V4-Pro`.

Per-operation routing means you can keep `gpt-5.4-nano` for `extraction`
and assign a stronger model to `answer` / `agent_loop` if desired
(`pkm model set answer --provider ... --model ...`).

---

## 6. Bug found during evaluation

Real models occasionally emit a **same-name self-loop relation**
(`["X", "X", ...]`). `IngestPipeline._step_relations`
([`src/world0/world/_ingest.py`](../src/world0/world/_ingest.py)) guards
only the *different-name / same-id* case
(`if src.id == tgt.id and src_name != tgt_name: continue`), so an identical
endpoint pair falls through to `RelationManager.discover()` which raises
`ValueError("Cannot create a self-relation")` — crashing `ingest_text`.

**Fixed:** the guard in `_step_relations` was relaxed to
`if src.id == tgt.id: continue` (drop all self-loops). Regression tests:
`test_ingest_skips_self_loop_relation`
([`src/world0/world/tests/test_pipelines_with_fakes.py`](../src/world0/world/tests/test_pipelines_with_fakes.py))
and `TestSelfLoopRelations`
([`tests/test_integration.py`](../tests/test_integration.py)).
The evaluation harness also sanitizes self-loops before ingest as a
defensive layer.

---

## 7. Caveats

- Single Azure resource, single 7-case corpus; results indicate direction,
  not a leaderboard. Re-run with `--runs 5` and a larger corpus before
  hard production decisions.
- Real LLM calls are non-deterministic; the `direction` and `chinese`
  flags are the most naming/resolution-sensitive — trust the multi-run
  pass-rate, not any single run.
- Costs real tokens. `quality/9` weights structural correctness, not
  downstream answer quality.

---

## 8. Reproduce

```bash
# list catalog + which backends have credentials
python scripts/eval_extraction_matrix.py --list

# the full matrix used here (needs Azure + Anthropic + GLM creds in env)
python scripts/eval_extraction_matrix.py --models all --runs 3 --md report.md

# just the recommended model, both prompts
python scripts/eval_extraction_matrix.py --models gpt-5.4-nano --prompts both --runs 3
```

Unavailable backends (missing keys) are skipped automatically, so the
harness runs anywhere and evaluates whatever it can reach.
