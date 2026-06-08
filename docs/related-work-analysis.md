# World 0 vs. the Agent-Memory / Concept-Graph Landscape

This document analyses World 0's core theory against current agent-memory and
concept-graph systems (products + papers, 2024–2026), identifies where World 0
is genuinely differentiated, where it is behind, and prioritises concrete
optimizations. It is the analytical basis for the optimization work tracked
alongside the other `docs/*-design.md` and `docs/*-priorities.md` files.

## 1. World 0 in one paragraph (recap of the core theory)

World 0 is a **cognitive layer**, not a memory store. Its five first-class
objects are Concept, Relation, Context, Activation, Projection. The read path
is `query → seed extraction → spreading activation → MMR projection → prompt
injection`; the write path is `text → LLM extraction → Observation → ingest`
(get-or-create concepts, reinforce, Hebbian co-occurrence learning, maturity
evolution, decay via `reflect()`). The explicit thesis (`docs/world0-paper.md`)
is that the Agent's *understanding problem* is distinct from its *storage,
memory, and fact problems* — World 0 only owns understanding, and its output is
a small, task-conditioned, structured projection rather than a large graph.

The relation model has since evolved to three cognitive **axes**
(positive / negative / parallel) with `probability` as a semantic belief score
(`docs/relation-probability-redesign.md`), plus color-field community dynamics.

## 2. The comparison set

| System | Type | Core idea | Primary unit |
| --- | --- | --- | --- |
| **Zep / Graphiti** | Product (OSS) | Bitemporal knowledge graph; edges carry validity intervals `(t_valid, t_invalid)` + ingestion time; hybrid semantic+BM25+graph retrieval | Entity / fact edge |
| **Mem0 / Mem0g** | Product (OSS) | Extract→consolidate memories; graph variant adds entity+relation+conflict detection | Memory item |
| **HippoRAG** | Paper (NeurIPS'24) | Hippocampal indexing: KG + **Personalized PageRank** for single-shot multi-hop retrieval | Passage / KG node |
| **A-MEM** | Paper (NeurIPS'25) | Zettelkasten agentic memory: each note gets attributes+links; **memory evolution** updates neighbors when new notes arrive | Note |
| **GraphRAG** | Product (MS) | Leiden **hierarchical communities** + multi-level community summaries for global sensemaking | Entity + community |
| **Conceptual Spaces** (Gärdenfors) | Theory | Concepts as convex regions over geometric **quality dimensions**; betweenness/similarity are geometric | Concept region |
| **ACT-R** | Cognitive arch. | Retrieval = base-level activation (power-law decay of use) + **spreading activation** from context | Chunk |

## 3. Where World 0 is genuinely differentiated (keep / defend)

1. **Understanding vs. storage boundary.** Every other *product* in the set is
   fundamentally a retrieval/memory store that bolts on a graph. World 0's
   refusal to be a fact DB (CLAUDE.md Rule 6) is the actual moat. None of
   Zep/Mem0/GraphRAG produce a *task-conditioned projection* as the primary
   output — they answer "what do I recall?" not "how should this task be
   understood?".
2. **Context-sensitivity is first-class.** Activation is task-aware
   (`γ_task` in the propagation formula), so the same seed yields different
   projections under different tasks. GraphRAG's local/global split is a *mode*
   switch, not continuous task conditioning; Zep/Mem0 ranking is query-similarity,
   not task-conditioned relevance.
3. **MMR projection (relevance + low redundancy).** This is closer to "context
   engineering" than to retrieval. Competitors return top-k by score; World 0
   deliberately shapes a compact, non-redundant view.
4. **Lifecycle / evolution as a built-in.** Maturity (embryonic→core→fading),
   decay, and Hebbian growth give World 0 a *living* structure. A-MEM is the
   only comparator with comparable "evolution"; the products mostly accrete.

These are exactly the CLAUDE.md priorities (concept integrity → relation
quality → context → projection → evolvability). The analysis below only
recommends changes that *strengthen* this chain — not features that would drift
World 0 toward being "yet another graph memory store".

## 4. Where World 0 is behind (candidate optimizations)

### G1 — No temporal validity / invalidation of relations *(from Zep/Graphiti)*
World 0 has `temporal_relevance()` (soft freshness decay) and
`disconfirmation_count`, but a contradicted relation is only *weakened*, never
*invalidated with a time boundary*. Zep's bitemporal model (when a fact was
true vs. when it was observed) lets an agent reason about *superseded* beliefs.
World 0 already ingests `contradicted_relations` — the missing piece is an
explicit invalidation state + timestamp rather than silent weight decay.
**Aligned with World 0?** Yes, if scoped as *relation quality*, not as event
storage. Risk: must not become an event log (Rule 6).

### G2 — No network-level diagnostics of projection health *(gap vs. all)*
World 0 can build projections but cannot *measure* whether its graph is
becoming diffuse/noisy — e.g. whether a concept spreads activation over many
weakly-distinguished neighbors (which produces broad, low-value projections),
or whether `parallel`/generic edges dominate. GraphRAG's own documented
weakness is "uneven hierarchies, inconsistent granularity" — World 0 has no
instrument to even detect this. `docs/world-network-entropy-design.md` already
specifies the fix (per-node relation-distribution entropy → world average).
**Aligned?** Strongly — it directly serves projection quality + evolvability,
adds no storage responsibility, and is fully local to the concept-relation
graph. **Lowest risk, highest readiness.**

### G3 — Activation is fixed-depth BFS, not converged diffusion *(from HippoRAG / ACT-R)*
HippoRAG shows Personalized PageRank over a KG beats iterative retrieval for
multi-hop, single-shot, and 10–20× cheaper. World 0's spreading activation is a
depth-limited multiplicative walk. A PPR-style (or ACT-R base-level +
spreading) variant could improve multi-hop projection recall without changing
the data model. **Aligned?** Yes (it's literally the Activation layer), but
**higher risk** — it touches the core dynamics that many tests pin, so it
should follow G2 once diagnostics exist to measure the before/after.

### G4 — Context is implicit, not an object *(from Conceptual Spaces; paper §11.3 self-admits)*
The paper lists "lightweight context modeling" as a known limitation. Context
is currently just task-string affinity + history. Gärdenfors' quality
dimensions suggest a richer **Context object** (active dimensions / domain /
perspective weights) that biases activation and projection compositionally.
**Aligned?** Yes — Context is a first-class concept in the theory but a
second-class citizen in code. Medium risk (schema + plumbing).

### G5 — No multi-level "global" projection *(from GraphRAG)*
World 0 projections are local (activation neighborhoods). For broad
"sensemaking" tasks, GraphRAG-style community summaries give a global view.
World 0 has `communities/` + color-field dynamics already — a community-summary
projection mode is incremental. **Aligned?** Partially; risk of drifting toward
summarization-as-archive. Lower priority than G2/G3.

### G6 — Memory-evolution feedback to neighbors *(from A-MEM)*
When a new concept is ingested, World 0 reinforces it and may form Hebbian
edges, but does not *re-evaluate the descriptions/attributes of existing
neighbors* the way A-MEM does. This is a natural extension of `reflect()`.
Medium risk (LLM cost, can introduce drift).

## 5. Prioritization

Ordering by `value × readiness ÷ risk`, constrained to "strengthens the
concept→relation→context→projection chain":

| Rank | Item | Why first | Risk |
| --- | --- | --- | --- |
| **1** | **G2 network entropy diagnostics** | Already designed; pure read-only metric; gives us the *instrument* needed to safely evaluate every later change | Low |
| 2 | G1 relation invalidation w/ temporal boundary | Extends existing `contradicted_relations`; clear relation-quality win | Low–Med |
| 3 | G3 PPR / base-level activation variant | Biggest projection-quality upside, but needs G2's metric to verify | Med–High |
| 4 | G4 explicit Context object | Closes a self-admitted theory gap | Med |
| 5 | G5 community-summary projection mode | Reuses communities module | Med |
| 6 | G6 neighbor evolution in reflect() | Nice-to-have; cost + drift risk | Med |

## 6. Decision for this iteration

Implement **G2 (network entropy diagnostics)** first. Rationale:

- It is the only item that is *fully specified*, *low-risk*, and *read-only*.
- It is a prerequisite instrument: G3/G4/G5 all change graph shape, and we
  currently have no quantitative way to tell whether a change made the world
  more focused or more diffuse. Entropy + relation-type entropy give exactly
  that signal.
- It maps a concrete competitor lesson (GraphRAG's "inconsistent granularity"
  failure mode) onto a World-0-native metric, without importing any storage or
  retrieval responsibility.

The design doc (`docs/world-network-entropy-design.md`) predates the axis-based
relation model, so the implementation re-grounds `effective_weight` on the
*current* model: `probability × RELATION_TYPE_FACTOR[axis] × explicitness`,
where explicitness is `1.0` for `is_explicit` edges and `0.75` for Hebbian.

## 7. Non-goals reaffirmed

None of the recommendations add: raw document storage, an event log, a vector
DB, or a workflow engine. Where a competitor's strength (Zep's bitemporality,
GraphRAG's summaries) risks pulling World 0 toward storage, the recommendation
is scoped to the *cognitive* slice only (relation belief state; conceptual
projection), per CLAUDE.md Rules 1 and 6.

## Sources

- Zep / Graphiti — temporal KG for agent memory:
  <https://arxiv.org/abs/2501.13956>, <https://github.com/getzep/graphiti>
- HippoRAG — neurobiologically inspired long-term memory (PPR over KG):
  <https://arxiv.org/abs/2405.14831>
- A-MEM — agentic memory, Zettelkasten linking + evolution:
  <https://arxiv.org/abs/2502.12110>
- GraphRAG — local-to-global query-focused summarization:
  <https://arxiv.org/abs/2404.16130>,
  <https://microsoft.github.io/graphrag/>
- Conceptual Spaces (Gärdenfors) — geometry of thought / quality dimensions:
  <https://mitpress.mit.edu/9780262572194/conceptual-spaces/>,
  <https://arxiv.org/pdf/1701.00464>
- ACT-R — base-level + spreading activation:
  <https://www.sciencedirect.com/science/article/abs/pii/S1389041716302121>
