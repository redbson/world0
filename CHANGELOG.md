# Changelog

All notable changes to World 0 are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Design-review adoptions** (from the 2026-06 external design review;
  full assessment in `docs/world0-relation-optimization.md` §6):
  - *Negative visibility policy + counter-signals* — negative relations
    now distinguish propagation (always inhibit) from projection
    visibility (`NEGATIVE_VISIBILITY_DEFAULTS`: suppress / expose /
    conditional per semantic relation).  Exposed relations surface as
    `Projection.counter_signals` and render as a `### Constraint
    Warnings` section — the agent sees *why a path is closed* even
    though inhibition removed the concept from the view.
    `violates_constraint` exposes by default; `conflict`/`instability`/
    `incompatible_ontology` are conditional, opted in per Perspective
    (`negative_visibility`; the built-in debug profile exposes
    conflict + instability).  Default render is byte-identical when no
    signals exist.
  - *Reflect-time generic-relation refinement* — a budgeted
    `RelationRefiner` re-judges the stock of `generic_relation` edges
    each `reflect()` cycle through the shared `RelationTypingJudge`
    (same prompt as ingest-time Hebbian typing): nameable structure is
    re-typed in place (evidence preserved, direction flippable);
    pairs with no nameable structure demote to `co_attention_only`,
    which halves their projection salience — frequent co-occurrence
    must not masquerade as semantic structure.  Results surface in
    `ReflectResult.retyped_relations` / `co_attention_relations`.
  - *Governance metrics* — `Projection.generic_pressure` (share of
    generic edges among projected relations) makes typed-structure
    pollution observable per projection.
  - *Agent-context ablation harness*
    (`scripts/eval_agent_ablation.py`) — scores six context variants
    (none / raw history / lexical recall / graph neighbors /
    projection / projection+counter-signals) on coverage,
    token cost and silent-constraint-violation rate over a fixture
    world; CI smoke test locks the mechanism facts (raw dumps and
    naive 1-hop neighborhoods leak forbidden concepts silently;
    projections never do).
  - *Score-semantics consumption boundaries* documented as an
    invariant (`docs/world0-core-concepts.md` §4.2b): each of
    structural_strength / propagation_strength / weight / confidence /
    probability has exactly one consumer class.
- **Relation-usage optimizations** (research-backed; evidence and
  rationale in `docs/world0-relation-optimization.md`):
  - *Seed specificity* — activation seeds are weighted by concept
    rarity (IDF-like, HippoRAG node specificity): seeds attested by
    fewer distinct sources keep full activation while ubiquitous seeds
    are blended down relative to the rarest seed in the set.  Tunable
    via `ActivationConfig.seed_specificity_weight`; neutral for a
    single seed.
  - *Fan dilution* — high-fan hub concepts spread less activation per
    edge (ACT-R fan effect), preventing over-connected nodes from
    flooding the network under fixed declared edge strengths.  Smooth
    log dilution past `FAN_DILUTION_THRESHOLD` (default 7) — unlike
    ACT-R's `smax − ln(fan)` it never flips inhibitory.
  - *Similarity linking* — `SimilarityLinker` (`dynamics/similarity.py`,
    `SimilarityLinkerP` protocol) creates explicit PARALLEL similarity
    edges between newly ingested concepts and existing near-duplicates,
    as a new ingest step reported in
    `IngestResult.similarity_relations`.  Two-stage (A-MEM/Mem0-style):
    signature-token matching recalls candidates cheaply, then — when an
    LLM provider is configured — the new `similarity.judge.system`
    prompt judges *meaning*, choosing `equivalence` /
    `approximate_equivalence` / `similarity_kernel` per pair.  The
    judge links zero-stem-overlap pairs ("vector database" ≈ "vector
    store", cross-language pairs) and rejects token lookalikes
    ("chain_2" vs "chain_3").  Without an LLM a strict lexical rule
    applies (Jaccard ≥ 0.5, single-token signatures never linked);
    judge failures fall back to that rule so ingest never breaks.
    Similarity strength is encoded as the edge weight, capped at the
    Hebbian ceiling (0.7) so implicit evidence never outranks declared
    relations.
  - *LLM-typed Hebbian relations* — co-occurrence pairs crossing the
    Hebbian threshold are now typed by the system LLM (new
    `relation.typing.system` prompt): the model picks the most specific
    semantic relation from the canonical 26-type inventory with
    direction and confidence, or rejects coincidental pairs (`"none"` →
    no edge; the pair re-accumulates and is re-judged if it keeps
    co-occurring).  This closes the last untyped relation-creation path
    (Rule 3: generic only as fallback) — with an LLM configured, every
    semantic decision (text extraction, similarity judging, discovered-
    relation typing) now runs through the same system-selected model.
    Without an LLM, or on judge failure, the original untyped PARALLEL
    edge is created as before.  Shared LLM-output parsing moved to
    `llm/parsing.py` (`extract_json`), used by extraction, similarity
    and relation typing alike.

### Changed
- **Repository split — World 0 is now the pure cognitive core.** The PKM /
  Agent application layer (`agents/` and `models/`) has been extracted into a
  separate project, **`world0-pkm`** (imported as the `pkm` package), which
  depends on this package via pip. `world0` no longer ships the `pkm` /
  `pkm-web` / `pkm-gui` console scripts, the `web` / `gui` extras, or the
  launcher scripts — those moved to `world0-pkm`. The dependency direction was
  already one-way (core never imported `agents`), so the core API is unchanged:
  `World`, `Observation`, `Projection`, `Perspective`, extraction, dynamics,
  projection, and persistence all stay here. Per-operation model routing
  (`models/`, the `pkm model …` CLI) moved with the agent layer.

### Added
- **Cognitive-quality optimization (4 phases)** — deepens context handling,
  projection quality, seed robustness, and the evaluation loop without
  touching the three-axis relation model or the autonomy-layer boundary:
  - *Context/Perspective deepening* — `Perspective` now weights individual
    semantic relations (`semantic_relation_weights`, resolved via
    `weight_for_relation`), not just the three axes; task affinity is graded
    token-containment instead of substring (`dynamics/affinity.py`); a
    `perspectives/` package ships named profiles (`default`/`debug`/`design`/
    `research`) with a persistable `PerspectiveRegistry`; a light `Context`
    model joins `Perspective`. `World.project(perspective=...)` accepts a
    profile name.
  - *Projection quality + explainability* — perspective now reaches the
    `ProjectionEngine` (relevance gains domain affinity; relations are
    ranked/filtered by perspective-weighted signal); every projected concept
    carries an `ActivationTrace` best-path (`activate_traced`), surfaced via
    `Projection.explain()`; `Projection.render(style=...)` adds
    `compact`/`detailed` styles (`projection/render.py`) with the `default`
    output byte-for-byte unchanged.
  - *Seed robustness + calibration* — missed seeds resolve by domain
    disambiguation then fuzzy token-containment (`resolve_in_context`,
    `SignatureMatcher.find_by_containment`), recorded in
    `Projection.seed_resolution`; activation/projection coefficients move onto
    tunable `ActivationConfig`/`ProjectionConfig`; time is injectable (`now=`)
    for deterministic evaluation; `scripts/sweep_projection_quality.py` reports
    coefficient sensitivity.
  - *Evaluation loop + consumer integration* — `World.apply_feedback()` makes
    usage feedback a public facade API (the autonomy layer no longer needs PKM
    internals); `scripts/eval_projection_matrix.py` scores projections offline
    (P@k/NDCG/noise/typed-ratio/inhibition) with an optional LLM judge;
    `ask`/`explore` accept perspectives and `explore` stops bypassing the
    projection layer; `agent_chat` injects a compact cognitive-context block;
    `research_topic` grounds itself in prior world-knowledge (research-lens
    projection injected into the brief) and renders its closing projection
    under the `research` profile; `ask(auto_feedback=True)` (experimental,
    off by default) reinforces the projected concepts an answer actually used.
- **External-agent consultations** — read-only consultations with the
  system-installed `claude` and `codex` CLIs, each run in an isolated
  per-problem workspace. Exposed as `PKMAgent.consult_external_agent`, the
  `consult_claude_code` / `consult_codex` agent tools, `/claude` and `/codex`
  commands, and `claude` / `codex` CLI subcommands. Provider aliases
  (`claude` → anthropic, `codex` → openai) flow through model detection,
  the web status endpoint, and CLI/GUI provider options.
- **Real-LLM extraction-quality test suite** (`tests/test_extraction_quality_llm.py`)
  — end-to-end `text → ConceptNode` quality checks (synonym/acronym dedup,
  generic-noise filtering, relation direction, domain-sense split,
  contradiction handling, Chinese language preservation, cross-text identity).
  Skips automatically when no LLM provider is configured.
- **Color-field dynamics** — community-born color diffusion over the relation
  graph (`dynamics/color_diffusion.py`, `dynamics/community.py`,
  `communities/`).
- **Spaces** — isolated concept worlds with their own stores and sessions.
- **Source library** — raw-source provenance layer decoupled from extracted
  concepts (`sources/`, `schemas/source.py`).
- **Network-entropy metrics** — structural diagnostics of conceptual
  attention concentration (`metrics/entropy.py`).
- **Per-operation model configuration** and a configurable prompt registry
  (`models/`, `prompts/`).

### Changed
- **Relations** now use a three-axis model (positive / negative / parallel)
  with a deterministic semantic-relation → structural/propagation mapping.
- **World** internals split into a modular `world/` package
  (facade + ingest/reflect/status pipelines) backed by Protocol-satisfying
  engines.
- **Concept identity** resolved via semantic identity keys and
  signature-based consolidation (sense-aware dedup and merge/split ops).

### Fixed
- `IngestResult` now reports endpoint disconfirmation: when a contradicted
  relation has no existing edge, the weakened endpoint concepts are recorded
  in `weakened_concepts` instead of being applied silently.

## [0.2.0]

- Strengthen World 0 as a configurable concept system.

## [0.1.0]

- Initial World 0 concept-world agent.
