# Changelog

All notable changes to World 0 are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
