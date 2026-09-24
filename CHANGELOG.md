# Changelog

All notable changes to World 0 are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Cognitive clock** (`schemas/clock.py`) — time in World 0 is counted in
  observations: `World.clock` advances once per `ingest()` and is persisted
  in `state.json` (`tick`); every concept/relation carries `*_tick`
  coordinates next to its timestamps, and decay, freshness, evidence
  floors and community coupling are functions of ticks elapsed.  The wall
  clock survives only as a slow drift term (`WALL_TICKS_PER_HOUR = 0.1`)
  so a dormant world still ages a little.  `WorldStatus.cognitive_tick`
  exposes the current tick; `world.clock.advance(n)` is the time
  simulator for tests and calibration.
- **Cognitive-dynamics analysis** (`docs/world0-cognitive-dynamics-analysis.md`)
  — mathematical review of decay, activation, projection, lifecycle and
  Hebbian learning with before/after probe evidence, parameter calibration
  and a roadmap toward a continuously updating cognitive layer; 36 new
  behavioral tests in `tests/test_dynamics_analysis.py`.
- **Task profile** — `ConceptNode.task_profile` aggregates activations per
  normalized task label; `task_affinity()` / `task_match_score()` provide
  graded, word-level task association (no more `"ml"` matching
  `"html parsing"`).  `reinforcement_log` is now a bounded recent-activity
  window (64 entries); legacy records back-fill the profile on load.
- **Hebbian state persistence** — pending co-occurrence counters are stored
  in `state.json` (`hebbian_pending`) and restored on start, so a pair seen
  once per session still reaches the discovery threshold.
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
- **Temporal dynamics run on cognitive time, not the calendar.**  All
  half-lives (`CONCEPT_HALF_LIFE`, `RELATION_BASE_HALF_LIFE`,
  `CONCEPT_TEMPORAL_HL`, `RELATION_TEMPORAL_HL`, `PROJECTION_TEMPORAL_HL`)
  keep their numeric values but are now measured in observations;
  `temporal_relevance(half_life, *, now_tick=, now=)` replaces the
  hours-based signature.  Tests simulate time with
  `world.clock.advance(n)` instead of rewinding `datetime` fields.
- **Decay is idempotent.**  Concepts and relations carry
  `last_decayed_tick` / `last_decayed_at`; only the interval since the last
  decay application is decayed, so `reflect()` frequency no longer changes
  the dynamics.
- **Evidence-anchored concept decay.**  Effective half-life stretches with
  activation count (`1 + 0.2·(n−1)`, capped at 8× and 8760 observations)
  and confidence relaxes toward an evidence floor (OU-style mean
  reversion) that itself forgets on a 4380-observation era scale.  A
  concept re-observed every 24 observations now reaches `established`
  after ~1400 observations (previously stuck at confidence ≈ 0.06
  forever); one re-observed every 168 settles as `developing` instead of
  fading; a one-off mention still fades after ~54 observations.
- **Activation aggregation.**  Contributions converging on one concept
  combine with a bounded noisy-OR scaled by the strongest seed (rewards
  conceptual intersection, never outranks a seed); propagation is layered
  with each concept expanded once at its first-reached depth (cycles cannot
  inflate scores, `record=True` counts once); the propagation floor is a
  rank-preserving band instead of a constant, so distance ordering is
  strict.
- **Deterministic projection.**  MMR visits candidates in `(score desc,
  id)` order, keeps selection order, sorts internal relations, and all
  freshness terms share one `now` per pass — identical output across
  processes regardless of `PYTHONHASHSEED`.
- **Hebbian pair cap** (`MAX_PAIRS`) now keeps pairs in observation
  (salience) order rather than lexicographic id order.

### Fixed
- Relation `probability` (belief the typed relation is correct) is no
  longer eroded by time decay, and `weaken()` lowers it by the
  disconfirmation penalty instead of overwriting it with `confidence`
  (which could *raise* it).

### Changed (earlier)
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
