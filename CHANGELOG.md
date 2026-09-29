# Changelog

All notable changes to World 0 are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Task context** (analysis doc §7.24, paper §7.3) — task words are weighted
  by how distinctive they are among the world's task labels
  (`TaskVocabulary`: `ln((L+1)/(df+1)) + 0.01`), so shared boilerplate
  ("… work", "fix … bug") no longer makes every task match every concept at
  ≥ 0.5.  A claim's context is the tasks it was *stated* under
  (`RelationEdge.claim_tasks`); when a concept in view has live claims in the
  current task's context, its claims stated only under other tasks move to
  `Projection.other_contexts` ("Seen in Other Tasks"), and the epistemic
  status still sees both.  LongRun bridge probe (12 queries): 0.44 → 0.93
  with the right task, unchanged without one.
- **Withdrawn claims** (analysis doc §7.25, paper §4.4) —
  `Observation.retracted_relations` withdraws a claim that no longer holds
  (`RelationEdge.retracted_tick`, `RelationManager.retract`): belief
  untouched, no activation, not a connection, not in views (listed under
  "No Longer Holds" for seeds), weight relaxes to zero, co-occurrence
  neither reinforces nor blocks the pair, restating restores it.
- **Read-time settlement** (paper Corollary 3.2′) — activation and
  projection read settled confidence and weights and treat dead edges and
  expired concepts as absent, so a view between two uses no longer depends
  on when a reflect ran (seed score 0.317 vs 0.489 before, 0.3165 both now).
- **LongRun evaluation** (`benchmarks/longrun/`, `docs/eval/`) — a reproducible
  comparison of World 0 with raw context (window, full history) and traditional
  memory (BM25 RAG, Generative-Agents recipe, summaries, a Mem0-style fact
  store, temporal / static graphs, a state document) in a long-running Agent:
  seeded hidden worlds with typed domain graphs, polysemous bridges, revisions
  and episodic tickets; gold restricted to what the Agent was told and is
  still true; extraction-error, task-label, chatter, verbosity, horizon,
  window-limit and cost studies; per-seed paired tests with Holm correction;
  World 0's depth / reflect chosen on dev seeds only; a real-LLM-reader stage
  (one reader per query and condition) after an independent five-lens review
  of the harness.  Findings (`docs/eval/01-report.md`, fourth version):
  raw context is the most reliable while it fits (real reader 1.00) and
  degrades with the window once the history does not (32k/64k/128k:
  0.67/0.77/0.81); with every task-using system on the same
  distinctiveness-weighted task matcher, per-task summaries (0.96),
  World 0 tuned (0.95, ~590 tokens), a state document (0.94) and a
  task-tagged fact store (0.93, ~190 tokens) are statistically tied and
  beat raw context, RAG (0.74) and a temporal graph (0.82).  They differ in
  what breaks them: summaries / state documents need the right task label
  (0.55 / 0.41 without one), World 0 is the most robust to missing or wrong
  labels (0.84 / 0.81) and the most sensitive to extraction error (−39 % at
  30 %); World 0 costs ~3× the fact store's tokens and ~500× its write
  time.  The shipped `render()` remains a poor prompt format (0.57 vs 0.90
  compact).  The real-LLM-reader stage predates the fixes below and is to
  be re-run.
- **Formal paper** (`docs/paper/world0-formal.md`) — the current design
  written as a mathematical model with propositions and proofs: the
  exact semigroup property of the moving-floor decay, reflect-cadence
  independence, the six-confirmation noise threshold, relation survival
  `E·log2(5p)`, Jaccard as a function of the two conditionals (gate 0.2 ⇔
  P ≥ 1/3), seed dominance and horizon completeness of activation, the
  off-task-as-duplicate lemma of the projection, focus capacity and
  lifetime, and the calibration of the prediction error.
  `docs/paper/verify.py` checks every proposition numerically against the
  engines; `tests/test_formal_properties.py` (19 tests) pins them.
- **Prediction error on ingest** (machine-consciousness track,
  `docs/mc/04-prediction.md`) — every observation is scored against
  learned co-occurrence before it is learned: `IngestResult.prediction`
  (`PredictionError`) lists strong companions that failed to appear and
  well-known concepts meeting for the first time.  Both errors are
  relative to the model's own expectation (absences beyond `Σ p(1−p)`,
  new pairs weighted by `1 − exp(−n_a·n_b/N)`), so a loose cluster's
  ordinary variability is not surprise: structured world 0.11 vs random
  0.26, and a concept drift raises the error from ≈0.01 to 0.51 before it
  settles.  `HebbianEngine` keeps co-occurrence counts for linked pairs
  (persisted with the learning state) to support the predictions.
- **Sustained focus and attention schema** (machine-consciousness track,
  `docs/mc/03-workspace.md`) — `World(sustained_attention=True)` keeps a
  limited-capacity `world0.context.Focus` across projections: selected
  concepts that ignite (seeds, or relevance ≥ half the view's strongest
  non-seed) enter at full strength, at most 7 are kept, strengths decay by
  0.6 per later projection and the focus clears on a new explicit task.
  The next view boosts candidates its own seeds reached that are in or
  next to the focus (seeds pass no focus to their neighbours), so a
  bridge concept follows the current line of attention while unrelated
  views are untouched.  Every projection now carries
  `Projection.attention` (`AttentionTrace`: seed / reached, via which
  concept and relation, task-named, task history, sustained, in focus,
  ignited), rendered as `### Why These Concepts`.  Off by default:
  projections stay a pure function of the world.
- **Metacognitive monitoring in projections** (machine-consciousness track,
  `docs/mc/`) — `Projection.epistemic` (`EpistemicStatus`) grades every
  projected concept by its evidence (`tentative` / `moderate` /
  `well_evidenced`) and reports opposing explicit claims about a pair
  with their beliefs as `contested` (margin < 0.25) or `leaning`
  (`world0.projection.metacognition.assess`).  The render shows the belief
  of every explicit claim, marks co-occurrence edges as such, and adds an
  `### Epistemic Status` section.  Projection selection is unchanged.
- **`docs/mc/`** — machine-consciousness research records: indicator
  properties from consciousness science (Butlin et al. 2023) mapped to
  World 0 with probe evidence and a roadmap; functional properties only,
  no claim of experience.
- **Directed and contradictory relation claims** — directed relations are
  now matched in their stated orientation: `X depends_on Y` and
  `Y depends_on X` are two edges, and the reverse claim no longer confirms
  the forward one.  Symmetric semantics on a directed axis (`conflict`,
  `disjointness`, `complement`, `incompatible_ontology`, `co_creation`,
  `mutual_reinforcement`, `future_coupling`) match either orientation and
  are not scaled by direction-conditioned perspectives.  An explicit claim
  on the negative axis disconfirms an explicit positive / parallel claim
  about the same pair and vice versa (reported in
  `IngestResult.weakened_relations`), so contradictory beliefs compete
  instead of both staying confident.  `RelationEdge.connects()`,
  `RelationEdge.opposes()`, `find_between(..., directed=)` (docs §7.18).
- **Task grounding** (`world0.context`) — a task now changes the projection
  through the concepts it *names*, not only through task labels recorded
  on past observations.  `name_coverage()` measures word-level coverage of
  a concept's name or aliases by the task (CJK names fall back to
  containment); `ground_task()` gives named concepts affinity 1.0, their
  direct neighbours 0.5 and partially named concepts their coverage
  (≥ 0.5).  Projection uses `max(history, grounding)`.  Previously a world
  built from unlabelled observations ignored every task string, and a new
  task such as `"kubernetes rollout"` changed nothing (docs §7.17).
- **Recurrence-based promotion** — `ConceptNode.recurrence_count` counts the
  distinct 24-tick windows in which a concept was activated (a burst counts
  once).  Lifecycle promotion accepts either the confidence gate or a
  recurrence gate (`≥3` windows for developing, `≥10` for established), so
  a concept re-observed every ~168 observations now reaches `established`;
  a faded concept revives to DEVELOPING only with `≥3` recurrences,
  otherwise it re-enters as EMBRYONIC.
- **`RelationEdge.confirm()`** — an explicit re-statement of a typed
  relation moves its semantic `probability` toward 1 with diminishing
  returns; the ingest pipeline calls it alongside `reinforce()` for
  explicit re-observations (Hebbian co-occurrence still only reinforces).
  `RelationManager.adjust_strength()` now moves `probability` by the
  confidence delta instead of overwriting it.
- **Directional propagation** — `Perspective.direction_weights`
  (`forward` / `backward`) scale activation along or against a relation's
  direction; defaults are neutral.
- **Light / automatic reflect** — `World.reflect(light=True)` runs decay,
  lifecycle and pruning only; `World(..., auto_reflect_every=N)` schedules
  it every N observations.
- **Cognitive clock** (`schemas/clock.py`) — time in World 0 is counted in
  observations: `World.clock` advances once per `ingest()` and is persisted
  in `state.json` (`tick`); every concept/relation carries `*_tick`
  coordinates next to its timestamps, and decay, freshness, evidence
  floors and community coupling are functions of ticks elapsed.  The wall
  clock survives only as a slow drift term (`WALL_TICKS_PER_HOUR = 0.1`)
  so a dormant world still ages a little.  `WorldStatus.cognitive_tick`
  exposes the current tick; `world.clock.advance(n)` is the time
  simulator for tests and calibration.
- **Perspective profiles** (`world0.perspectives`) — named, documented
  reading strategies over the same relations: `dependency_map` (what a
  concept relies on), `impact_map` (what relies on it), `taxonomy`,
  `analogy`, `contrast`, `default`.  `World.project(seeds,
  perspective="taxonomy", task=…)` accepts a profile name.
- **Projection stability tests** (`tests/test_projection_stability.py`)
  — unrelated observations, reflect cadence and alternating mentions
  must leave a projection identical; an in-domain re-mention may only
  reorder an exact tie.
- **Layer-boundary test** (`tests/test_layer_boundaries.py`) — walks the
  imports of every core module and fails if the conceptual core reaches
  into `agents`, `llm`, `extraction` or the presentation packages, or if
  anything outside `agents` imports it (AGENTS.md Rule 6).
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
- **Synonym resolution is indexed** — `ConceptManager._find_synonym_match`
  scores only a shortlist (exact-label hits plus the postings of the
  rarest probe tokens) with cached per-concept signatures instead of
  re-tokenizing every concept; decisions are unchanged.  Building 1 000
  semantic concepts drops from 16.8 s to 0.14 s; 5 000 take 2.4 s.
- **Relative activation cut** — activation and projection accept
  candidates above `min(min_activation, 0.02 × strongest seed)`, so weak
  (embryonic) seeds keep the same multi-hop horizon as confident ones.
- **Evidence and salience accessors** — `ConceptNode.evidence()` (Beta
  posterior × count saturation, time-independent) and
  `ConceptNode.salience()` (cognitive-time freshness); `Projection.render()`
  prints evidence next to confidence.
- **MMR λ 0.3 → 0.5** — calibrated on a redundancy-sensitive scenario (six
  near-identical siblings vs a three-concept chain): at 0.3 the projection
  spent half its slots on siblings, at 0.5 it covers both regions; the
  cognitive benchmark is unchanged.
- **Numeric identity tokens** — `tokenize_signature` keeps purely numeric
  tokens of any length, so "GPT 4" and "GPT 5" are no longer signature
  twins that consolidation merges.
- **SQLite storage backend** — `world0.store.SqliteStore` (single WAL
  file, one transaction per flush, same JSON payloads as the file
  backend).  `World(store_path, backend="auto"|"json"|"sqlite")`; `auto`
  picks SQLite for `.sqlite` / `.sqlite3` / `.db` paths and JSON-per-file
  otherwise, so existing stores are unaffected.  Flushing 60 × 40 dirty
  concepts: 0.59 s (JSON) → 0.23 s (SQLite).
- **Salience is a real quantity; time is charged once per hop.**
  `ConceptNode.salience()` is now freshness *or* evidence-backed
  persistence (`0.7 × evidence()`, forgotten on the same 4380-observation
  era as the confidence floor), whichever is larger; activation and
  projection read it instead of raw `temporal_relevance()`.  Propagation
  readiness is `max(confidence, evidence(), 0.3)` instead of
  `max(confidence, 0.3)`.  A dependency confirmed fifty times then dormant
  scored ~0.2× a same-observation one-off and never entered a
  slot-limited projection; it now holds a top slot for ~500 observations,
  reaches parity near 1 000 and yields to fresh context after that.  The
  cognitive benchmark is unchanged; one-off concepts (evidence ≈ 0.06)
  still collapse to the freshness floor.  Sweep: `scripts/sweep_salience.py`.
- **Prune grace period: fade fast, delete slowly.**  A FADING concept
  is deleted only after `PRUNE_MIN_IDLE_TICKS` (720) idle observations.
  Previously a concept mentioned once was deleted unless re-mentioned
  within ~80 observations and the second mention started a fresh node
  from zero; in a 100-topic world with periodic light reflects 1 937
  created concepts shrank to 797.  Fading (reversible) is unchanged.
- **Learning-state persistence is amortised.**  Hebbian co-occurrence
  counters and mention statistics now live in a separate store record
  (`learning.json` / the `learning` row) written on every observation
  while small and at most every 20 observations once large, plus at
  `reflect()` and the new `World.close()` (`with World(...) as w`).
  Serialising them into `state.json` on every observation was 76 % of
  ingest cost in a 2 000-concept world.  Older stores with inline
  counters are migrated on first open; concepts and relations are still
  flushed on every observation.
  `PKMAgent.close()` persists it; the CLI calls it on exit and the web
  app on shutdown.
- **Explicit relations relax toward a probability-anchored floor.**
  Relation `weight` / `confidence` decay toward `0.1 × probability`
  (forgotten on the 4380-observation era scale) instead of toward 0, so
  an explicitly stated relation is no longer pruned after ~1 000 idle
  observations while its never-decayed `probability` was still 0.70; it
  now persists ~7 900 observations without re-statement, and
  `confirm()` raises the floor.  Auto-discovered edges get no floor.
- **Seeds are always part of their projection.**  `World.project()`
  passes the seed ids to the projection engine, which selects them
  first (by score, capped by `max_concepts`) and exempts them from the
  activation cut before MMR fills the remaining slots.  Previously a
  cross-domain second seed under a task, or several seeds from one
  cluster, could be displaced by a "more diverse" neighbour.
- **Task-aware projection redundancy.**  In MMR an off-task candidate
  is now as redundant as a duplicate (`redundancy = max(sim, 1 −
  affinity)`) while better-matching candidates remain, so a dense
  on-task cluster — whose members share one neighbourhood and look
  fully redundant with each other — no longer loses slots to an
  unrelated cluster that merely looks diverse.  After a concept drifts
  from data-engineering to ML usage, `project(task="data eng")` showed
  the ML concept `checkpoint` in second place; it now returns the
  data-engineering neighbourhood only.  The cognitive benchmark rises
  from ML 0.67/0.67, Ops 0.83/0.83 to 1.00/1.00 on both.  Without a task
  nothing changes.
- **Hebbian discovery is gated on association strength.**  Besides
  co-occurring at least twice, a pair is linked only when its Jaccard
  association over observations (`co-occurrences / (mentions_a +
  mentions_b − co-occurrences)`) is at least `HEBBIAN_MIN_ASSOCIATION`
  (0.2).  With the count gate alone, 60 concepts observed six at a time
  linked 85 % of all pairs with untyped `generic_relation` edges after
  400 random observations and buried the explicit backbone; the gate
  keeps that at 9 % while every pair drawn from one topic stays linked
  and the benchmark is unchanged.  Observation and mention counts are
  persisted as `hebbian_stats`; stores without it start counting from
  zero.  Sweep: `scripts/sweep_hebbian.py`.
  `reflect()` additionally revalidates auto-discovered generic edges
  against the accumulated statistics (`HebbianEngine.revalidate()`,
  hysteresis 0.5, judged once both concepts total ≥ 20 mentions) and
  reports removals in `ReflectResult.stale_relations`; edges linked on
  thin early statistics no longer survive on chance reinforcement
  (151 → 10 in the random world after one reflect).
- **Perspectives condition propagation at the semantic level.**
  `Perspective.relation_type_weights` accepts semantic relation names and
  aliases (`dependence` / `depends_on`, `inclusion` / `contains`, …) as
  well as axes; a semantic key wins over its axis key, and unknown keys
  are rejected at construction instead of being silently ignored.
  Direction weights now apply only to directed (positive / negative)
  relations — a parallel / Hebbian edge's stored orientation is
  arbitrary, so scaling it merely rescaled every neighbor and left the
  projection unchanged.  `RelationEdge.is_directed` exposes the rule.

### Fixed
- **Maturity no longer depends on how often `reflect` runs** (`docs/paper/world0-formal.md`
  §3.5, F6, Theorem 3.7).  Promotion gates now contain only quantities that
  become true at events (confirmations, disconfirmations, spaced recurrence,
  live connections) and are evaluated at those events (activation, relation
  create/reinforce); `reflect` only catches up out-of-band edits.  Settling is
  the exact solution of the moving-floor relaxation, so it is an exact
  semigroup and the fading crossing is exact.  A concept used every 720 ticks
  now reaches ESTABLISHED at its 12th use (it needed 324 uses / 232,561
  observations), and the same stream ends in the same (confidence, maturity,
  existence) under any reflect schedule (spread ≤ 3e-9; before: core /
  established / developing / embryonic).  Projection ties are broken on
  scores quantised to 1e-6 of the strongest activation (recency, name, id)
  so equal concepts no longer depend on wall-clock noise.  The claim that a
  sparse cadence reaches ESTABLISHED at the 12th use holds up to the prune
  grace (720 observations); beyond it the concept is forgotten as noise
  between uses, in every world (see the existence entry below).
- **Existence and liveness no longer depend on how often `reflect` runs**
  (paper F7, Definition 3.6, Theorem 3.7; found by independent review).  A
  concept or relation that is already dead (a concept: fading, below 0.02
  and idle ≥ 720; a relation: settled weight below 0.02) is now treated as
  absent at every event — `IngestPipeline` deletes an expired concept (with
  its relations) before a mention starts a fresh one, `reap_dead_between`
  does the same for dead edges on restatement and co-occurrence — so
  physical deletion at reflect is garbage collection.  Before, a
  never-reflecting world revived the old node (n accumulating) and a
  reflecting world created a new one: a concept used every 800 observations
  reached ESTABLISHED at the 12th use in one and never in the other, and
  five one-shot partners kept an edge count that made a concept CORE
  (0.804) where the reflecting world had ESTABLISHED (0.790).  The CORE gate
  counts only edges above the prune line whose other end is not expired.
  Relation settling is now the same exact moving-floor solution as
  concepts (the cadence-dependent survival 8 070–9 000 became one value),
  and every relation event settles first (`weaken`, restatement with a
  prior, `adjust_strength`).  Other leaks closed: restating an edge through
  `relation_priors` and `adjust_strength` now fire the CORE evaluation,
  `ActivationEngine.activate(record=True)` and positive `adjust_confidence`
  fire the promotion hook, recurrences are counted at least 24 ticks
  apart instead of on a fixed grid (a 26-tick burst is no longer three
  windows), a revived concept lands at the fading line instead of below it,
  a negative claim's inhibition gain is not overwritten by an extractor
  prior at creation (0.176 → 0.776 with prior 0.7 before),
  `split_concept` stamps the new node with the current tick (it was pruned
  by the next reflect), the projection's score quantum is relative to the
  strongest activation (faded seeds kept their ranking), and the lifecycle
  hooks call through `World._lifecycle` so a swapped policy keeps sole
  authority.  340 random streams end in the same state under reflect
  every 1 / 50 / 1 000 observations and never
  (`tests/test_schedule_independence.py`, `docs/paper/verify.py`).
  Side effect: the shipped `Projection.render()` prints more for a world
  that never reflects — concepts now reach CORE/ESTABLISHED at the event,
  and the "Core Understanding" section carries descriptions and link lists
  (a 200-concept, 3 000-observation world: 1 380 → 3 180 tokens per
  15-concept view; the compact renderer is unaffected).
- **A stated negative claim carries a belief, not an inhibition gain**
  (paper §4.2 note, analysis doc §7.21).  `RelationManager.discover` seeded an
  explicit edge's `probability` from `propagation_strength`, which on the
  negative axis is the gain of the inhibition channel (0.05-0.12).  A once-
  stated `conflict` therefore started at belief 0.10, its relation floor
  (`0.1 × probability`) sat under the 0.02 prune threshold, and with both
  concepts kept alive it was pruned after 450 idle observations
  (`disjointness` 350) against 8 100 for a `depends_on` stated the same way;
  in a contested pair it started 0.76 vs 0.10, so `assess()` called one
  `enables` plus one `conflict` "leaning" (0.705 vs 0.100).  The default
  belief of a negative-axis claim is now `NEGATIVE_CLAIM_PRIOR` = 0.70 (what
  a once-stated `dependence` carries), via `SemanticRelationSpec.claim_prior`
  (positive / parallel unchanged, bit for bit); the inhibition gain
  (`weight`) and activation are untouched.  Stated-once negative claims now
  live 8 100 observations like a dependence (restated x5: 9 250), a contested
  pair started at equal belief is symmetric under swapping the axes
  (`dependence x10` then `conflict x10`: 0.410 / 0.811, mirror 0.811 / 0.410;
  `enables` starts at 0.76, so its pair with `conflict` is not an exact
  mirror: 0.447 / 0.811 and 0.849 / 0.410; before 0.447 / 0.433 and
  0.849 / 0.032), and beliefs below 0.2 (an
  extractor's "probably not", a claim disconfirmed that far) are still
  forgotten.  Stores written earlier are migrated once on load
  (`RelationEdge.adopt_claim_prior`, new field `belief_prior`; a belief that
  the legacy seed and the edge's own counters do not explain is kept).
  Side effects: negative claims now count by belief in the network entropy
  (toy world: negative mass 0.09 → 0.84); an extractor prior of exactly 0.3
  is no longer replaced by the label default (nor, on reload, by the
  structural confidence); restating a negative edge
  through `relation_priors` no longer writes the belief into its inhibition
  gain.
- **Forgetting no longer depends on how often `reflect()` runs** (paper
  Theorem 3.2).  Activation moved the decay reference point to "now" and
  the decay owed for the interval before it was dropped: a concept used
  every 24 observations ended at confidence 0.31 when reflect ran every
  observation and 0.93 when it ran every 1 000 or never.  The owed decay
  is now settled right before every reinforcement
  (`dynamics.decay.settle_concept` / `settle_relation`); all four
  cadences give 0.31.
- A faded or inhibited **seed is kept** in the activation result, so the
  seeds-first rule of the projection holds for it too.
- **Opposition between claims is symmetric**: a `generic_relation` claim
  no longer weakens an explicit negative claim (it asserts nothing, in
  either direction).
- Hebbian **revalidation uses the exact co-occurrence count** when it is
  tracked; the `reinforcements + 2` estimate undercounted links whose
  creation the association gate had delayed (8 true vs 4 estimated).
- The **prediction error counts each companion once**, however many
  relations (e.g. opposing explicit claims) link the pair.
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
