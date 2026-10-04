# Changelog

All notable changes to World 0 are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Long-term memory: a slow forgetting curve past a consolidation gate**
  (paper §3.6, analysis doc §7.34) — a concept that has recurred in at
  least 5 spaced windows (uses ≥ 24 observations apart; a burst does not
  count), is well evidenced (`evidence() ≥ 0.5`, i.e. ≥ 12 uses with none
  against) and is uncontested (balance ≥ 0.8) enters long-term memory at
  that activation (`ConceptNode.consolidated_tick`, `ConceptNode.long_term`,
  `dynamics/lifecycle.consolidation_gate`).  It then relaxes with a
  half-life of 35 040 observations whatever its maturity, and its
  evidence floor and salience persistence forget on the same era
  (`LONG_TERM_HALF_LIFE`, `LONG_TERM_ERA_HL`; `relax_confidence` gained an
  `era` argument).  Dropped after twelve spaced uses it stays above the
  FADING line for ~145 000 observations instead of ~11 000; it is slow,
  not immortal.  A disconfirmation that takes the balance below 0.8
  (`on_weaken`) or a fade ends the mode; a revived concept re-earns it.
  The gate is made of event-time counters, so it turns true only at an
  activation and the curve in force during a gap is fixed at the gap's
  start: the settle operator stays an exact semigroup and the state stays
  a function of the observation stream (Prop. 3.8; `verify.py §3.6`).
  `World(long_term_memory=None|True|False)` (default on;
  `WORLD0_LONG_TERM_MEMORY=0` changes the default), `IngestResult` /
  `ReflectResult.consolidated_concepts`, `ConceptCard.long_term`,
  `WorldStatus.long_term_concepts`, "long-term" in the full render.  A
  claim between two long-term concepts (`RelationEdge.long_term`, written
  at endpoint events after settling the edge under its old profile) keeps
  its weight floor on the long era too: a p = 0.7 claim dies after ~70 000
  idle observations instead of ~8 000, so a concept kept in reach keeps
  its claims.
  `tests/test_long_term_memory.py`.  LongRun is unchanged with the mode on
  or off (`main` 0.986 / 0.986, bigworld 0.871 / 0.870): in the 40×40 world
  481 of 1 618 concepts consolidate, but nothing on the ESTABLISHED curve
  dies within a 6 000-event horizon anyway, and the concepts the bigworld
  gap loses are the 811 mentioned once (noise by design).  The mode
  matters past ~10 000 idle observations, where the slow curve keeps a
  consolidated concept in reach for another ~130 000.
- **`world0.api`: the interface data shapes** (`docs/world0-api.md`, 0.4
  step of the roadmap) — `Statement`, `ConceptCardInput`, `ConceptCard`,
  `Claim` and `API_VERSION` (`"world0/1"`), exported from `world0`.
  `ConceptNode.to_card()` and `RelationEdge.to_claim()` produce them.
  `Observation` accepts `statements` / `withdrawals` / `denials` / `cards`
  next to the pipeline fields (`relations`, `relation_priors`,
  `retracted_relations`, `contradicted_relations`, `concept_candidates`),
  folds them in at construction and reads them back as properties; a
  statement mentions its endpoints, a withdrawal or denial does not.
  `Projection` gained the structured form of what `render()` prints:
  `cards`, `claims`, `no_longer_holds`, `other_tasks`, `hold_loosely`
  (claims with `status` contested / outvoted / doubted), `why` (one line
  per concept), `seeds`, `perspective` and `api`; the render's
  classification moved into `Projection._classify()` so text and
  structure cannot drift apart.  `World.card()`, `World.claims(task=)` and
  `World.find()` are the zero- and one-hop reads; `World.state()` /
  `withdraw()` / `deny()` are spellings of `ingest`.  `IngestResult`,
  `ReflectResult` and `WorldStatus` carry `api`.  The pipeline field
  names stay accepted without warning until the internals read the new
  ones (0.5); `tests/test_api_contract.py`.
- **The unified API's entry points** (`docs/world0-api.md` §5, 0.5 step) —
  one operation table, `world0.ops` (`call(world, op, params)`: request
  model, JSON result and error code per operation; errors are
  `not_found`, `invalid_observation`, `unknown_relation`,
  `identity_conflict`, `llm_unavailable`, `invalid_request`,
  `unknown_operation`, `internal_error`), behind three surfaces that cannot drift from it:
  the `world0` command (`world0.cli`; `world0 project api --task backend`,
  `world0 ingest obs.json`, `world0 state api depends_on db`; `--json`
  prints the HTTP body), HTTP `/v1/<op>` (`world0.http`: `v1_router(world)`
  is mounted in the Agent shell's web app next to `/api/*`, `create_app`
  serves it alone), and World 0 as an MCP server
  (`python -m world0.agents.mcp.server --store .world0`: tools
  `world0_<op>` over stdio, input schemas from the request models, a
  projection's first text content is the render, the JSON follows, and
  every result is `structuredContent`; malformed frames and tool errors
  are answered, never fatal).  HTTP requests run under one lock and a
  non-object body is `invalid_request`.  `McpClient.call_tool_raw()` returns the MCP
  result object.  The wire form of `project` carries `text` and
  activation scores quantised to 1e-6.  `tests/test_api_surfaces.py` reads
  the same world through all four entry points and asserts identical JSON.

## [0.3.0] - 2026-10-01

**A cognitive layer whose state is a function of the observation stream.**
Since 0.2.0 the chain concept → card → typed relation → task context →
activation → projection was reworked in 30 rounds (analysis doc §7, paper,
LongRun benchmark).  The headline properties of this release:

- *Schedule independence*: maturity, existence, forgetting, confidence and
  the set of co-occurrence edges no longer depend on when `reflect()` runs
  (paper Theorems 3.2 and 3.7, §5 Prop. 5.4; `docs/paper/verify.py` checks
  every proposition against the code).
- *Views are task-conditioned and honest*: claims are split by the task that
  stated them, a view is never filled with another task's concepts, a
  retracted claim leaves the view, a stated conflict brings its partner into
  it, and the default render is the compact prompt-ready form.
- *Relations are typed and say what was stated*: claim identity is
  (source, target, axis, label), negative claims carry belief, `contrast`
  is a first-class relation and the bare negative-axis words map to it,
  the weakest negative relation, not to `conflict`.
- *Evidence*: a formal model with proofs and a numerical verifier, and the
  reproducible LongRun benchmark (10 seeds, real-LLM extraction, a
  real-LLM-reader stage) in which `world0_tuned` ties well-configured
  structured memory and beats raw context at 200–600 tokens
  (`docs/eval/01-report.md`).

### Migration from 0.2.0

- `Projection.render()` now returns the compact prompt form; pass
  `render(style="full")` for the diagnostic view with ids, maturity and
  link lists.
- Relation labels: 0.2.0 stored the label itself (`depends_on`, `part_of`,
  `contrasts`, ...) in `relation_type`; 0.3.0 stores a three-axis
  `relation_type` plus a `semantic_relation`.  A 0.2.0 store is migrated
  on load: each label keeps its semantics (`depends_on` → `dependence`,
  `supports` / `activates` → `enables`, `contrasts` → `contrast`,
  `similar_to` → `similarity_kernel`, `related_to` → `generic_relation`),
  and a label phrased from the other end is stored in the canonical
  direction: `(wheel, car, "part_of")` becomes `inclusion` from `car`
  ("car contains wheel"), `(build, deploy, "precedes")` becomes
  `dependence` from `deploy`.  The same holds for new input: `part_of` is
  inclusion read from the part.
- `contrasts` (and the bare axis words `negative` / `repulsion`) now mean
  `contrast` — "differ in a way worth keeping apart" — not `conflict`; say
  `conflict` when you mean it.  `mutual_understanding` maps to
  `recursive_co_modeling`.
- A negative claim's belief is seeded at 0.70, like a positive one; a
  0.2.0 / early-dev edge whose belief was the inhibition gain is rebased
  once on load.  Learning-state counters stored inline by early dev
  versions are moved to their own record on first open.
- `PROPAGATION_MIN_RATIO` lives in `world0.dynamics.coefficients`.
- The optional SQLite backend (`World(store_path, backend="sqlite")`) is not
  the default; `backend="auto"` picks it only for `.sqlite` / `.sqlite3` /
  `.db` paths.  JSON stores open unchanged.
- New since 0.2.0, for callers that adapt extraction output:
  `Observation.retracted_relations` withdraws a claim that no longer holds
  (it leaves views and is listed under "No longer holds");
  `Observation.contradicted_relations` only lowers a claim's belief, and a
  denial of a claim nobody made changes nothing.  `IngestResult` gained
  `retracted_relations`, `stale_relations` and `prediction`;
  `ReflectResult` gained `stale_relations`.  `world0.__version__` is new.

### Changed
- **A 0.2.0 store keeps what its relations said** — 0.2.0 stored the label
  itself (`depends_on`, `part_of`, `contrasts`, ...) in `relation_type`;
  opening such a store used to collapse every label to its axis default
  (`depends_on` and `part_of` both became `mutual_reinforcement`,
  `contrasts` became `conflict`).  `RelationEdge` now migrates a legacy
  label on load: the semantics are kept and a label phrased from the other
  end (`part_of`, `precedes`) is stored in the canonical direction, as a
  stated claim is.  Found by the release review;
  `tests/test_legacy_store_relations.py`.
- **Evaluation refresh** (analysis doc §7.33, LongRun report v5) — the
  real-LLM-reader stage was rerun on the release code with Claude Haiku 4.5
  readers (132 questions × 7 conditions, 4 seeds, no cross-world leakage):
  `world0_tuned`@1200 0.95 [0.92, 0.97], `state_doc` 0.93, `fact_task`
  0.91, `rag` 0.80, `full_context` 0.74 — the raw-history reader misses
  the correction on 17 % of revision questions, the structured systems on
  none; no answer is a concept id any more (54–59 % before compact render
  became the default).  Template text (seed 0) was re-extracted with the
  shipped prompt (0.968 vs 0.994 with the old one: a single sentence
  missed at step 399); the natural-text runs on three seeds cost every
  structured system 0.03–0.05 against generator extraction, World 0 the
  most (0.996 → 0.952).  All LongRun studies were rerun on the release
  candidate (`docs/eval/results/*.meta.json`): at 1 200 tokens
  `world0_tuned` scores 0.99 (fourth version 0.95), +0.05 over `fact_task`
  (10/10 seeds, Holm p = 0.027) and +0.04 / +0.03 over `state_doc` /
  `summary_task`; at 600 tokens the three are still tied.  `readers.py`
  gained the `world0_tuned` conditions at 600 and 1 200 tokens.
- **Hebbian revalidation happens at the event** (analysis doc §7.32, paper
  §3.5 / §5.3) — a co-occurrence edge's association changes only when one
  of its endpoints is mentioned, so `HebbianEngine.learn()` now re-judges
  exactly the generic edges incident to the concepts it just counted and
  removes those below the gate; `IngestResult.stale_relations` reports them.
  Reflect's whole-store `revalidate()` stays as a backstop and finds nothing
  after an ordinary stream.  The set of generic edges is therefore a
  function of the observation stream alone, and Theorem 3.7 (state does not
  depend on the reflect cadence) no longer carries its last premise.
- **Floor-band candidates fill a projection only after the candidates
  above the band** (analysis doc §7.31) — a candidate the activation engine
  lifted into the floor band carries no evidential strength to trade
  against diversity, so MMR runs over the candidates above the band and the
  band fills what is left in activation order.  Before, the redundancy term
  dominated the band's ~1 % relevance differences and a far third-hop
  concept displaced a strongly reached second-hop one.  LongRun bigworld
  utility 0.865 → 0.871, wrong-label 0.807 → 0.812; `main` unchanged.
- **A stated contrast makes its partner visible, as a terminal** (analysis
  doc §7.30, paper Prop. 6.6) — "A conflicts with B" is knowledge about the
  pair, so B now appears in A's view at the strength of the contrast
  (`max(visibility, excitation − inhibition)`), while activation does not
  spread on through B and B's excitation through other paths is still
  inhibited.  Before, the partner of a contrast was only ever *suppressed*:
  on LongRun focus queries 16 % of gold conflict claims were missing because
  their adjacent partner was not selected.  Conflict recall 0.84 → 0.99;
  `main` utility 0.956 → 0.987 (focus 0.92 → 0.99, bridge 0.95 → 1.00).
- **Relation semantics: `contrast`, honest axis-word aliases, `part_of` is
  inclusion** — new semantic relation `contrast` (negative axis, seeded at
  0.70 / 0.06, symmetric: "differ in a way worth keeping apart, without
  conflicting"); the bare axis words `contrasts` / `negative` / `repulsion`
  now map to it instead of to `conflict` (a bare axis word asserts only the
  axis: the weakest claim on it), `mutual_understanding` maps to
  `recursive_co_modeling`, and `part_of` is `inclusion` read from the part —
  `(wheel, car, "part_of")` is stored and rendered as "car contains wheel"
  (before it was stored as `membership` in the wrong direction).  The
  extraction prompt lists `contrast` under the negative labels.
- **A task-conditioned projection is not filled with another task's
  concepts** (analysis doc §7.29) — `max_concepts` is a ceiling, not a
  target.  A candidate whose known context is another task (task-profile
  affinity for the current task below `CONTEXT_MATCH`) and whose activation
  sits in the floor band (kept for horizon completeness, not on the strength
  of its evidence) is filler and never selected; off-task concepts the seeds
  reach strongly still compete on merit, and with no task or no candidate in
  the task's context nothing changes.  LongRun focus precision 0.57 → 0.63 at
  the same utility (0.956), 17 % fewer tokens; wrong task label 0.71 → 0.77.
  `PROPAGATION_MIN_RATIO` moved to `dynamics.coefficients`.
- **Extraction prompt: direction convention and label scope** (analysis doc
  §7.28) — every relation reads as "<source> <label> <target>"; direction
  follows what the text asserts, not word order, with examples for passive,
  conditional and part-of phrasings; a "needs" statement is dependence with
  the needing side as source, never a reversed enables; membership,
  functional_map, disjointness, incompatible_ontology and conflict are
  scoped; a denial goes in `contradicted_relations`.  On paraphrased LongRun
  text the old prompt kept 50 % of claims (37 % reversed, 13 % relabelled);
  the new one keeps 90 / 88 / 93 % on seeds 0 / 1 / 2 (new events for seeds 1
  and 2, but the same paraphrase phrasings).
- **A denial of a claim nobody made weakens nothing** — a
  `contradicted_relations` entry with no matching claim used to weaken both
  endpoint concepts; a denial is about the relation, not about the concepts.

### Added
- **Natural (paraphrased) LongRun text** (`benchmarks/longrun/paraphrase.py`,
  `GenConfig(text_style="natural")`) — the same stream in varied wording:
  active and passive phrasings, pronouns, case and articles, varied
  withdrawals, negated distractors.  Events and gold sets are identical to
  the template text of the same seed.  Real LLM extraction is cached for
  seeds 0–2 (`llm_extract.py --style natural`); at 1 200 tokens, excluding
  episodic detail queries, real extraction costs `world0_tuned` 0.94 → 0.90,
  `fact_task` 0.92 → 0.87, `state_doc` 0.95 → 0.90.  A negated pair is never a
  claim of the stream (any domain, any time, including later revisions).  `llm_cache/prompt_v1/`
  keeps the previous prompt's extractions (`GenConfig(llm_version=...)`).
- **Withdrawn relations from the LLM extractor** (analysis doc §7.27) — the
  extraction prompt and `ConceptExtractor` emit `retracted_relations`
  (a relation that held and no longer holds: removed, replaced, changed),
  distinct from `contradicted_relations` (it is or was wrong).  Before, the
  withdrawal mechanism of round 23 could never be triggered by real
  extraction.
- **LongRun with real LLM extraction** (`benchmarks/longrun/llm_extract.py`,
  `GenConfig(extract_mode="llm")`, `run.py --study llm`) — every event's text
  through the production prompt and parser, raw answers cached per
  (seed, horizon); `profile` measures the extractor against gold.  Seed 0
  (600 events, Claude Haiku as the extractor): 959/959 claims and 6/6
  withdrawals right; every system scores as with gold extraction.
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
- **A claim is (source, target, label)** (analysis doc §7.27) — a second
  label on the same pair and axis is a second claim instead of overwriting
  the first (one misread "contains" replaced a "depends on" stated many
  times; a revision can make both true).  Withdrawal, contradiction and
  merge match by label.  The compact render weighs support
  (`RelationEdge.support`): a claim stated `OUTVOTE_RATIO` (2) times less
  often than an opposing claim or another label for the same pair is
  listed under "Hold loosely" ("outvoted" / "also stated as"), not as
  current; "Hold loosely" is capped at ten lines.  Activation takes one
  channel per neighbour, axis and direction (the strongest claim), and the
  CORE gate counts distinct neighbours.  LongRun at 1 200 tokens:
  wrong labels only (p = 0.3) 0.52 → 0.65; mixed extraction error p = 0.1 /
  0.3: 0.79 / 0.53 → 0.82 / 0.60; main study `world0_tuned` 0.95 → 0.96.
- **Compact render is the default** (analysis doc §7.26) —
  `Projection.render()` now returns the prompt form LongRun's readers did
  best with (0.90 vs 0.57 for the full render at the same budget): current
  claims as sentences with belief, strongest first
  (`api depends on db (belief 0.82)`), other concepts in view, concept-card
  definitions, and labelled sections for what to discount — withdrawn
  claims, claims made under other tasks, contested claims (the leader
  against its opponents), claims disconfirmed below even odds, thin
  evidence.  Homonyms (in view or endpoints of withdrawn claims) carry their
  sense; user text is kept to one line.  `render(style="full")` keeps the
  diagnostic view; the PKM agent's answer prompt describes the new form.
  `RELATION_PHRASES` / `relation_phrase()` phrase every semantic relation.
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
- **A typed claim stated on a co-occurrence edge** (e.g. `similar_to` on a
  pair already linked by co-occurrence) relabelled the edge but left it a
  co-occurrence edge, so the claim never showed; the co-occurrence edge now
  gives way to a fresh claim (the same state whether co-occurrence came
  first or not).  Withdrawal and contradiction never take a co-occurrence
  edge, and co-occurrence no longer deletes a withdrawn claim.
- **`inclusion` direction** — the extraction prompt and the relation spec
  described `inclusion` as "A is contained in B" while the `contains` alias
  (and every stated `contains` claim) reads source-contains-target; both now
  say "A contains B" (`proper_inclusion` likewise).  Stores built by the
  LLM extractor under the old wording may hold `inclusion` edges pointing
  from the part to the whole; they are not migrated (nothing records which
  wording produced an edge).
- **`precedes` direction** — "A precedes B" was stored as "A depends on B";
  it is now stored as "B depends on A" (`orient_relation`, used by ingest,
  the extractor and `PKMAgent.connect`), so statements, contradictions and
  withdrawals with that label find the same edge.
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
