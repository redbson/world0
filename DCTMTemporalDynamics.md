# DCTM-Based Temporal Dynamics Design for World 0

## 1. Purpose

This document proposes a temporal dynamics system for World 0 that is:

- concept-first rather than document-first
- relation-first rather than event-log-first
- context-sensitive rather than globally time-decayed
- projection-oriented rather than archival

The goal is not to turn World 0 into a memory timeline. The goal is to
give concepts and relations a richer temporal life: emergence, persistence,
phase shifts, decay, recurrence, and forecastable next relevance.


## 2. Theoretical Basis

### 2.1 Interpretation of "DCTM"

`DCTM` is ambiguous across fields. For World 0, the most useful reading is:

- `Dynamic Cognitive Task Modeling` from cognitive engineering
- combined with adjacent state-space ideas from dynamic topic models
  and continuous-time dynamic topic models

This is an explicit design inference, not a claim that World 0 should become
a topic model.

### 2.2 Imported Ideas

From Dynamic Cognitive Task Modeling:

- model cognition as a changing task-conditioned process, not a static map
- combine an "ideal task structure" with observed process traces
- care about functions, transitions, and actual paths through a task

From Dynamic Topic Models:

- latent semantic structure evolves smoothly across time slices
- the current state should inherit from the previous state, not reset
- drift should be modeled in parameter space, not only through hard events

From Continuous-Time Dynamic Topic Models:

- time should not depend only on arbitrary fixed bins
- the interval between events matters
- bounded continuous-time evolution is preferable to simple stepwise decay

### 2.3 Primary References

- Gagnon et al., "Dynamic Cognitive Task Modeling of Complexity Discovery:
  A Mix of Process Tracing and Task Analysis" (2011)
  https://journals.sagepub.com/doi/pdf/10.1177/1071181311551280
- Blei and Lafferty, "Dynamic Topic Models" (2006)
  https://www.cs.columbia.edu/~blei/papers/BleiLafferty2006a.pdf
- Wang, Blei, and Heckerman, "Continuous Time Dynamic Topic Models" (2008)
  https://www.cs.columbia.edu/~blei/papers/WangBleiHeckerman2008.pdf


## 3. Why the Current System Is Not Enough

The current World 0 time behavior already has useful temporal pieces:

- concept freshness via `last_activated`
- relation freshness via `last_reinforced`
- maturity-sensitive decay
- temporal weighting in activation and projection

But it is still mostly a `recency-weighted graph`.

What is missing for a DCTM-aligned system:

- task episodes and phases
- temporal traces of actual concept use
- distinction between stable recurrence and accidental recent mention
- bounded drift of concept state, not just multiplicative decay
- temporal relations between functions and stages
- forecast of likely next conceptual needs


## 4. Design Principles

1. Time must shape cognition, not only confidence.
2. Task phase must matter as much as recency.
3. Repeated recurrence should create temporal stability.
4. Drift must be bounded; concepts should not wander indefinitely.
5. Projection should expose temporal structure only when it helps action.
6. Raw dialogue history stays outside the core model; World 0 stores
   compressed temporal structure, not transcripts.


## 5. Temporal Ontology for World 0

### 5.1 New Core Temporal Objects

#### TaskEpisode

A bounded interval of work under one dominant task frame.

Fields:

- `id`
- `task_label`
- `started_at`
- `ended_at`
- `perspective`
- `goal_stack`
- `phase`
- `trace_ids`

Role:

- creates a temporal container for cognition
- prevents all activations from collapsing into one global stream

#### CognitiveTraceEvent

A compact process-tracing event, not a raw log record.

Fields:

- `timestamp`
- `episode_id`
- `event_type`
- `concept_ids`
- `relation_ids`
- `source`
- `weight`
- `phase_hint`

Examples:

- `observe`
- `infer`
- `query`
- `resolve`
- `handoff`
- `failure`
- `reflection`

Role:

- records what functions were actually performed
- provides the empirical side of DCTM

#### TaskPhase

A coarse-grained functional stage inside an episode.

Suggested initial phases:

- `orient`
- `decompose`
- `analyze`
- `synthesize`
- `decide`
- `verify`
- `reflect`

Role:

- lets the same concept mean different things at different moments

#### ConceptTemporalState

A bounded latent state for one concept inside one episode.

Fields:

- `concept_id`
- `episode_id`
- `activation_level`
- `stability`
- `drift`
- `phase_affinity`
- `recurrence_score`
- `novelty_score`
- `fatigue`
- `predicted_next_relevance`

Role:

- separates "globally known concept" from "current temporal state"

#### RelationTemporalState

Temporal state for one relation inside one episode.

Fields:

- `relation_id`
- `episode_id`
- `activation_level`
- `stability`
- `phase_affinity`
- `transition_support`
- `recurrence_score`

Role:

- captures when a relation is phase-critical, not only structurally present


## 6. Time Scales

The system should operate across four scales.

### 6.1 Pulse Scale

Seconds to minutes.

Use:

- local activation updates
- micro-relevance during one active turn

### 6.2 Episode Scale

Minutes to hours.

Use:

- phase transition
- function sequence modeling
- emergence and temporary suppression

### 6.3 Consolidation Scale

Days to weeks.

Use:

- promotion, fading, pruning
- recurrence-based stabilization

### 6.4 Era Scale

Weeks to months.

Use:

- domain drift
- long-term perspective change
- persistent re-centering of the world


## 7. DCTM Mapping Into World 0

World 0 should map DCTM as follows:

### 7.1 Ideal Task Structure

Represent a task as a lightweight function graph:

- goals
- subgoals
- allowable phase transitions
- expected concept classes

This is not workflow execution. It is a cognitive expectation model.

### 7.2 Process Trace

During use, World 0 records compact events that indicate:

- what concepts were touched
- under which phase
- in what sequence
- with what result

### 7.3 Dynamic Alignment

The system compares:

- expected task phase path
- actual cognitive trace path

This enables:

- phase-sensitive activation
- anomaly detection
- re-projection when the current path diverges from expected reasoning


## 8. Proposed Dynamic Equations

World 0 does not need a full probabilistic topic model. But it does need
state-update equations inspired by state-space models.

### 8.1 Concept State Update

For concept `c` in episode `e` at time `t`:

```text
state_t(c) =
  bounded_decay(state_t-1(c))
  + event_input(c)
  + relation_inflow(c)
  + phase_gain(c, phase_t)
  + recurrence_bonus(c, e)
  - fatigue_penalty(c, e)
```

Where:

- `bounded_decay` should be closer to OU-style mean reversion than pure
  Brownian drift
- `event_input` comes from direct use
- `relation_inflow` comes from spreading activation
- `phase_gain` boosts concepts that fit the current task phase
- `recurrence_bonus` rewards repeated useful return
- `fatigue_penalty` reduces concepts that are over-activated but not useful

### 8.2 Relation State Update

```text
rel_state_t(r) =
  bounded_decay(rel_state_t-1(r))
  + co_activation_support(r)
  + phase_transition_support(r)
  + task_match_bonus(r)
```

`phase_transition_support` is important: some relations matter because they
bridge phases, not because they are always strong.

### 8.3 Episode-to-Global Consolidation

At episode end:

```text
global_concept_confidence +=
  usefulness * recurrence * cross-episode stability

global_relation_weight +=
  transition_support * recurrence * explanatory_value
```

This prevents one noisy episode from permanently reshaping the world.


## 9. New Temporal Metrics

The current system should be extended with these metrics.

### 9.1 Recurrence Score

How often a concept returns across distinct episodes under meaningful use.

Use:

- distinguish durable concepts from one-off mentions

### 9.2 Stability

How consistent a concept's task and phase role remains over time.

Use:

- core promotion
- projection ranking

### 9.3 Drift

How much the concept's neighborhood and task-role have shifted.

Use:

- detect domain migration
- trigger concept review or split

### 9.4 Novelty Score

How recently and unexpectedly a concept became active.

Use:

- emerging concept surfacing
- exploration prompts

### 9.5 Fatigue

Temporary down-weighting for concepts repeatedly activated in a narrow loop
without improving projection usefulness.

Use:

- reduce pathological fixation

### 9.6 Transition Support

How strongly a relation helps move from one cognitive phase to another.

Use:

- task-phase-aware projection


## 10. Architecture Changes

### 10.1 New Schemas

Add:

- `src/world0/schemas/temporal.py`

Suggested models:

- `TaskEpisode`
- `TaskPhase`
- `CognitiveTraceEvent`
- `ConceptTemporalState`
- `RelationTemporalState`
- `TemporalProjection`

### 10.2 New Engines

Add:

- `src/world0/dynamics/episode.py`
- `src/world0/dynamics/temporal_trace.py`
- `src/world0/dynamics/state_space.py`
- `src/world0/dynamics/phase.py`
- `src/world0/dynamics/forecast.py`

### 10.3 Integration With Existing Engines

#### ActivationEngine

Extend from:

- confidence
- relation weight
- recency

To:

- confidence
- phase affinity
- recurrence
- transition support
- bounded temporal state

#### DecayEngine

Extend from:

- half-life decay

To:

- decay + mean reversion + fatigue recovery

#### LifecycleEngine

Extend promotion logic with:

- recurrence threshold
- stability threshold
- cross-episode usefulness

#### ProjectionEngine

Extend selection logic with:

- current phase relevance
- emerging vs stable balance
- predicted next-step concepts


## 11. Storage Strategy

World 0 must not become a raw event warehouse.

Store:

- recent compact trace events
- episode summaries
- rolling temporal statistics
- sparse state snapshots at important transitions

Do not store:

- full transcripts as core temporal state
- every token-level interaction
- arbitrary long event history without summarization

Rule:

When an episode closes, compress its trace into:

- phase path
- key concept state deltas
- key relation state deltas
- anomalies
- forecast errors


## 12. Projection Output Changes

Projection should gain a temporal mode.

### 12.1 Temporal Projection Contents

A temporal projection should optionally include:

- `current_phase`
- `stable_concepts`
- `emerging_concepts`
- `decaying_concepts`
- `phase_bridging_relations`
- `predicted_next_concepts`
- `recent_shift_summary`

### 12.2 Example

```text
Current phase: analyze

Stable concepts:
- model serving
- deployment

Emerging concepts:
- latency budget

Decaying concepts:
- authentication

Phase-bridging relations:
- monitoring -> activates -> latency
- latency -> depends_on -> deployment

Predicted next concepts:
- autoscaling
- query optimization
```

This is still a local conceptual view, not a timeline dump.


## 13. Minimal Viable Implementation Plan

### Phase 1: Episode-Aware Time

Implement:

- `TaskEpisode`
- `TaskPhase`
- episode start/end hooks
- phase-aware trace events

No new learning yet. Only better temporal segmentation.

### Phase 2: Temporal State Summary

Implement:

- recurrence
- stability
- novelty
- temporal state snapshots

Use them in projection ranking.

### Phase 3: Phase-Sensitive Activation

Implement:

- phase affinity
- transition support
- fatigue
- predicted next relevance

### Phase 4: Consolidation and Forecast

Implement:

- cross-episode consolidation
- drift review
- forecast feedback loop


## 14. Acceptance Tests

The design is successful when the following become testable.

1. The same concept receives different activation under different task phases.
2. A recurrent concept outranks a merely recent concept when both are relevant.
3. A concept can be temporarily foregrounded inside one episode without
   permanently becoming globally central.
4. Relations that bridge task phases are preferred over generic hubs.
5. Projection exposes emerging and decaying concepts distinctly.
6. A long task path can be summarized as a phase trajectory rather than a
   transcript.
7. Forecasted next concepts measurably improve downstream projection usefulness.


## 15. Recommended First Code Changes

If this design is implemented incrementally, the first changes should be:

1. Add `schemas/temporal.py` with episode, phase, and trace models.
2. Add episode metadata to `Session` and dialogue sedimentation events.
3. Extend `Observation` with optional `phase` and `episode_id`.
4. Add `recurrence_score`, `stability`, and `novelty_score` to concept and
   relation state summaries.
5. Update `ProjectionEngine` to rank by:

```text
relevance
* phase_affinity
* recurrence_bonus
* temporal_freshness
- redundancy
```


## 16. Final Position

World 0 should not model time as simple forgetting.

A DCTM-aligned temporal system should model:

- what phase the agent is in
- what functions it is performing
- what concepts recur usefully
- what concepts are drifting
- what concepts are likely needed next

That keeps the temporal layer aligned with the project's core purpose:
supporting understanding through concept, relation, context, activation,
and projection.
