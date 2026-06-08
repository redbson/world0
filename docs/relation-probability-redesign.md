# Relation Probability Redesign

## Goal

World 0 relations should represent probabilistic beliefs.

A relation is no longer only "discovered" or "reinforced". It carries:

```text
P(source --type--> target) in [0, 1]
```

This probability means:

> Given the current concept world and the current extraction evidence, how
> likely is this typed relation to be correct?

It is not a raw frequency, graph weight, or embedding similarity.

## Data Model

`RelationEdge` now has:

```python
probability: float
probability_observation_count: int
```

Existing fields remain:

```python
weight: float
confidence: float
```

Compatibility rule:

- `probability` is the semantic belief score.
- `weight/confidence` remain operational fields for activation, decay, and old
  tests.
- When probability evidence is ingested, `weight/confidence` are initialized or
  updated to the probability.
- Hebbian reinforcement may still strengthen operational weight without
  changing semantic probability.

## Preset Relations

Extraction/ingest can receive preset relation priors:

```python
RelationPrior(
    source="retrieval augmented generation",
    target="vector search",
    relation_type="depends_on",
    probability=0.65,
    strength=1.0,
)
```

These priors are included in the extractor prompt and preserved on the
`Observation`.

The LLM should re-evaluate preset relations against the current text and output
updated relation probabilities when supported or contradicted.

## Probability Update

For an existing relation, the new probability is recalculated from:

1. Current relation probability
2. Optional preset prior probability
3. Optional extraction evidence probability

Formula:

```text
old_strength =
  2
  + reinforcement_count
  + disconfirmation_count
  + probability_observation_count

new_probability =
  (
    old_probability * old_strength
    + prior_probability * prior_strength
    + evidence_probability * evidence_strength
  )
  /
  (old_strength + prior_strength + evidence_strength)
```

Defaults:

```text
prior_strength = 1.0
evidence_strength = 2.0
```

Rationale:

- Existing world belief should not be overwritten by one extraction.
- Current text evidence should matter more than a preset.
- Repeated observations make relations more stable over time.

## New Relation Initialization

For a new relation:

```text
if evidence and preset exist:
    probability = weighted_average(preset, evidence)
elif evidence exists:
    probability = evidence
elif preset exists:
    probability = preset
else:
    probability = default relation strength
```

Explicit relations default stronger than Hebbian relations, preserving the
existing World 0 distinction.

## Extraction Contract

Relation output should include:

```json
{
  "source": "c1",
  "target": "c2",
  "type": "depends_on",
  "probability": 0.82,
  "confidence": 0.78,
  "evidence": "short evidence",
  "rationale": "why this relation and direction are likely"
}
```

`probability` is preferred. If absent, World 0 falls back to relation
`confidence` as the evidence probability.

## Boundary

Do not reduce relation probability to token co-occurrence.

Co-occurrence can strengthen operational graph traversal, but semantic relation
probability should primarily be updated by explicit extraction evidence,
presets, contradictions, and deliberate relation edits.
