# World Network Entropy Design

## Goal

Add a metric that estimates the average information entropy of the World 0
concept network.

This metric should answer:

- Is conceptual attention concentrated in a few deterministic paths?
- Is the graph too diffuse and noisy?
- Which concepts have clear relation structure, and which concepts spread
  activation across many weakly distinguished neighbors?

The metric is about the explicit concept-relation network. It is not token
entropy, embedding entropy, source-document entropy, or raw text diversity.

## Unit Of Entropy

The primary unit is the local relation distribution of one concept node.

For a concept `c`, collect all neighboring concepts connected by relations:

```text
N(c) = {neighbor concept ids connected to c}
```

Each relation contributes an information mass to the neighbor it connects.
Multiple relations between the same pair are summed.

```text
mass(c -> n) = sum effective_weight(relation r between c and n)
```

Then convert masses into a probability distribution:

```text
p(c -> n) = mass(c -> n) / sum mass(c -> *)
```

The local entropy of concept `c` is:

```text
H(c) = - sum p(c -> n) * log2(p(c -> n))
```

Normalize by the maximum possible entropy for the node degree:

```text
H_norm(c) = H(c) / log2(|N(c)|)
```

If `|N(c)| < 2`, define:

```text
H_norm(c) = 0
```

Reason: a concept with zero or one neighbor has no branching uncertainty.

## Effective Relation Weight

Raw relation weight alone is not enough. A weak, generic `related_to` edge
should not contribute the same information mass as a reinforced, specific,
high-confidence edge.

Use:

```text
effective_weight(r) =
    r.weight
  * r.confidence
  * relation_type_factor(r.type)
  * explicitness_factor(r)
```

Initial relation type factors:

```text
contains      1.00
part_of       1.00
depends_on    1.00
supports      0.95
activates     0.95
precedes      0.90
derived_from  0.90
contrasts     0.85
similar_to    0.70
related_to    0.45
```

Explicitness factor:

```text
explicit relation: 1.00
hebbian relation:  0.75
```

Rationale:

- Specific typed relations carry more conceptual structure.
- Generic `related_to` is useful but should not inflate information quality.
- Hebbian relations are useful co-activation evidence, but less semantically
  explicit than declared relations.

Temporal relevance should not be part of the default entropy score. Entropy
should describe current graph structure, not only recent activation. A separate
`temporal_entropy` variant can multiply by `r.temporal_relevance()` later.

## World Average Entropy

The main world-level metric is the weighted mean of normalized local entropy:

```text
world_entropy =
    sum node_importance(c) * H_norm(c)
    / sum node_importance(c)
```

Initial node importance:

```text
node_importance(c) = max(c.confidence, 0.05) * maturity_factor(c.maturity)
```

Maturity factors:

```text
embryonic    0.50
developing   0.75
established  1.00
core         1.20
fading       0.35
```

Rationale:

- A noisy embryonic concept should affect the world less than a stable concept.
- Core concepts should matter more because their local uncertainty shapes
  projections disproportionately.
- The 0.05 confidence floor prevents a large number of low-confidence nodes
  from disappearing entirely.

If there are no concepts or all node weights are zero:

```text
world_entropy = 0.0
```

## Interpretation

The normalized score is in `[0, 1]`.

```text
0.00 - 0.20  highly concentrated / under-connected
0.20 - 0.45  structured and focused
0.45 - 0.70  diverse but still interpretable
0.70 - 1.00  highly diffuse; likely noisy unless intentional
```

Important: high entropy is not automatically bad.

- For a bridge concept, high entropy can mean it connects multiple meaningful
  neighborhoods.
- For an embryonic concept, high entropy may indicate noisy extraction or
  excessive generic relations.

The metric should be reported with supporting counts:

```text
entropy_nodes_considered
entropy_isolated_nodes
entropy_bridge_nodes
entropy_high_nodes
entropy_low_nodes
```

## Diagnostic Extensions

The first implementation should expose only `avg_network_entropy`.

Later extensions can add:

### 1. Local Entropy Per Concept

Return the top high-entropy concepts:

```text
[(concept_id, representation, entropy, degree)]
```

Useful for cleanup and relation refinement.

### 2. Relation Type Entropy

Compute entropy over relation types:

```text
P(type) = count/type-weighted mass of relation type / total relation mass
```

This diagnoses whether the graph is dominated by generic `related_to`.

### 3. Community Entropy

Compute entropy of concept mass across detected communities.

This diagnoses whether the world is dominated by one conceptual region or
balanced across several regions.

### 4. Temporal Entropy

Multiply relation mass by `r.temporal_relevance()`.

This answers: "How diffuse is the currently active/recent part of the world?"

## API Shape

Recommended minimal public API:

```python
class NetworkEntropy(BaseModel):
    avg_network_entropy: float
    nodes_considered: int
    isolated_nodes: int
    high_entropy_nodes: int
    low_entropy_nodes: int

class WorldStatus(BaseModel):
    ...
    avg_network_entropy: float = 0.0
```

Implementation location:

```text
src/world0/metrics/entropy.py
```

Status integration:

```text
src/world0/world/_status.py
```

## Algorithm

```text
1. Load all concepts and relations.
2. Build adjacency: concept_id -> neighbor_id -> total effective mass.
3. For each concept:
   a. If neighbor count < 2, entropy = 0.
   b. Else normalize masses into probabilities.
   c. Compute Shannon entropy.
   d. Normalize by log2(neighbor_count).
4. Compute weighted average using node_importance.
5. Return metric and diagnostic counts.
```

## Edge Cases

- No concepts: entropy `0.0`
- Concepts but no relations: entropy `0.0`
- One relation per concept: local entropy `0.0`
- Multiple relations to the same neighbor: summed before entropy
- Self-relations: should already be rejected; ignore defensively if present
- Missing relation endpoints: ignore relation defensively

## Why This Matches World 0

This metric preserves the project boundary:

- Concept-first: entropy is computed per concept.
- Relation-first: uncertainty comes from typed relation distribution.
- Context-sensitive ready: temporal/task-conditioned variants can be added
  without redefining the base metric.
- Projection-oriented: high local entropy predicts concepts that may produce
  broad or noisy projections.
- Not RAG-like: it does not measure token frequency or document coverage.
