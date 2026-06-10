# Projection-quality coefficient sweep

Sensitivity of projection quality to activation/projection coefficients, over a fixed deterministic world with a deep relevant chain, a redundant near-hub cluster, and distractor domains (see `scripts/sweep_projection_quality.py`). This is a *sensitivity report*, not an auto-tuner — it shows how the output moves so tuning stays a human, explainable decision.

Swept: `projection.mmr_lambda`, `activation.propagation_min_ratio`. Budget: top-6 concepts. Baseline (shipped defaults) — P@6 0.333, NDCG@6 0.494, cross-domain distance 0.000.

Regenerate with:

```bash
python scripts/sweep_projection_quality.py \
    --param projection.mmr_lambda=0.0,0.1,0.3,0.5,0.7,0.9 \
    --param activation.propagation_min_ratio=0.03,0.1,0.3 \
    --md docs/projection-sweep-baseline.md
```

| config | P@6 | NDCG@6 | cross-domain Δ | ΔP | ΔNDCG |
|---|---|---|---|---|---|
| projection.mmr_lambda=0, activation.propagation_min_ratio=0.03 | 0.583 | 0.671 | 0.800 | +0.250 | +0.178 |
| projection.mmr_lambda=0, activation.propagation_min_ratio=0.1 | 0.583 | 0.671 | 0.800 | +0.250 | +0.178 |
| projection.mmr_lambda=0, activation.propagation_min_ratio=0.3 | 1.000 | 1.000 | 0.909 | +0.667 | +0.506 |
| projection.mmr_lambda=0.1, activation.propagation_min_ratio=0.03 | 0.500 | 0.606 | 0.667 | +0.167 | +0.112 |
| projection.mmr_lambda=0.1, activation.propagation_min_ratio=0.1 | 0.500 | 0.606 | 0.667 | +0.167 | +0.112 |
| projection.mmr_lambda=0.1, activation.propagation_min_ratio=0.3 | 1.000 | 1.000 | 0.909 | +0.667 | +0.506 |
| projection.mmr_lambda=0.3, activation.propagation_min_ratio=0.03 | 0.333 | 0.494 | 0.000 | +0.000 | +0.000 |
| projection.mmr_lambda=0.3, activation.propagation_min_ratio=0.1 | 0.333 | 0.494 | 0.000 | +0.000 | +0.000 |
| projection.mmr_lambda=0.3, activation.propagation_min_ratio=0.3 | 0.667 | 0.730 | 0.500 | +0.333 | +0.236 |
| projection.mmr_lambda=0.5, activation.propagation_min_ratio=0.03 | 0.333 | 0.494 | 0.000 | +0.000 | +0.000 |
| projection.mmr_lambda=0.5, activation.propagation_min_ratio=0.1 | 0.333 | 0.494 | 0.000 | +0.000 | +0.000 |
| projection.mmr_lambda=0.5, activation.propagation_min_ratio=0.3 | 0.500 | 0.623 | 0.667 | +0.167 | +0.130 |
| projection.mmr_lambda=0.7, activation.propagation_min_ratio=0.03 | 0.333 | 0.494 | 0.000 | +0.000 | +0.000 |
| projection.mmr_lambda=0.7, activation.propagation_min_ratio=0.1 | 0.333 | 0.494 | 0.000 | +0.000 | +0.000 |
| projection.mmr_lambda=0.7, activation.propagation_min_ratio=0.3 | 0.333 | 0.494 | 0.500 | +0.000 | +0.000 |
| projection.mmr_lambda=0.9, activation.propagation_min_ratio=0.03 | 0.333 | 0.494 | 0.000 | +0.000 | +0.000 |
| projection.mmr_lambda=0.9, activation.propagation_min_ratio=0.1 | 0.333 | 0.494 | 0.000 | +0.000 | +0.000 |
| projection.mmr_lambda=0.9, activation.propagation_min_ratio=0.3 | 0.333 | 0.494 | 0.500 | +0.000 | +0.000 |
