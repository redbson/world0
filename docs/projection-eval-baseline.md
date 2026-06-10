# Projection-quality eval baseline

Offline retrieval metrics for World 0 projections over a fixed, deterministic, LLM-free world (see `scripts/eval_projection_matrix.py`). Regenerate with:

```bash
python scripts/eval_projection_matrix.py --md docs/projection-eval-baseline.md
```

Perspective: `(none)`

- **seed_rate** — fraction of seed labels that resolved (incl. fuzzy)
- **p@5 / p@8** — precision of the top-5 / full selection vs the gold set
- **r@8** — recall of the gold set within the projection
- **ndcg@8** — ranking quality (relevant concepts ranked high)
- **noise_rate** — fraction of the projection that is distractor noise
- **typed_ratio** — share of relations that are typed (not generic_relation)
- **inhibition** — `ok` when no should-be-suppressed concept leaked in

| case | seed_rate | p@5 | p@8 | r@8 | ndcg@8 | noise_rate | typed_ratio | inhibition |
|---|---|---|---|---|---|---|---|---|
| ml_from_serving | 1.000 | 0.600 | 0.750 | 1.000 | 0.915 | 0.250 | 0.389 | ok |
| ops_from_serving | 1.000 | 0.800 | 0.750 | 1.000 | 0.971 | 0.250 | 0.333 | ok |
| typo_seed | 1.000 | 0.600 | 0.750 | 1.000 | 0.915 | 0.000 | 0.389 | ok |
