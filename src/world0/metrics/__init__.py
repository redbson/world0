"""World 0 cognitive-structure metrics.

These are *read-only* diagnostics computed over the explicit concept /
relation graph.  They never mutate state and never touch storage beyond
reading.  Their purpose is to make projection health measurable — e.g.
whether the world is becoming diffuse/noisy or staying focused — so that
later changes to activation, extraction or projection can be evaluated
against a quantitative signal rather than subjective inspection.

See ``docs/world-network-entropy-design.md`` and
``docs/related-work-analysis.md`` (item G2) for rationale.
"""

from __future__ import annotations

from world0.metrics.entropy import NetworkEntropy, compute_network_entropy

__all__ = ["NetworkEntropy", "compute_network_entropy"]
