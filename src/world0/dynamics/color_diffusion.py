"""Color-field dynamics for World 0.

Implements the observation layer described in
``docs/world0-color-field-dynamics.md`` §29 "阶段 A":

- coarse external seeding (the legacy task/domain injection) coexists
  with **community-driven** color sources (``seed_from_communities``);
- diffusion runs along the same relation graph used by activation,
  but with its own weights — consistent with doc §7.3 ("染色 ≠
  激活，但共用结构");
- each concept's per-component (per color) load undergoes **independent
  fade** (``fade_step``) driven by its local neighborhood support, so a
  concept that drifts out of one community loses that color while
  keeping others — the doc's §9.4 property.

The engine never mutates projection output: its product lives entirely
inside ``ConceptNode.domain_profile``, so downstream code can choose
whether to read it.
"""

from __future__ import annotations

import re
from collections import defaultdict
from statistics import median
from typing import TYPE_CHECKING

from world0.dynamics.coefficients import RELATION_TYPE_FACTOR
from world0.schemas.community import Community

if TYPE_CHECKING:
    from world0.core import ConceptStore, RelationStore

DOMAIN_SEED_STRENGTH: float = 0.55
DOMAIN_DIFFUSION_RATE: float = 0.18
DOMAIN_DIFFUSION_DECAY: float = 0.62
DOMAIN_REFLECT_RATE: float = 0.08
DOMAIN_DIFFUSION_STEPS: int = 2
DOMAIN_MIN_TRANSFER: float = 0.01

# Community-source injection strength.  Higher than a single task/domain
# inject because communities are *earned* (stability-gated) and should
# dominate the color field over short-lived task labels.
COMMUNITY_CORE_SEED_STRENGTH: float = 0.65
COMMUNITY_MEMBER_SEED_STRENGTH: float = 0.35

# Per-component fade (doc §19.3).  ``FADE_TAU`` is the base relaxation
# time in "reflect cycles"; ``FADE_FLOOR`` guarantees weak components
# eventually vanish even when support is ambiguous.
FADE_TAU: float = 4.0
FADE_FLOOR: float = 0.02
# Components below this strength are discarded entirely after fade
# (doc §11.5 "弱颜色自动蒸发").
COMPONENT_EVAPORATE_THRESHOLD: float = 0.015

GENERIC_DOMAIN_LABELS = {
    "",
    "knowledge intake",
    "manual connection",
}


def normalize_domain_label(label: str) -> str:
    """Convert free-form task/domain text into a stable domain label."""
    normalized = re.sub(r"\s+", " ", label.strip().lower())
    if not normalized:
        return ""

    if ":" in normalized:
        head, tail = normalized.split(":", 1)
        if head and len(head.split()) <= 3:
            normalized = head.strip() or tail.strip()

    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff _/-]+", "", normalized)
    normalized = normalized.strip(" _/-")
    if normalized in GENERIC_DOMAIN_LABELS:
        return ""
    return normalized[:64]


class ColorDiffusionEngine:
    """Seeds domain colors and diffuses them through the concept graph.

    Implements the ``ColorField`` Protocol from ``world0.core``.
    """

    def __init__(
        self,
        concepts: "ConceptStore",
        relations: "RelationStore",
    ) -> None:
        self._concepts = concepts
        self._relations = relations

    def seed_and_diffuse(
        self,
        concept_ids: list[str],
        *,
        domain_label: str,
        steps: int = DOMAIN_DIFFUSION_STEPS,
        rate: float = DOMAIN_DIFFUSION_RATE,
        decay: float = DOMAIN_DIFFUSION_DECAY,
    ) -> None:
        normalized = normalize_domain_label(domain_label)
        if not normalized or not concept_ids:
            return

        seeded: list[str] = []
        for concept_id in concept_ids:
            node = self._concepts.get(concept_id)
            if not node:
                continue
            self._blend(node, normalized, DOMAIN_SEED_STRENGTH)
            seeded.append(concept_id)

        if seeded:
            self.diffuse_from_sources(
                seeded,
                steps=steps,
                rate=rate,
                decay=decay,
            )

    def settle(self, steps: int = 1, rate: float = DOMAIN_REFLECT_RATE) -> None:
        """Run a slow global diffusion pass during reflection."""
        source_ids = [
            node.id for node in self._concepts.all() if node.domain_profile
        ]
        self.diffuse_from_sources(
            source_ids,
            steps=steps,
            rate=rate,
            decay=DOMAIN_DIFFUSION_DECAY,
        )

    # ── community-driven color sources (doc §4.3) ──────────────────────

    def seed_from_communities(
        self,
        communities: list[Community],
        *,
        diffuse: bool = True,
    ) -> int:
        """Inject community color into member concepts.

        Core members receive a stronger seed than the wider ring, as
        specified in doc §5.2 ("从核心开始最纯, 向内圈逐步增强").  Only
        stable communities (``Community.is_color_source()``) are valid
        sources; weaker candidates are ignored so a one-cycle
        coincidence never gets to dye the field.

        Returns the number of concepts that were touched.
        """
        if not communities:
            return 0

        touched: set[str] = set()
        frontier: list[str] = []
        for com in communities:
            if not com.is_color_source():
                continue
            core_set = set(com.core_ids)
            for member_id in com.member_ids:
                node = self._concepts.get(member_id)
                if not node:
                    continue
                strength = (
                    COMMUNITY_CORE_SEED_STRENGTH
                    if member_id in core_set
                    else COMMUNITY_MEMBER_SEED_STRENGTH
                )
                # Community id acts as the color label; this re-uses
                # the domain_profile / domain_color_for pipeline without
                # conflicting with free-form task/domain strings.
                if self._blend(node, com.id, strength):
                    touched.add(member_id)
                    frontier.append(member_id)

        if diffuse and frontier:
            # Fewer diffusion steps than a task-injection cycle: the
            # community already placed color at the right anchors, so
            # we only need to wash it gently into neighbors.
            self.diffuse_from_sources(
                list(dict.fromkeys(frontier)),
                steps=1,
                rate=DOMAIN_DIFFUSION_RATE * 0.7,
                decay=DOMAIN_DIFFUSION_DECAY,
            )
        return len(touched)

    # ── per-component fade (doc §9.4 + §19.3) ──────────────────────────

    def fade_step(
        self,
        *,
        dt: float = 1.0,
        tau: float = FADE_TAU,
        evaporate: float = COMPONENT_EVAPORATE_THRESHOLD,
    ) -> int:
        """Relax each color component toward what its neighborhood supports.

        For concept ``i`` with color components ``c_i^alpha``:

          q_i^alpha  = weighted avg of c_j^alpha over coupled neighbors j
          mu_i^alpha = (1/tau) * max(0, 1 - q_i^alpha / (median_beta q_i^beta))
          c_i^alpha *= (1 - dt * mu_i^alpha)

        A component drops out entirely below ``evaporate``.  A concept
        whose neighborhood still supplies a color receives ``mu ≈ 0``
        and keeps that component; a component whose support has
        collapsed fades at the natural time scale ``tau``.

        Returns the number of concepts whose profile was modified.
        """
        coupling = self._build_coupling()
        touched = 0

        for node in self._concepts.all():
            if not node.domain_profile:
                continue
            neighbors = coupling.get(node.id, {})
            support = self._neighborhood_support(
                neighbors, list(node.domain_profile.keys())
            )
            # Median across this concept's *own* components (doc §19.3
            # uses the per-node distribution as its own normaliser).
            values = list(support.values())
            ref = median(values) if values else 0.0

            updated_profile: dict[str, float] = {}
            changed = False
            for label, strength in node.domain_profile.items():
                q = support.get(label, 0.0)
                if ref <= 0:
                    # No evidence either way — light baseline fade so
                    # isolated colors do not linger forever.
                    mu = FADE_FLOOR
                else:
                    deficit = max(0.0, 1.0 - q / (ref + 1e-9))
                    mu = max(FADE_FLOOR, deficit / tau)
                new_strength = max(0.0, strength * (1.0 - dt * mu))
                if new_strength < evaporate:
                    changed = True
                    continue
                if abs(new_strength - strength) > 1e-6:
                    changed = True
                updated_profile[label] = round(new_strength, 6)

            if changed:
                node.domain_profile = updated_profile
                self._sync_dominant_domain(node)
                self._concepts.mark_dirty(node.id)
                touched += 1

        return touched

    def _neighborhood_support(
        self,
        neighbors: dict[str, float],
        labels: list[str],
    ) -> dict[str, float]:
        """q_i^alpha per color label — coupling-weighted neighbor average."""
        if not neighbors:
            return {label: 0.0 for label in labels}
        total_coupling = sum(neighbors.values()) or 1.0
        result = {label: 0.0 for label in labels}
        for neighbor_id, coupling_w in neighbors.items():
            neighbor = self._concepts.get(neighbor_id)
            if not neighbor or not neighbor.domain_profile:
                continue
            for label in labels:
                result[label] += (
                    coupling_w * neighbor.domain_profile.get(label, 0.0)
                )
        return {label: value / total_coupling for label, value in result.items()}

    def _build_coupling(self) -> dict[str, dict[str, float]]:
        coupling: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for edge in self._relations.all():
            type_factor = RELATION_TYPE_FACTOR.get(edge.relation_type, 0.5)
            if type_factor <= 0 or edge.weight <= 0:
                continue
            k = edge.weight * type_factor
            coupling[edge.source_id][edge.target_id] += k
            coupling[edge.target_id][edge.source_id] += k
        return coupling

    def diffuse_from_sources(
        self,
        source_ids: list[str],
        *,
        steps: int,
        rate: float,
        decay: float,
    ) -> None:
        frontier = list(dict.fromkeys(source_ids))
        for step in range(steps):
            if not frontier:
                break

            transfers: dict[str, dict[str, float]] = defaultdict(
                lambda: defaultdict(float)
            )
            step_rate = rate * (decay ** step)

            for concept_id in frontier:
                node = self._concepts.get(concept_id)
                if not node or not node.domain_profile:
                    continue

                for rel in self._relations.for_concept(concept_id):
                    neighbor_id = rel.other_end(concept_id)
                    neighbor = self._concepts.get(neighbor_id) if neighbor_id else None
                    if not neighbor:
                        continue

                    edge_factor = rel.weight * RELATION_TYPE_FACTOR.get(
                        rel.relation_type,
                        0.5,
                    )
                    if edge_factor <= 0:
                        continue

                    transfer_factor = step_rate * edge_factor
                    if transfer_factor < DOMAIN_MIN_TRANSFER:
                        continue

                    for domain_label, strength in node.domain_profile.items():
                        amount = strength * transfer_factor
                        if amount >= DOMAIN_MIN_TRANSFER:
                            transfers[neighbor.id][domain_label] += amount

            next_frontier: list[str] = []
            for concept_id, domain_transfers in transfers.items():
                node = self._concepts.get(concept_id)
                if not node:
                    continue

                changed = False
                for domain_label, amount in domain_transfers.items():
                    changed = self._blend(node, domain_label, amount) or changed
                if changed:
                    next_frontier.append(concept_id)

            frontier = next_frontier

    def _blend(self, node, domain_label: str, amount: float) -> bool:
        domain_label = normalize_domain_label(domain_label)
        if not domain_label or amount <= 0:
            return False

        current = node.domain_profile.get(domain_label, 0.0)
        blended = min(1.0, current + amount * (1.0 - current))
        if abs(blended - current) < 1e-6:
            return False

        node.domain_profile[domain_label] = round(blended, 6)
        self._sync_dominant_domain(node)
        self._concepts.mark_dirty(node.id)
        return True

    @staticmethod
    def _sync_dominant_domain(node) -> None:
        if not node.domain_profile:
            node.domain = ""
            return
        node.domain = max(
            node.domain_profile.items(),
            key=lambda item: item[1],
        )[0]
