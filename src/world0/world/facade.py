"""``World`` — the unified Agent interface.

This file is intentionally short.  The constructor wires up every Lego
brick (each one a Protocol-satisfying engine), and the public methods
delegate to small pipeline classes that live in sibling files.

If you need to swap an engine, subclass ``World`` and override the
relevant attribute after ``super().__init__`` — every method goes
through the attribute, never through a direct symbol import.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from world0.communities.manager import CommunityManager
from world0.concepts.api import Concepts
from world0.dynamics.activation import ActivationEngine
from world0.dynamics.coefficients import (
    ActivationConfig,
    ProjectionConfig,
)
from world0.dynamics.color_diffusion import ColorDiffusionEngine
from world0.dynamics.community import CommunityDetector
from world0.dynamics.decay import DecayEngine
from world0.dynamics.hebbian import HebbianEngine
from world0.dynamics.lifecycle import LifecycleEngine
from world0.extraction.extractor import ConceptExtractor
from world0.perspectives import PerspectiveRegistry
from world0.prompts import PromptRegistry
from world0.projection.engine import ProjectionEngine
from world0.relations.manager import RelationManager
from world0.schemas.context import Perspective
from world0.schemas.relation import (
    is_known_relation_type,
    normalize_semantic_relation,
    semantic_relation_spec,
)
from world0.schemas.types import (
    FeedbackResult,
    IngestResult,
    Observation,
    Projection,
    ReflectResult,
    RelationPrior,
    WorldStatus,
)
from world0.sources import SourceLibrary
from world0.store.json_store import JsonStore
from world0.visualization.renderer import visualize as _visualize
from world0.world._identity import IdentityOps
from world0.world._ingest import IngestPipeline
from world0.world._reflect import ReflectPipeline
from world0.world._status import build_status

if TYPE_CHECKING:
    from world0.core import LLMProvider


class World:
    """World 0 — a persistent cognitive layer for LLM Agents.

    Usage::

        w = World(store_path=".world0")

        # Agent submits observations from its work
        w.ingest(Observation(
            concepts=["Python", "deployment", "latency"],
            relations=[("Python", "latency", "generic_relation")],
            task="optimize ML serving",
            source="session_001",
        ))

        # Agent requests a cognitive projection for a new task
        proj = w.project(["Python", "deployment"], task="debug prod issue")
        print(proj.render())  # inject this into the Agent's prompt

        # After task completion, consolidate
        w.reflect()
    """

    def __init__(
        self,
        store_path: str | Path = ".world0",
        llm: LLMProvider | None = None,
        prompt_registry: PromptRegistry | None = None,
        activation_config: ActivationConfig | None = None,
        projection_config: ProjectionConfig | None = None,
    ) -> None:
        self._store = JsonStore(store_path)
        self._prompts = prompt_registry or PromptRegistry()

        # ── Stores ────────────────────────────────────────────────────
        self.concepts = Concepts(self._store)
        self.relations = RelationManager(self._store)
        self.sources = SourceLibrary(self._store)
        self.concepts.load()
        self.relations.load()

        # ── Dynamics engines (each implements a core Protocol) ────────
        self._activation = ActivationEngine(
            self.concepts, self.relations, config=activation_config
        )
        self._color_diffusion = ColorDiffusionEngine(
            self.concepts, self.relations
        )
        self._hebbian = HebbianEngine(self.relations)
        self._decay = DecayEngine(self.concepts, self.relations)
        self._lifecycle = LifecycleEngine(self.concepts, self.relations)
        self._projection = ProjectionEngine(
            self.concepts, self.relations, config=projection_config
        )

        # Optional LLM-powered extraction
        self._extractor = (
            ConceptExtractor(llm, prompt_registry=self._prompts) if llm else None
        )

        # ── Cross-cycle state ────────────────────────────────────────
        self._state = self._store.load_state()
        self.perspectives = PerspectiveRegistry(
            self._state,
            save=lambda: self._store.save_state(self._state),
        )
        self._community_detector = CommunityDetector(
            self.concepts, self.relations
        )
        self._communities = CommunityManager.from_snapshot(
            self._state.get("communities"), self._community_detector
        )

        # ── Pipelines ────────────────────────────────────────────────
        self._ingest_pipeline = IngestPipeline(
            concepts=self.concepts,
            relations=self.relations,
            hebbian=self._hebbian,
            color=self._color_diffusion,
        )
        self._reflect_pipeline = ReflectPipeline(
            decay=self._decay,
            lifecycle=self._lifecycle,
            color=self._color_diffusion,
            communities=self._communities,
        )
        self._identity = IdentityOps(
            concepts=self.concepts, relations=self.relations
        )

    # ── Agent interface ───────────────────────────────────────────────

    def ingest(self, observation: Observation) -> IngestResult:
        """Agent submits observations. World 0 updates itself."""
        result = self._ingest_pipeline.run(observation)
        # Pipelines never persist — facade owns the flush boundary.
        self.concepts.flush()
        self.relations.flush()
        return result

    def ingest_text(
        self,
        text: str,
        *,
        task: str = "",
        source: str = "",
        llm: LLMProvider | None = None,
        preset_relations: list[RelationPrior | dict] | None = None,
    ) -> IngestResult:
        """Extract concepts from raw text (LLM-powered) and ingest them."""
        extractor = (
            ConceptExtractor(llm, prompt_registry=self._prompts)
            if llm is not None
            else self._extractor
        )
        if not extractor:
            raise RuntimeError(
                "ingest_text() requires an LLM provider. "
                "Pass llm=OpenAIProvider() or llm=AnthropicProvider() "
                "when creating the World instance."
            )
        observation = extractor.extract(
            text,
            task=task,
            source=source,
            preset_relations=preset_relations,
        )
        source_record = self.sources.record_raw(text, task=task, source=source)
        observation.source_id = source_record.id
        self.sources.attach_observation(source_record.id, observation)
        return self.ingest(observation)

    def set_llm(self, llm: LLMProvider | None) -> None:
        """Update the LLM provider used for text extraction."""
        self._extractor = (
            ConceptExtractor(llm, prompt_registry=self._prompts) if llm else None
        )

    def project(
        self,
        seeds: list[str],
        *,
        task: str = "",
        perspective: Perspective | str | None = None,
        max_concepts: int = 15,
        max_depth: int = 2,
        decay: float = 0.5,
        fuzzy_seeds: bool = True,
        now: datetime | None = None,
    ) -> Projection:
        """Generate a cognitive projection for the current task.

        ``perspective`` accepts either a ``Perspective`` instance or the
        name of a registered profile (see ``world.perspectives``).

        Seed labels resolve gracefully: exact name/alias first, then
        domain disambiguation, then fuzzy signature match (disable with
        ``fuzzy_seeds=False``).  How each seed resolved is recorded in
        ``Projection.seed_resolution`` so consumers can report which
        seeds were guessed or missed.
        """
        if isinstance(perspective, str):
            resolved = self.perspectives.get(perspective)
            if resolved is None:
                raise KeyError(
                    f"Unknown perspective profile {perspective!r}. "
                    f"Available: {', '.join(self.perspectives.names())}"
                )
            perspective = resolved

        active_domains = (
            perspective.active_domains if perspective else None
        )
        seed_ids: list[str] = []
        seed_resolution: dict[str, str] = {}
        for name in seeds:
            node, method = self.concepts.resolve_in_context(
                name,
                active_domains=active_domains,
                fuzzy=fuzzy_seeds,
            )
            if node is not None:
                if node.id not in seed_ids:
                    seed_ids.append(node.id)
                label = (
                    node.name if method == "exact" else f"{node.name} ({method})"
                )
                seed_resolution[name] = label
            else:
                seed_resolution[name] = "unresolved"

        effective_task = (
            perspective.task if perspective and perspective.task else task
        )

        if not seed_ids:
            return Projection(
                task=effective_task,
                seeds=list(seeds),
                seed_resolution=seed_resolution,
            )

        activations, traces = self._activation.activate_traced(
            seed_ids,
            max_depth=max_depth,
            decay=decay,
            source="projection",
            task=effective_task,
            record=False,
            perspective=perspective,
            now=now,
        )

        projection = self._projection.project(
            activations,
            max_concepts=max_concepts,
            task=effective_task,
            perspective=perspective,
            traces=traces,
            now=now,
        )
        projection.seeds = list(seeds)
        projection.seed_resolution = seed_resolution
        return projection

    def reflect(self) -> ReflectResult:
        """Cognitive consolidation — run after a task is complete."""
        result = self._reflect_pipeline.run()
        self.concepts.flush()
        self.relations.flush()
        self._state["last_reflect"] = datetime.now(timezone.utc).isoformat()
        self._state["communities"] = self._communities.snapshot()
        self._store.save_state(self._state)
        return result

    # ── Identity operations (delegate to IdentityOps) ───────────────

    def merge(self, keeper: str, absorbed: str) -> bool:
        ok = self._identity.merge(keeper, absorbed)
        if ok:
            self.concepts.flush()
            self.relations.flush()
        return ok

    def split(
        self,
        source: str,
        new_name: str,
        *,
        aliases_to_move: list[str] | None = None,
        description: str = "",
    ) -> str | None:
        new_id = self._identity.split(
            source,
            new_name,
            aliases_to_move=aliases_to_move,
            description=description,
        )
        if new_id is not None:
            self.concepts.flush()
        return new_id

    def weaken(
        self, concept: str, *, source: str = "", task: str = ""
    ) -> bool:
        ok = self._identity.weaken(concept, source=source, task=task)
        if ok:
            self.concepts.flush()
        return ok

    def find_similar(
        self,
        text: str,
        *,
        domain: str = "",
        min_similarity: float = 0.3,
        limit: int = 5,
    ) -> list[tuple[str, float]]:
        return self._identity.find_similar(
            text,
            domain=domain,
            min_similarity=min_similarity,
            limit=limit,
        )

    # ── Usage feedback (public API for any consumer) ─────────────────

    def apply_feedback(
        self,
        *,
        useful_concepts: list[str] | None = None,
        missing_concepts: list[str] | None = None,
        noisy_concepts: list[str] | None = None,
        useful_relations: list[str] | None = None,
        weak_relations: list[str] | None = None,
        task: str = "",
        source: str = "projection_feedback",
        confidence_delta: float = 0.05,
    ) -> FeedbackResult:
        """Apply usage feedback to the world through the public facade.

        Concept references resolve by id or name; relation references are
        relation ids or ``"src -> relation -> tgt"`` labels.  This is the
        canonical channel for agents and the autonomy loop to reinforce
        what helped and weaken what misled — no internal access required.
        """
        result = FeedbackResult()
        noisy_ids: set[str] = set()
        weakened_relation_ids: set[str] = set()

        for name in _clean(noisy_concepts):
            node = self.concepts.resolve(name)
            if node:
                adjusted = self.concepts.adjust_confidence(
                    node.id, -confidence_delta
                )
                if adjusted:
                    noisy_ids.add(adjusted.id)
                    result.demoted_concepts.append(adjusted.name)

        for name in _clean(missing_concepts):
            node, is_new = self.concepts.get_or_create(
                name, origin=source, task=task
            )
            self.concepts.reinforce(node.id, source=source, task=task)
            if is_new:
                result.created_concepts.append(node.name)
            else:
                result.reinforced_concepts.append(node.name)

        for ref in _clean(useful_concepts):
            node = self.concepts.resolve(ref)
            if node and node.id not in noisy_ids:
                self.concepts.reinforce(node.id, source=source, task=task)
                if node.name not in result.reinforced_concepts:
                    result.reinforced_concepts.append(node.name)

        for ref in _clean(weak_relations):
            rid = self._resolve_relation_ref(ref)
            if rid:
                adjusted = self.relations.adjust_strength(
                    rid,
                    weight_delta=-confidence_delta,
                    confidence_delta=-confidence_delta,
                )
                if adjusted:
                    weakened_relation_ids.add(adjusted.id)
                    result.weakened_relations.append(ref)

        for ref in _clean(useful_relations):
            rid = self._resolve_relation_ref(ref)
            if rid and rid not in weakened_relation_ids:
                self.relations.reinforce(rid, provenance=f"{source}:{task}")
                result.reinforced_relations.append(ref)

        self.concepts.flush()
        self.relations.flush()
        return result

    def _resolve_relation_ref(self, ref: str) -> str | None:
        """Resolve a relation id or ``src -> relation -> tgt`` label."""
        if self.relations.get(ref) is not None:
            return ref
        match = re.match(
            r"^\s*(.+?)\s*->\s*([a-z_]+)\s*->\s*(.+?)\s*$",
            ref,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        source_name, rel_type_name, target_name = match.groups()
        source = self.concepts.resolve(source_name)
        target = self.concepts.resolve(target_name)
        if not source or not target:
            return None
        semantic_relation = (
            normalize_semantic_relation(rel_type_name.strip().lower())
            if is_known_relation_type(rel_type_name)
            else ""
        )
        rel_type = (
            semantic_relation_spec(semantic_relation).axis
            if semantic_relation
            else None
        )
        relation = None
        if semantic_relation:
            for candidate in self.relations.find_any_between(
                source.id, target.id
            ):
                if candidate.semantic_relation == semantic_relation:
                    relation = candidate
                    break
        if relation is None:
            relation = self.relations.find_between(
                source.id, target.id, rel_type
            )
        if relation is None and rel_type is not None:
            relation = self.relations.find_between(source.id, target.id, None)
        return relation.id if relation else None

    # ── Status / Visualization ──────────────────────────────────────

    def status(self) -> WorldStatus:
        return build_status(
            concepts=self.concepts,
            relations=self.relations,
            communities=self._communities,
            last_reflect_iso=self._state.get("last_reflect"),
        )

    def visualize(
        self,
        output: str | Path | None = None,
        *,
        open_browser: bool = True,
    ) -> Path:
        """Generate an interactive HTML visualization of the concept network."""
        return _visualize(self, output=output, open_browser=open_browser)


def _clean(items: list[str] | None) -> list[str]:
    """Strip blanks from a feedback reference list."""
    return [item.strip() for item in (items or []) if item and item.strip()]
