"""``World`` — the unified Agent interface.

This file is intentionally short.  The constructor wires up every Lego
brick (each one a Protocol-satisfying engine), and the public methods
delegate to small pipeline classes that live in sibling files.

If you need to swap an engine, subclass ``World`` and override the
relevant attribute after ``super().__init__`` — every method goes
through the attribute, never through a direct symbol import.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from world0.communities.manager import CommunityManager
from world0.concepts.api import Concepts
from world0.dynamics.activation import ActivationEngine
from world0.dynamics.color_diffusion import ColorDiffusionEngine
from world0.dynamics.community import CommunityDetector
from world0.dynamics.decay import DecayEngine
from world0.dynamics.hebbian import HebbianEngine
from world0.dynamics.lifecycle import LifecycleEngine
from world0.extraction.extractor import ConceptExtractor
from world0.prompts import PromptRegistry
from world0.projection.engine import ProjectionEngine
from world0.relations.manager import RelationManager
from world0.schemas.context import Perspective
from world0.schemas.types import (
    IngestResult,
    Observation,
    Projection,
    ReflectResult,
    WorldStatus,
)
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
            relations=[("Python", "latency", "related_to")],
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
    ) -> None:
        self._store = JsonStore(store_path)
        self._prompts = prompt_registry or PromptRegistry()

        # ── Stores ────────────────────────────────────────────────────
        self.concepts = Concepts(self._store)
        self.relations = RelationManager(self._store)
        self.concepts.load()
        self.relations.load()

        # ── Dynamics engines (each implements a core Protocol) ────────
        self._activation = ActivationEngine(self.concepts, self.relations)
        self._color_diffusion = ColorDiffusionEngine(
            self.concepts, self.relations
        )
        self._hebbian = HebbianEngine(self.relations)
        self._decay = DecayEngine(self.concepts, self.relations)
        self._lifecycle = LifecycleEngine(self.concepts, self.relations)
        self._projection = ProjectionEngine(self.concepts, self.relations)

        # Optional LLM-powered extraction
        self._extractor = (
            ConceptExtractor(llm, prompt_registry=self._prompts) if llm else None
        )

        # ── Cross-cycle state ────────────────────────────────────────
        self._state = self._store.load_state()
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
    ) -> IngestResult:
        """Extract concepts from raw text (LLM-powered) and ingest them."""
        if not self._extractor:
            raise RuntimeError(
                "ingest_text() requires an LLM provider. "
                "Pass llm=OpenAIProvider() or llm=AnthropicProvider() "
                "when creating the World instance."
            )
        observation = self._extractor.extract(text, task=task, source=source)
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
        perspective: Perspective | None = None,
        max_concepts: int = 15,
        max_depth: int = 2,
        decay: float = 0.5,
    ) -> Projection:
        """Generate a cognitive projection for the current task."""
        seed_ids: list[str] = []
        for name in seeds:
            node = self.concepts.resolve(name)
            if node:
                seed_ids.append(node.id)

        effective_task = (
            perspective.task if perspective and perspective.task else task
        )

        if not seed_ids:
            return Projection(task=effective_task)

        activations = self._activation.activate(
            seed_ids,
            max_depth=max_depth,
            decay=decay,
            source="projection",
            task=effective_task,
            record=False,
            perspective=perspective,
        )

        return self._projection.project(
            activations, max_concepts=max_concepts, task=effective_task
        )

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
