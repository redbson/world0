"""World — the unified Agent interface for World 0.

This is the only class an Agent needs to interact with.
It orchestrates: ingest → activate → project → reflect.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from world0.concepts.manager import ConceptManager
from world0.dynamics.activation import ActivationEngine
from world0.dynamics.decay import DecayEngine
from world0.dynamics.hebbian import HebbianEngine
from world0.dynamics.lifecycle import LifecycleEngine
from world0.extraction.extractor import ConceptExtractor
from world0.llm.base import LLMProvider
from world0.projection.engine import ProjectionEngine
from world0.relations.manager import RelationManager
from world0.schemas.concept import Maturity
from world0.schemas.relation import RelationType
from world0.schemas.types import (
    IngestResult,
    Observation,
    Projection,
    ReflectResult,
    WorldStatus,
)
from world0.store.json_store import JsonStore
from world0.visualization.renderer import visualize as _visualize


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
    ) -> None:
        self._store = JsonStore(store_path)
        self.concepts = ConceptManager(self._store)
        self.relations = RelationManager(self._store)

        # Load existing state
        self.concepts.load()
        self.relations.load()

        # Engines
        self._activation = ActivationEngine(self.concepts, self.relations)
        self._hebbian = HebbianEngine(self.relations)
        self._decay = DecayEngine(self.concepts, self.relations)
        self._lifecycle = LifecycleEngine(self.concepts, self.relations)
        self._projection = ProjectionEngine(self.concepts, self.relations)

        # Optional LLM-powered extraction
        self._extractor = ConceptExtractor(llm) if llm else None

        # Global state
        self._state = self._store.load_state()

    # ── Agent interface ───────────────────────────────────────────────

    def ingest(self, observation: Observation) -> IngestResult:
        """Agent submits observations. World 0 updates itself.

        1. For each concept name: get-or-create, then reinforce
        2. For each explicit relation: discover or reinforce
        3. Hebbian learning: co-occurring concepts strengthen connections
        4. Update descriptions if provided
        5. Persist
        """
        result = IngestResult()
        resolved_ids: list[str] = []

        # 1. Process concepts
        for name in observation.concepts:
            node, is_new = self.concepts.get_or_create(
                name, origin=observation.source, task=observation.task
            )
            # Always reinforce — creation is also an activation event
            self.concepts.reinforce(
                node.id, source=observation.source, task=observation.task
            )
            if is_new:
                result.new_concepts.append(node.name)
            else:
                result.reinforced_concepts.append(node.name)
            resolved_ids.append(node.id)

        # 2. Process explicit relations
        for src_name, tgt_name, rel_type_str in observation.relations:
            src = self.concepts.resolve(src_name)
            tgt = self.concepts.resolve(tgt_name)
            if not src or not tgt:
                continue

            try:
                rel_type = RelationType(rel_type_str)
            except ValueError:
                rel_type = RelationType.RELATED_TO

            edge, is_new = self.relations.discover(
                src.id, tgt.id, rel_type, provenance=observation.task
            )
            if is_new:
                result.new_relations.append(
                    f"{src.name} → {rel_type.value} → {tgt.name}"
                )
            else:
                self.relations.reinforce(edge.id, provenance=observation.task)
                result.reinforced_relations.append(
                    f"{src.name} → {rel_type.value} → {tgt.name}"
                )

        # 3. Hebbian learning on co-occurring concepts
        if len(resolved_ids) > 1:
            new_hebbian = self._hebbian.learn(
                resolved_ids, provenance=observation.task
            )
            for rid in new_hebbian:
                edge = self.relations.get(rid)
                if edge:
                    src = self.concepts.get(edge.source_id)
                    tgt = self.concepts.get(edge.target_id)
                    if src and tgt:
                        result.hebbian_relations.append(
                            f"{src.name} ↔ {tgt.name}"
                        )

        # 4. Update descriptions
        for name, desc in observation.descriptions.items():
            node = self.concepts.resolve(name)
            if node:
                self.concepts.update_description(node.id, desc)

        # 5. Batch-flush dirty objects to disk
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
        """Extract concepts from raw text and ingest them.

        Uses the LLM provider to automatically identify concepts and
        relations, then feeds them into the cognitive world.

        Requires an LLM provider to be configured::

            from world0.llm import OpenAIProvider
            w = World(store_path=".world0", llm=OpenAIProvider())
            result = w.ingest_text("We deployed the model with Docker...")

        Args:
            text: Raw text (conversation, document, notes, etc.).
            task: Task context label.
            source: Provenance label.

        Returns:
            IngestResult summarizing what was created/reinforced.

        Raises:
            RuntimeError: If no LLM provider is configured.
        """
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
        self._extractor = ConceptExtractor(llm) if llm else None

    def project(
        self,
        seeds: list[str],
        *,
        task: str = "",
        max_concepts: int = 15,
        max_depth: int = 2,
        decay: float = 0.5,
    ) -> Projection:
        """Generate a cognitive projection for the current task.

        Seeds are concept names/aliases provided by the Agent.
        Returns a Projection whose .render() output can be injected
        into the Agent's prompt.
        """
        # Resolve seed names to ids
        seed_ids: list[str] = []
        for name in seeds:
            node = self.concepts.resolve(name)
            if node:
                seed_ids.append(node.id)

        if not seed_ids:
            return Projection(task=task)

        # Spread activation (read-only — projection does not mutate state)
        activations = self._activation.activate(
            seed_ids,
            max_depth=max_depth,
            decay=decay,
            source="projection",
            task=task,
            record=False,
        )

        # Generate projection
        return self._projection.project(
            activations, max_concepts=max_concepts, task=task
        )

    def reflect(self) -> ReflectResult:
        """Cognitive consolidation — run after a task is complete.

        1. Decay unused concepts and relations
        2. Evaluate lifecycle promotions/demotions
        3. Prune deeply decayed items
        4. Persist everything
        """
        result = ReflectResult()

        # 1. Decay
        result.decayed_concepts = self._decay.decay_concepts()
        result.decayed_relations = self._decay.decay_relations()

        # 2. Lifecycle
        promoted, demoted = self._lifecycle.evaluate()
        result.promoted_concepts = promoted
        result.demoted_concepts = demoted

        # 3. Prune
        result.pruned_relations = self._decay.prune_relations()
        result.pruned_concepts = self._decay.prune_concepts()

        # 4. Persist — reflect is a consolidation point, flush everything
        self.concepts.flush()
        self.relations.flush()
        self._state["last_reflect"] = datetime.now(timezone.utc).isoformat()
        self._store.save_state(self._state)

        return result

    def status(self) -> WorldStatus:
        """Overview of the cognitive world's current state."""
        all_concepts = self.concepts.all()
        by_maturity: dict[str, int] = {}
        total_confidence = 0.0
        for c in all_concepts:
            by_maturity[c.maturity.value] = by_maturity.get(c.maturity.value, 0) + 1
            total_confidence += c.confidence

        last_reflect = self._state.get("last_reflect")
        return WorldStatus(
            total_concepts=len(all_concepts),
            total_relations=len(self.relations),
            by_maturity=by_maturity,
            avg_confidence=(
                total_confidence / len(all_concepts) if all_concepts else 0.0
            ),
            last_reflect=(
                datetime.fromisoformat(last_reflect) if last_reflect else None
            ),
        )

    def visualize(
        self,
        output: str | Path | None = None,
        *,
        open_browser: bool = True,
    ) -> Path:
        """Generate an interactive HTML visualization of the concept network.

        Args:
            output: Output file path. Defaults to "world0_viz.html".
            open_browser: Open in default browser automatically.

        Returns:
            Path to the generated HTML file.
        """
        return _visualize(self, output=output, open_browser=open_browser)
