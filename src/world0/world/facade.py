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
from world0.perspectives import get_perspective
from world0.prompts import PromptRegistry
from world0.projection.engine import ProjectionEngine
from world0.relations.manager import RelationManager
from world0.schemas.clock import CognitiveClock
from world0.schemas.context import Perspective
from world0.schemas.types import (
    IngestResult,
    Observation,
    Projection,
    ReflectResult,
    RelationPrior,
    WorldStatus,
)
from world0.sources import SourceLibrary
from world0.store.json_store import JsonStore
from world0.store.sqlite_store import SqliteStore
from world0.visualization.renderer import visualize as _visualize
from world0.world._identity import IdentityOps
from world0.world._ingest import IngestPipeline
from world0.world._reflect import ReflectPipeline
from world0.world._status import build_status

if TYPE_CHECKING:
    from world0.core import LLMProvider


# Learning-record persistence policy (see ``World._persist_learning_state``).
LEARNING_EAGER_LIMIT: int = 2_000
LEARNING_PERSIST_EVERY: int = 20


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
        auto_reflect_every: int | None = None,
        backend: str = "auto",
    ) -> None:
        # ``backend``: "json" (one file per record under ``store_path``),
        # "sqlite" (a single database file at ``store_path``), or "auto"
        # (sqlite when ``store_path`` ends in .sqlite/.sqlite3/.db, else json).
        self._store = self._open_store(store_path, backend)
        self._prompts = prompt_registry or PromptRegistry()
        # Continuous mode: run a light reflect (decay + lifecycle + prune,
        # no community / colour passes) every N observations so the world
        # keeps consolidating without anyone remembering to call
        # ``reflect()``.  Decay is idempotent in cognitive time, so the
        # cadence only changes *when* forgetting is applied, never how much.
        self._auto_reflect_every = (
            int(auto_reflect_every) if auto_reflect_every and auto_reflect_every > 0 else None
        )

        # ── Cross-cycle state + cognitive clock ───────────────────────
        # Time in World 0 is counted in observations: the clock advances
        # once per ``ingest()`` and is persisted with the world state.
        self._state = self._store.load_state()
        self._clock = CognitiveClock(int(self._state.get("tick") or 0))

        # ── Stores ────────────────────────────────────────────────────
        self.concepts = Concepts(self._store, clock=self._clock)
        self.relations = RelationManager(self._store, clock=self._clock)
        self.sources = SourceLibrary(self._store)
        self.concepts.load()
        self.relations.load()

        # ── Dynamics engines (each implements a core Protocol) ────────
        self._activation = ActivationEngine(
            self.concepts, self.relations, clock=self._clock
        )
        self._color_diffusion = ColorDiffusionEngine(
            self.concepts, self.relations
        )
        self._hebbian = HebbianEngine(self.relations)
        self._decay = DecayEngine(self.concepts, self.relations, clock=self._clock)
        self._lifecycle = LifecycleEngine(self.concepts, self.relations)
        self._projection = ProjectionEngine(
            self.concepts, self.relations, clock=self._clock
        )

        # Optional LLM-powered extraction
        self._extractor = (
            ConceptExtractor(llm, prompt_registry=self._prompts) if llm else None
        )

        # ── Communities ──────────────────────────────────────────────
        self._community_detector = CommunityDetector(
            self.concepts, self.relations, clock=self._clock
        )
        self._communities = CommunityManager.from_snapshot(
            self._state.get("communities"), self._community_detector
        )
        # Hebbian co-occurrence counters are learning state: restore them
        # so a pair first seen last session and again now still crosses
        # the discovery threshold.
        learning = self._store.load_learning_state()
        migrated = False
        if not learning and (
            "hebbian_pending" in self._state or "hebbian_stats" in self._state
        ):
            # Stores written before the learning record existed kept the
            # counters inside state.json; migrate them once.
            learning = {
                "hebbian_pending": self._state.pop("hebbian_pending", None),
                "hebbian_stats": self._state.pop("hebbian_stats", None),
            }
            migrated = True
        self._hebbian.restore(learning.get("hebbian_pending"))
        self._hebbian.restore_stats(learning.get("hebbian_stats"))
        self._learning_saved: dict = {
            "hebbian_pending": self._hebbian.snapshot(),
            "hebbian_stats": self._hebbian.stats_snapshot(),
        }
        self._learning_saved_tick = self._clock.tick
        if migrated:
            # Write the migrated counters to their new home at once so a
            # crash before the next change cannot lose them.
            self._store.save_learning_state(self._learning_saved)
            self._store.save_state(self._state)

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
            hebbian=self._hebbian,
        )
        self._identity = IdentityOps(
            concepts=self.concepts, relations=self.relations
        )

    # ── Agent interface ───────────────────────────────────────────────

    @staticmethod
    def _open_store(store_path: str | Path, backend: str):
        choice = (backend or "auto").strip().lower()
        if choice == "auto":
            suffix = Path(store_path).suffix.lower()
            choice = "sqlite" if suffix in {".sqlite", ".sqlite3", ".db"} else "json"
        if choice == "sqlite":
            return SqliteStore(store_path)
        if choice == "json":
            return JsonStore(store_path)
        raise ValueError(f"unknown store backend: {backend!r} (use 'json', 'sqlite' or 'auto')")

    @property
    def clock(self) -> CognitiveClock:
        """The world's cognitive clock (one tick per observation)."""
        return self._clock

    def ingest(self, observation: Observation) -> IngestResult:
        """Agent submits observations. World 0 updates itself."""
        # Every observation is one unit of cognitive time.
        self._clock.advance()
        result = self._ingest_pipeline.run(observation)
        # Pipelines never persist — facade owns the flush boundary.
        self.concepts.flush()
        self.relations.flush()
        self._persist_learning_state()
        if (
            self._auto_reflect_every
            and self._clock.tick % self._auto_reflect_every == 0
        ):
            self.reflect(light=True)
        return result

    def _persist_learning_state(self, *, force: bool = False) -> None:
        """Save the clock (every observation) and the Hebbian counters.

        The counters are bulky at scale (up to ``MAX_PENDING_PAIRS`` pairs
        plus one mention count per concept) and serialising them was 76 %
        of ingest cost in a 2 000-concept world.  They are therefore
        written to the store's separate learning record: on every
        observation while small (``LEARNING_EAGER_LIMIT`` entries, so a
        small world stays restart-exact), otherwise at most every
        ``LEARNING_PERSIST_EVERY`` observations, and always at ``reflect()``
        and ``close()``.  Concepts and relations are flushed on every
        observation regardless; a crash can only lose a few observations'
        worth of co-occurrence *counters*.
        """
        if self._state.get("tick") != self._clock.tick:
            self._state["tick"] = self._clock.tick
            self._store.save_state(self._state)
        size = self._hebbian.pending_pairs + self._hebbian.tracked_concepts
        due = (
            force
            or size <= LEARNING_EAGER_LIMIT
            or self._clock.tick - self._learning_saved_tick >= LEARNING_PERSIST_EVERY
        )
        if not due:
            return
        learning = {
            "hebbian_pending": self._hebbian.snapshot(),
            "hebbian_stats": self._hebbian.stats_snapshot(),
        }
        if learning != self._learning_saved:
            self._store.save_learning_state(learning)
            self._learning_saved = learning
        self._learning_saved_tick = self._clock.tick

    def close(self) -> None:
        """Persist everything and release the store.

        Call this (or use the ``with`` form) before discarding a world so
        the amortised learning record is exact on disk.
        """
        self.concepts.flush()
        self.relations.flush()
        self._persist_learning_state(force=True)
        close = getattr(self._store, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "World":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

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
    ) -> Projection:
        """Generate a cognitive projection for the current task.

        ``perspective`` may be a ``Perspective`` or the name of a profile
        from ``world0.perspectives`` (``"dependency_map"``, ``"taxonomy"``,
        …); a bare ``task`` is applied to a named profile that has none.
        """
        if isinstance(perspective, str):
            perspective = get_perspective(perspective, task=task)
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
            activations,
            max_concepts=max_concepts,
            task=effective_task,
            seed_ids=seed_ids,
        )

    def reflect(self, *, light: bool = False) -> ReflectResult:
        """Cognitive consolidation — run after a task is complete.

        ``light=True`` applies decay, lifecycle and pruning only, skipping
        community detection and colour-field dynamics; it is what
        ``auto_reflect_every`` schedules between explicit reflects.
        """
        result = self._reflect_pipeline.run(light=light)
        self.concepts.flush()
        self.relations.flush()
        self._state["tick"] = self._clock.tick
        if not light:
            self._state["last_reflect"] = datetime.now(timezone.utc).isoformat()
            self._state["last_reflect_tick"] = self._clock.tick
            self._state["communities"] = self._communities.snapshot()
        self._store.save_state(self._state)
        self._persist_learning_state(force=True)
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
            cognitive_tick=self._clock.tick,
        )

    def visualize(
        self,
        output: str | Path | None = None,
        *,
        open_browser: bool = True,
    ) -> Path:
        """Generate an interactive HTML visualization of the concept network."""
        return _visualize(self, output=output, open_browser=open_browser)
