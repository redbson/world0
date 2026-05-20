"""PKMAgent — a World 0 concept-world agent.

This agent provides a task-facing interface to World 0.
Instead of folders and tags, understanding is organized through concepts,
typed relations, activation, and context-sensitive projection.

Supports two modes:
- **Direct mode**: command→response (learn, ask, explore, etc.)
- **Agentic mode**: LLM autonomously decides which tools to call,
  with session persistence and multi-provider routing.

Usage::

    from world0.agents import PKMAgent
    from world0.llm import AnthropicProvider

    agent = PKMAgent(
        store_path="~/.pkm_world",
        llm=AnthropicProvider(),
    )

    # Direct mode
    agent.learn("Transformers use self-attention mechanisms...")
    agent.ask("How does attention work?")

    # Agentic mode — LLM picks the tools
    response = agent.agent_chat("Tell me what I know about ML and find connections")

    # Session management
    agent.save_session()
    agent.resume_session("abc123")
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from world0.agents import research as research_utils
from world0.agents.failure import (
    FailureClass,
    FailureReport,
    RecoveryAction,
    classify_exception,
    turn_summary_failure_report,
)
from world0.agents.provider import create_provider, default_model_for_provider
from world0.agents.state import (
    AgentLifecycleStatus,
    AgentStateSnapshot,
    session_state_snapshot,
)
from world0.agents.session import ProjectionFeedbackEntry
from world0.llm.base import LLMError, LLMProvider
from world0.prompts import PromptRegistry, load_prompt_registry, prompt_config_path
from world0.schemas.relation import RelationType
from world0.schemas.types import Observation, Projection
from world0.world import World


class PKMAgent:
    """Personal Knowledge Management Agent built on World 0.

    Provides a high-level interface for learning, querying, exploring,
    and managing personal knowledge through a cognitive concept world.

    Two interaction modes:
    - Direct: call learn/ask/explore/connect/etc. methods directly
    - Agentic: call agent_chat() to let the LLM pick tools autonomously
    """

    def __init__(
        self,
        store_path: str | Path = ".pkm_world",
        llm: LLMProvider | None = None,
        *,
        space_id: str | None = None,
    ) -> None:
        root_path = Path(store_path).expanduser()
        self._root_path = root_path

        # Resolve the active space.  When a registry already exists or
        # ``space_id`` is explicit, PKMAgent routes the World + sessions
        # to that space's subdirectory.  Otherwise we fall back to the
        # legacy behaviour: the root itself is the store path.
        from world0.spaces import SpaceRegistry

        self._space_registry = SpaceRegistry(root_path)
        active = (
            self._space_registry.resolve(space_id)
            if space_id is not None
            else self._space_registry.active()
        )
        if active is None and self._space_registry.list():
            active = self._space_registry.list()[0]
        if active is not None:
            self._space_registry.set_active(active.id)
            self._space_registry.touch(active.id)
            store_path = self._space_registry.path_for(active.id)
        else:
            store_path = root_path

        self._space = active
        self._store_path = Path(store_path)
        prompt_paths = [prompt_config_path(root_path)]
        if self._store_path != root_path:
            prompt_paths.append(prompt_config_path(self._store_path))
        self._prompts = load_prompt_registry(*prompt_paths)
        self._world = World(
            store_path=self._store_path,
            llm=llm,
            prompt_registry=self._prompts,
        )
        self._llm = llm
        self._history: list[dict[str, str]] = []
        self._language = "en"
        self._runtime_settings: dict[str, Any] = {
            "language": "en",
            "provider": "none" if llm is None else llm.__class__.__name__.replace("Provider", "").lower(),
            "model": "",
            "api_key": "",
            "base_url": "",
            "azure_endpoint": "",
            "api_version": "2024-10-21",
            "auto_sediment_dialogue": True,
            "dialogue_sediment_interval": 1,
        }

        # Agentic components (lazy-initialized)
        self._tool_registry = None
        self._session_store = None
        self._current_session = None
        self._agent_loop = None
        self._chat_provider = None
        self._runtime_phase = AgentLifecycleStatus.BLOCKED if llm is None else None
        self._runtime_phase_reason = "No LLM provider configured." if llm is None else None
        self._current_task: str | None = None
        self._last_failure_report: FailureReport | None = None

        # MCP & Skill (lazy-initialized)
        self._mcp_manager = None
        self._skill_registry = None
        self._skill_executor = None

    @property
    def world(self) -> World:
        """Access the underlying World 0 instance."""
        return self._world

    @property
    def prompts(self) -> PromptRegistry:
        """Access the effective prompt registry."""
        return self._prompts

    @property
    def space(self):
        """Currently active ``Space`` (or ``None`` in legacy single-store mode)."""
        return self._space

    @property
    def space_registry(self):
        """Access the underlying ``SpaceRegistry``."""
        return self._space_registry

    # ── Agentic mode ─────────────────────────────────────────────────

    def _ensure_tools(self):
        """Lazy-initialize the tool registry."""
        if self._tool_registry is None:
            from world0.agents.tools.pkm_tools import build_pkm_tools
            self._tool_registry = build_pkm_tools(self)
        return self._tool_registry

    def _ensure_session_store(self):
        """Lazy-initialize the session store."""
        if self._session_store is None:
            from world0.agents.session import SessionStore
            self._session_store = SessionStore(self._store_path)
        return self._session_store

    def _ensure_session(self):
        """Lazy-initialize or return current session."""
        if self._current_session is None:
            from world0.agents.session import Session
            self._current_session = Session()
        return self._current_session

    @property
    def tool_registry(self):
        return self._ensure_tools()

    @property
    def session(self):
        return self._ensure_session()

    def init_agentic(
        self,
        model: str = "sonnet",
        api_key: str | None = None,
        base_url: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
    ) -> None:
        """Initialize the agentic mode with a ChatProvider.

        This enables agent_chat() with autonomous tool use.
        """
        from world0.agents.provider import ChatProvider
        self._chat_provider = ChatProvider(
            model=model,
            api_key=api_key,
            base_url=base_url,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        self._ensure_tools()
        self._ensure_session_store()
        self._ensure_session()

    @property
    def language(self) -> str:
        return self._language

    def runtime_settings(self) -> dict[str, Any]:
        settings = dict(self._runtime_settings)
        if self._chat_provider:
            settings["provider"] = self._chat_provider.provider_name
            settings["model"] = self._chat_provider.model
        settings["api_key_source"] = self._api_key_source(
            settings.get("provider", "none")
        )
        return settings

    def session_state(self, session=None):
        """Return a machine-readable state snapshot for a session."""
        target = session or self._ensure_session()
        return session_state_snapshot(target)

    def agent_state(self) -> AgentStateSnapshot:
        """Return a machine-readable runtime state for the agent."""
        session = self._ensure_session()
        session_state = self.session_state(session)
        llm_enabled = self._llm is not None
        agentic_ready = self._chat_provider is not None
        latest_turn = session.latest_turn_summary()
        degraded_sources: list[str] = []
        reason = self._runtime_phase_reason

        mcp_report = None
        if self._mcp_manager is not None:
            mcp_report = self._mcp_manager.health()

        if self._runtime_phase is not None:
            status = self._runtime_phase
        elif not llm_enabled:
            status = AgentLifecycleStatus.BLOCKED
            reason = "No LLM provider configured."
        elif not agentic_ready:
            status = AgentLifecycleStatus.BLOCKED
            reason = (
                "Agentic mode unavailable. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
                "AZURE_OPENAI_API_KEY, or AZURE_OPENAI_KEY."
            )
        elif latest_turn and latest_turn.failure_class != "none":
            status = AgentLifecycleStatus.DEGRADED
            reason = f"Latest turn ended with {latest_turn.failure_class}."
            degraded_sources.append("latest_turn")
        elif self._last_failure_report is not None:
            status = AgentLifecycleStatus.DEGRADED
            reason = self._last_failure_report.message
            degraded_sources.append("runtime_failure")
        elif mcp_report and mcp_report.failed:
            status = AgentLifecycleStatus.DEGRADED
            reason = f"MCP servers unavailable: {', '.join(mcp_report.failed)}."
            degraded_sources.append("mcp")
        else:
            status = AgentLifecycleStatus.READY
            reason = None

        if session_state.status.value == "attention_needed" and "latest_turn" not in degraded_sources:
            degraded_sources.append("session")

        provider = self._chat_provider.provider_name if self._chat_provider else None
        model = self._chat_provider.model if self._chat_provider else None
        return AgentStateSnapshot(
            status=status,
            reason=reason,
            agentic_ready=agentic_ready,
            llm_enabled=llm_enabled,
            provider=provider,
            model=model,
            session_id=session.id,
            session_message_count=session.message_count(),
            turn_count=len(session.turn_summaries),
            latest_failure_class=latest_turn.failure_class if latest_turn else "none",
            failed_tools=latest_turn.failed_tools if latest_turn else [],
            has_compaction=session.compaction is not None,
            open_loops=session.compaction.open_loops if session.compaction else [],
            mcp_total_servers=mcp_report.total_servers if mcp_report else 0,
            mcp_connected_servers=len(mcp_report.connected) if mcp_report else 0,
            mcp_failed_servers=mcp_report.failed if mcp_report else [],
            degraded_sources=degraded_sources,
            current_task=self._current_task,
        )

    def _set_runtime_phase(
        self,
        status: AgentLifecycleStatus | None,
        *,
        reason: str | None = None,
        current_task: str | None = None,
    ) -> None:
        self._runtime_phase = status
        self._runtime_phase_reason = reason
        self._current_task = current_task

    def latest_failure(self) -> FailureReport | None:
        """Return the most relevant structured failure for the current session/runtime."""
        latest_turn = self.session.latest_turn_summary()
        turn_report = turn_summary_failure_report(latest_turn) if latest_turn else None
        if turn_report is not None:
            return turn_report
        return self._last_failure_report

    def _remember_failure(self, report: FailureReport | None) -> None:
        self._last_failure_report = report

    def _clear_runtime_failure(self) -> None:
        self._last_failure_report = None

    @staticmethod
    def _api_key_source(provider: str) -> str:
        if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            return "env"
        if provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            return "env"
        if provider == "azure-openai" and (
            os.environ.get("AZURE_OPENAI_API_KEY")
            or os.environ.get("AZURE_OPENAI_KEY")
        ):
            return "env"
        return "explicit" if provider != "none" else "none"

    def configure_runtime(
        self,
        *,
        language: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        auto_sediment_dialogue: bool | None = None,
        dialogue_sediment_interval: int | None = None,
    ) -> None:
        """Update language and LLM runtime settings."""
        if language:
            self._language = language
            self._runtime_settings["language"] = language
        if auto_sediment_dialogue is not None:
            self._runtime_settings["auto_sediment_dialogue"] = bool(
                auto_sediment_dialogue
            )
        if dialogue_sediment_interval is not None:
            interval = int(dialogue_sediment_interval)
            if interval < 1 or interval > 20:
                raise ValueError(
                    "dialogue_sediment_interval must be between 1 and 20."
                )
            self._runtime_settings["dialogue_sediment_interval"] = interval

        provider_name = provider or self._runtime_settings.get("provider", "none")
        fallback_model = (
            default_model_for_provider(provider_name)
            if provider_name != "none"
            else ""
        )
        chosen_model = model or self._runtime_settings.get("model") or fallback_model
        chosen_api_key = api_key if api_key is not None else self._runtime_settings.get("api_key", "")
        chosen_base_url = base_url if base_url is not None else self._runtime_settings.get("base_url", "")
        chosen_azure_endpoint = (
            azure_endpoint if azure_endpoint is not None
            else self._runtime_settings.get("azure_endpoint", "")
        )
        chosen_api_version = (
            api_version if api_version is not None
            else self._runtime_settings.get("api_version", "2024-10-21")
        )

        self._runtime_settings.update({
            "provider": provider_name,
            "model": chosen_model,
            "api_key": chosen_api_key,
            "base_url": chosen_base_url,
            "azure_endpoint": chosen_azure_endpoint,
            "api_version": chosen_api_version,
        })

        if provider_name == "none":
            self._llm = None
            self._world.set_llm(None)
            self._chat_provider = None
            self._set_runtime_phase(
                AgentLifecycleStatus.BLOCKED,
                reason="No LLM provider configured.",
            )
            return

        provider_model = chosen_model
        if provider_name in ("openai", "anthropic", "azure-openai"):
            provider_model = f"{provider_name}/{chosen_model}"

        self._llm = create_provider(
            model=provider_model,
            api_key=chosen_api_key or None,
            base_url=chosen_base_url or None,
            azure_endpoint=chosen_azure_endpoint or None,
            api_version=chosen_api_version or None,
        )
        self._world.set_llm(self._llm)
        self.init_agentic(
            model=provider_model,
            api_key=chosen_api_key or None,
            base_url=chosen_base_url or None,
            azure_endpoint=chosen_azure_endpoint or None,
            api_version=chosen_api_version or None,
        )
        self._set_runtime_phase(None)
        self._clear_runtime_failure()

    def agent_chat(
        self,
        user_input: str,
        *,
        on_tool_call=None,
        on_tool_result=None,
    ) -> str:
        """Agentic chat — LLM autonomously decides which tools to call.

        Requires init_agentic() to be called first.

        Returns the agent's final text response.
        """
        if not self._chat_provider:
            raise RuntimeError(
                "Agentic mode not initialized. Call init_agentic() first."
            )

        from world0.agents.loop import AgentLoop
        self._prepare_session_for_agentic()
        self._set_runtime_phase(
            AgentLifecycleStatus.RUNNING,
            reason="Agent turn in progress.",
            current_task=user_input,
        )

        try:
            loop = AgentLoop(
                self._chat_provider,
                self._ensure_tools(),
                self._ensure_session(),
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
                language=self._language,
                prompt_registry=self._prompts,
            )
            self._agent_loop = loop
            result = loop.run(user_input)
            self._auto_sediment_agent_turn(user_input, result)
            self._clear_runtime_failure()
            return result
        except Exception as exc:
            self._remember_failure(classify_exception(exc, context="llm"))
            self._set_runtime_phase(
                AgentLifecycleStatus.FAILED,
                reason=str(exc),
                current_task=user_input,
            )
            raise
        finally:
            if self._runtime_phase in (
                AgentLifecycleStatus.RUNNING,
                AgentLifecycleStatus.RECOVERING,
            ):
                self._set_runtime_phase(None)

    # ── Session management ───────────────────────────────────────────

    def save_session(self) -> str:
        """Save the current session. Returns session ID."""
        store = self._ensure_session_store()
        session = self._ensure_session()

        # Auto-title from first user message
        if not session.title:
            for msg in session.messages:
                if msg.role == "user":
                    session.title = msg.content[:60]
                    break

        store.save(session)
        return session.id

    def resume_session(self, session_id: str) -> bool:
        """Resume a previous session by ID. Use 'latest' for most recent."""
        store = self._ensure_session_store()
        if session_id == "latest":
            session = store.load_latest()
        else:
            session = store.load(session_id)

        if session:
            self._current_session = session
            return True
        return False

    def list_sessions(self, limit: int = 20) -> list[str]:
        """List available sessions as summary strings."""
        store = self._ensure_session_store()
        return [s.summary() for s in store.list_sessions(limit)]

    def list_session_summaries(self, limit: int = 20) -> list[dict[str, Any]]:
        """List available sessions with structured metadata."""
        store = self._ensure_session_store()
        sessions = store.list_sessions(limit)
        return [
            {
                "id": s.id,
                "title": s.title or "Untitled",
                "summary": s.summary(),
                "message_count": s.message_count(),
                "updated_at": s.updated_at.isoformat(),
                "created_at": s.created_at.isoformat(),
                "state": self.session_state(s).model_dump(mode="json"),
            }
            for s in sessions
        ]

    def rename_session(self, title: str, session_id: str | None = None) -> str:
        """Rename the current or a saved session."""
        clean = title.strip()
        if not clean:
            raise ValueError("Session title cannot be empty.")
        target = self.get_session(session_id) if session_id else self._ensure_session()
        if target is None:
            raise ValueError(f"Session '{session_id}' not found.")
        target.title = clean
        self._ensure_session_store().save(target)
        if self._current_session and self._current_session.id == target.id:
            self._current_session = target
        return target.id

    def compact_session(self, session_id: str | None = None) -> dict[str, Any]:
        """Manually compact a session and return the resulting metadata."""
        target = self.get_session(session_id) if session_id else self._ensure_session()
        if target is None:
            raise ValueError(f"Session '{session_id}' not found.")
        before = target.compaction.covered_messages if target.compaction else 0
        self._compact_session(target)
        after = target.compaction.covered_messages if target.compaction else 0
        self._ensure_session_store().save(target)
        if self._current_session and self._current_session.id == target.id:
            self._current_session = target
        return {
            "session_id": target.id,
            "compacted": after > before,
            "covered_messages": after,
            "state": self.session_state(target).model_dump(mode="json"),
        }

    def latest_projection_feedback(self) -> ProjectionFeedbackEntry | None:
        """Return the latest recorded projection feedback for the current session."""
        return self.session.latest_projection_feedback()

    def latest_projection_snapshot(self) -> dict[str, Any] | None:
        """Return the last projection snapshot captured during ask()."""
        snapshot = self.session.metadata.get("last_projection")
        return snapshot if isinstance(snapshot, dict) else None

    def apply_projection_feedback(
        self,
        *,
        useful: bool | None = None,
        missing_concepts: list[str] | None = None,
        noisy_concepts: list[str] | None = None,
        weak_relations: list[str] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """Record and apply feedback about the latest projection."""
        snapshot = self.latest_projection_snapshot()
        if not snapshot:
            raise ValueError("No recent projection is available for feedback.")

        missing_concepts = [item.strip() for item in (missing_concepts or []) if item.strip()]
        noisy_concepts = [item.strip() for item in (noisy_concepts or []) if item.strip()]
        weak_relations = [item.strip() for item in (weak_relations or []) if item.strip()]

        created_missing: list[str] = []
        reinforced_missing: list[str] = []
        adjusted_noisy: list[str] = []
        adjusted_relations: list[str] = []

        for name in missing_concepts:
            node, is_new = self._world.concepts.get_or_create(
                name,
                origin="projection_feedback",
                task=f"feedback:{snapshot.get('query', '')}".strip(":"),
            )
            self._world.concepts.reinforce(
                node.id,
                source="projection_feedback",
                task=snapshot.get("query", ""),
            )
            if is_new:
                created_missing.append(node.name)
            else:
                reinforced_missing.append(node.name)

        projection_concepts = {
            item["name"].strip().lower(): item["id"]
            for item in snapshot.get("concepts", [])
            if item.get("name") and item.get("id")
        }
        for name in noisy_concepts:
            cid = projection_concepts.get(name.strip().lower())
            if not cid:
                node = self._world.concepts.resolve(name)
                cid = node.id if node else None
            if cid:
                adjusted = self._world.concepts.adjust_confidence(cid, -0.05)
                if adjusted:
                    adjusted_noisy.append(adjusted.name)

        projection_relations = {
            item["label"].strip().lower(): item["id"]
            for item in snapshot.get("relations", [])
            if item.get("label") and item.get("id")
        }
        for label in weak_relations:
            rid = projection_relations.get(label.strip().lower())
            if not rid:
                rid = self._resolve_relation_feedback_label(label)
            if rid:
                adjusted = self._world.relations.adjust_strength(
                    rid,
                    weight_delta=-0.05,
                    confidence_delta=-0.05,
                )
                if adjusted:
                    adjusted_relations.append(label)

        if useful is True:
            for item in snapshot.get("concepts", [])[:5]:
                cid = item.get("id")
                if cid:
                    self._world.concepts.reinforce(
                        cid,
                        source="projection_feedback",
                        task=snapshot.get("query", ""),
                    )
            for item in snapshot.get("relations", [])[:5]:
                rid = item.get("id")
                if rid:
                    self._world.relations.reinforce(
                        rid,
                        provenance=f"projection_feedback:{snapshot.get('query', '')}",
                    )

        self._world.concepts.flush()
        self._world.relations.flush()

        feedback = self.session.add_projection_feedback(ProjectionFeedbackEntry(
            query=str(snapshot.get("query", "")),
            useful=useful,
            missing_concepts=missing_concepts,
            noisy_concepts=adjusted_noisy,
            weak_relations=adjusted_relations,
            notes=notes.strip(),
            concept_names=[item.get("name", "") for item in snapshot.get("concepts", []) if item.get("name")],
            relation_labels=[item.get("label", "") for item in snapshot.get("relations", []) if item.get("label")],
        ))
        self.save_session()
        return {
            "feedback": feedback.model_dump(mode="json"),
            "created_missing_concepts": created_missing,
            "reinforced_missing_concepts": reinforced_missing,
            "demoted_noisy_concepts": adjusted_noisy,
            "weakened_relations": adjusted_relations,
        }

    def get_session(self, session_id: str):
        """Load a session by id. Use 'latest' for the most recent session."""
        store = self._ensure_session_store()
        if session_id == "latest":
            return store.load_latest()
        return store.load(session_id)

    def new_session(self) -> str:
        """Start a new session (saves current if exists). Returns new session ID."""
        if self._current_session and self._current_session.messages:
            self.save_session()
        from world0.agents.session import Session
        self._current_session = Session()
        return self._current_session.id

    def record_direct_turn(
        self,
        user_input: str,
        assistant_output: str,
        *,
        mode: str,
        save: bool = True,
        auto_sediment: bool = False,
    ) -> str:
        """Record a non-agentic UI interaction into the current session."""
        session = self._ensure_session()
        session.add_message("user", user_input, mode=mode)
        session.add_message("assistant", assistant_output, mode=mode)
        if auto_sediment:
            self._auto_sediment_dialogue_turn(
                user_input,
                assistant_output,
                mode=mode,
            )
        if save:
            return self.save_session()
        return session.id

    def latest_dialogue_sediment(self) -> dict[str, Any] | None:
        """Return the latest automatic dialogue sedimentation snapshot."""
        return self.session.metadata.get("last_dialogue_sediment")

    def sediment_dialogue_turn(
        self,
        user_input: str,
        assistant_output: str,
        *,
        mode: str = "chat",
        task: str = "",
        source: str = "",
    ) -> dict[str, Any]:
        """Extract concepts from one dialogue turn and ingest them into World 0."""
        session = self._ensure_session()
        event = {
            "status": "skipped",
            "mode": mode,
            "task": task or self._dialogue_task_label(user_input, mode=mode),
            "source": source,
            "reason": "",
            "pending_turns": 0,
            "required_turns": 1,
            "new_concepts": [],
            "reinforced_concepts": [],
            "new_relations": [],
            "reinforced_relations": [],
            "hebbian_relations": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if not self._runtime_settings.get("auto_sediment_dialogue", True):
            event["reason"] = "Automatic dialogue sedimentation is disabled."
            session.metadata["last_dialogue_sediment"] = event
            return event

        if not self._llm:
            event["reason"] = "No LLM provider configured for dialogue sedimentation."
            session.metadata["last_dialogue_sediment"] = event
            return event

        if not user_input.strip() and not assistant_output.strip():
            event["reason"] = "Dialogue turn is empty."
            session.metadata["last_dialogue_sediment"] = event
            return event

        task_label = task or self._dialogue_task_label(user_input, mode=mode)
        source_label = source or self._dialogue_source_label(session, mode=mode)
        dialogue_text = self._render_dialogue_for_learning(
            user_input,
            assistant_output,
            mode=mode,
        )

        return self._ingest_dialogue_text(
            dialogue_text,
            mode=mode,
            task=task_label,
            source=source_label,
            pending_turns=1,
            required_turns=1,
        )

    def _ingest_dialogue_text(
        self,
        dialogue_text: str,
        *,
        mode: str,
        task: str,
        source: str,
        pending_turns: int,
        required_turns: int,
    ) -> dict[str, Any]:
        """Ingest rendered dialogue text into World 0."""
        session = self._ensure_session()
        event = {
            "status": "skipped",
            "mode": mode,
            "task": task,
            "source": source,
            "reason": "",
            "pending_turns": pending_turns,
            "required_turns": required_turns,
            "new_concepts": [],
            "reinforced_concepts": [],
            "new_relations": [],
            "reinforced_relations": [],
            "hebbian_relations": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            result = self._world.ingest_text(
                dialogue_text,
                task=task,
                source=source,
            )
        except RuntimeError as exc:
            event["reason"] = str(exc)
            session.metadata["last_dialogue_sediment"] = event
            return event

        event.update({
            "status": "ingested",
            "new_concepts": result.new_concepts,
            "reinforced_concepts": result.reinforced_concepts,
            "new_relations": result.new_relations,
            "reinforced_relations": result.reinforced_relations,
            "hebbian_relations": result.hebbian_relations,
        })
        session.metadata["last_dialogue_sediment"] = event
        return event

    # ── MCP integration ──────────────────────────────────────────────

    def _ensure_mcp(self):
        """Lazy-initialize the MCP manager."""
        if self._mcp_manager is None:
            from world0.agents.mcp.manager import McpManager
            self._mcp_manager = McpManager(self._ensure_tools())
        return self._mcp_manager

    @property
    def mcp(self):
        """Access the MCP manager."""
        return self._ensure_mcp()

    def load_mcp_config(self, config_path: str | None = None) -> str:
        """Load MCP server configurations and connect.

        If no path given, looks for mcp.json in the store directory.
        Returns a status summary.
        """
        manager = self._ensure_mcp()
        path = config_path or str(self._store_path / "mcp.json")

        count = manager.load_config(path)
        if count == 0:
            return f"No MCP servers found in {path}"

        report = manager.connect_all()
        return report.summary()

    def add_mcp_server(
        self, name: str, command: str, args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        """Add and connect a single MCP server."""
        from world0.agents.mcp.client import McpServerConfig
        manager = self._ensure_mcp()
        config = McpServerConfig(
            name=name, command=command,
            args=args or [], env=env or {},
        )
        manager.add_server(config)
        ok = manager.connect_server(name)
        if ok:
            client = manager.get_client(name)
            tool_count = len(client.tools) if client else 0
            return f"Connected to {name}: {tool_count} tools available"
        return f"Failed to connect to {name}"

    def mcp_status(self) -> str:
        """Get MCP server status summary."""
        manager = self._ensure_mcp()
        statuses = manager.server_statuses()
        if not statuses:
            return "No MCP servers configured."
        lines = ["## MCP Servers", ""]
        for s in statuses:
            icon = "●" if s.status == "connected" else "○"
            err = f" — {s.error}" if s.error else ""
            lines.append(
                f"- {icon} **{s.name}** ({s.status}) "
                f"tools: {s.tool_count}, resources: {s.resource_count}{err}"
            )
        return "\n".join(lines)

    # ── Skill system ─────────────────────────────────────────────────

    def _ensure_skills(self):
        """Lazy-initialize the skill registry with built-in skills."""
        if self._skill_registry is None:
            from world0.agents.skill import SkillRegistry, register_builtin_skills
            self._skill_registry = SkillRegistry()
            register_builtin_skills(self._skill_registry, self._prompts)
            # Load custom skills if available
            custom_path = self._store_path / "skills.json"
            if custom_path.exists():
                self._skill_registry.load_from_file(str(custom_path))
        return self._skill_registry

    def _ensure_skill_executor(self):
        if self._skill_executor is None:
            from world0.agents.skill import SkillExecutor
            self._skill_executor = SkillExecutor(self)
        return self._skill_executor

    @property
    def skills(self):
        """Access the skill registry."""
        return self._ensure_skills()

    def run_skill(
        self,
        skill_name: str,
        on_tool_call=None,
        on_tool_result=None,
        **kwargs,
    ) -> str:
        """Execute a skill by name with the given parameters.

        Requires init_agentic() to be called first.
        """
        registry = self._ensure_skills()
        executor = self._ensure_skill_executor()
        return executor.execute_by_name(
            skill_name, registry,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            **kwargs,
        )

    def list_skills(self) -> str:
        """List all available skills."""
        registry = self._ensure_skills()
        skills = registry.all()
        if not skills:
            return "No skills available."
        lines = ["## Available Skills", ""]
        for s in skills:
            params = ", ".join(
                f"`{p.name}`" + ("*" if p.required else "")
                for p in s.parameters
            )
            param_str = f" ({params})" if params else ""
            lines.append(f"- **{s.name}**{param_str}: {s.description}")
        return "\n".join(lines)

    # ── Core operations ───────────────────────────────────────────────

    def learn(
        self,
        text: str,
        *,
        task: str = "knowledge intake",
        source: str = "",
    ) -> str:
        """Ingest knowledge from text into the concept world.

        Uses LLM-powered extraction if available, otherwise requires
        structured Observation input via learn_structured().

        Returns a human-readable summary of what was learned.
        """
        if not text.strip():
            return "Nothing to learn — empty input."

        if not source:
            source = f"pkm_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        result = self._world.ingest_text(text, task=task, source=source)

        summary_parts = []
        if result.new_concepts:
            summary_parts.append(
                f"New concepts: {', '.join(result.new_concepts)}"
            )
        if result.reinforced_concepts:
            summary_parts.append(
                f"Reinforced: {', '.join(result.reinforced_concepts)}"
            )
        if result.new_relations:
            summary_parts.append(
                f"New relations: {', '.join(result.new_relations)}"
            )
        if result.reinforced_relations:
            summary_parts.append(
                f"Reinforced relations: {', '.join(result.reinforced_relations)}"
            )
        if result.hebbian_relations:
            summary_parts.append(
                f"Co-occurrence links: {', '.join(result.hebbian_relations)}"
            )

        if not summary_parts:
            return "Text processed but no concepts extracted."

        summary = "\n".join(summary_parts)

        # Generate LLM summary if available
        if self._llm:
            try:
                llm_summary = self._llm.complete_json(
                    self._prompts.render(
                        "agent.learn_inline_summary.system",
                        language_instruction=self._language_instruction(),
                    ),
                    f"Summarize what was learned:\n{summary}",
                )
                parsed = json.loads(self._extract_json(llm_summary))
                if "summary" in parsed:
                    return f"{parsed['summary']}\n\n{summary}"
            except (LLMError, json.JSONDecodeError, KeyError):
                pass

        return summary

    def learn_structured(self, observation: Observation) -> str:
        """Ingest a pre-structured Observation (no LLM needed)."""
        result = self._world.ingest(observation)

        parts = []
        if result.new_concepts:
            parts.append(f"New: {', '.join(result.new_concepts)}")
        if result.reinforced_concepts:
            parts.append(f"Reinforced: {', '.join(result.reinforced_concepts)}")
        if result.new_relations:
            parts.append(f"Relations: {', '.join(result.new_relations)}")
        return " | ".join(parts) if parts else "No changes."

    def research_topic(
        self,
        topic: str,
        *,
        focus: str = "",
        max_sources: int = 4,
        save_findings: bool = True,
    ) -> str:
        """Run a compact web research workflow and ground it in World 0."""
        topic = topic.strip()
        focus = focus.strip()
        if not topic:
            return "Please provide a research topic."
        if not self._llm:
            return (
                "Research mode requires an LLM provider. "
                "Configure OpenAI, Anthropic, or Azure AI first."
            )

        source_limit = max(1, min(int(max_sources), 8))
        search_query = topic if not focus else f"{topic} {focus}"
        results, recovery_notes, failure = self._search_web_with_recovery(
            query=topic,
            focus=focus,
            max_results=source_limit,
            domains=None,
        )

        if not results:
            self._remember_failure(
                failure or FailureReport(
                    failure_class=FailureClass.SEARCH_FETCH_FAILED,
                    message=f"No web results found for '{search_query}'.",
                    retryable=True,
                    context="search",
                    recovery_actions=[
                        RecoveryAction.RETRY_WITHOUT_FOCUS,
                        RecoveryAction.RETRY_WITHOUT_DOMAINS,
                    ],
                )
            )
            return f"No web results found for '{search_query}'."

        source_notes: list[dict[str, Any]] = []
        learning_log: list[str] = []
        skipped_sources: list[str] = []
        last_fetch_failure: FailureReport | None = None

        for result in results[:source_limit]:
            try:
                doc = research_utils.fetch_web_document(result.url)
            except Exception as exc:
                last_fetch_failure = classify_exception(exc, context="fetch")
                skipped_sources.append(
                    f"- [{result.title}]({result.url}) — {last_fetch_failure.message}"
                )
                continue
            if not doc.text.strip():
                skipped_sources.append(
                    f"- [{result.title}]({result.url}) — Empty source text after fetch."
                )
                continue

            note = self._distill_research_source(
                topic=topic,
                focus=focus,
                title=doc.title or result.title,
                url=result.url,
                snippet=result.snippet,
                text=doc.text,
            )
            note["title"] = doc.title or result.title
            note["url"] = result.url
            source_notes.append(note)

            if save_findings:
                ingest_text = "\n".join([
                    f"Title: {doc.title or result.title}",
                    f"URL: {result.url}",
                    f"Search snippet: {result.snippet}",
                    "",
                    doc.text[:6000],
                ])
                learn_result = self.learn(
                    ingest_text,
                    task=f"research: {topic}",
                    source=result.url,
                )
                learning_log.append(
                    f"- [{doc.title or result.title}]({result.url}) — "
                    f"{learn_result.splitlines()[0]}"
                )

        if not source_notes:
            self._remember_failure(
                last_fetch_failure or FailureReport(
                    failure_class=FailureClass.SEARCH_FETCH_FAILED,
                    message=(
                        f"I found search results for '{search_query}', but couldn't extract readable source text."
                    ),
                    retryable=True,
                    context="fetch",
                    recovery_actions=[RecoveryAction.SKIP_SOURCE],
                )
            )
            return (
                f"I found search results for '{search_query}', but couldn't extract readable source text."
            )

        self._clear_runtime_failure()

        brief = self._compose_research_brief(
            topic=topic,
            focus=focus,
            source_notes=source_notes,
        )
        projection_query = topic if not focus else f"{topic} {focus}"
        projection = self.ask(projection_query, max_concepts=18, max_depth=2)

        lines = [
            "## Research Brief",
            "",
            f"**Topic:** {topic}",
        ]
        if focus:
            lines.append(f"**Focus:** {focus}")
        lines.extend([
            f"**Sources reviewed:** {len(source_notes)}",
            "",
            f"**Summary:** {brief['summary']}",
            "",
            "### Sources",
        ])
        for note in source_notes:
            lines.append(
                f"- [{note['title']}]({note['url']}) — {note['summary']}"
            )

        if brief["findings"]:
            lines.extend(["", "### Findings"])
            lines.extend(f"- {item}" for item in brief["findings"])

        if brief["gaps"]:
            lines.extend(["", "### Gaps"])
            lines.extend(f"- {item}" for item in brief["gaps"])

        if brief["next_steps"]:
            lines.extend(["", "### Next Steps"])
            lines.extend(f"- {item}" for item in brief["next_steps"])

        if recovery_notes or skipped_sources:
            lines.extend(["", "### Recovery Notes"])
            lines.extend(f"- {item}" for item in recovery_notes)
            lines.extend(skipped_sources[:4])

        if learning_log:
            lines.extend(["", "### World 0 Update"])
            lines.extend(learning_log)

        lines.extend(["", "### Projection Into World 0", "", projection])
        return "\n".join(lines)

    def search_web(
        self,
        query: str,
        *,
        focus: str = "",
        max_results: int = 5,
        domains: str | list[str] | tuple[str, ...] | None = None,
        fetch_pages: bool = False,
    ) -> str:
        """Search the public web and optionally fetch top pages for quick review."""
        query = query.strip()
        focus = focus.strip()
        if not query:
            return "Please provide a search query."

        result_limit = max(1, min(int(max_results), 10))
        domain_filters = self._parse_domain_filters(domains)
        search_query = query if not focus else f"{query} {focus}"
        results, recovery_notes, failure = self._search_web_with_recovery(
            query=query,
            focus=focus,
            max_results=result_limit,
            domains=domain_filters,
        )

        if not results:
            self._remember_failure(
                failure or FailureReport(
                    failure_class=FailureClass.SEARCH_FETCH_FAILED,
                    message=f"No web results found for '{search_query}'.",
                    retryable=True,
                    context="search",
                    recovery_actions=[
                        RecoveryAction.RETRY_WITHOUT_FOCUS,
                        RecoveryAction.RETRY_WITHOUT_DOMAINS,
                    ],
                )
            )
            scoped = ""
            if domain_filters:
                scoped = f" within {', '.join(domain_filters)}"
            return f"No web results found for '{search_query}'{scoped}."

        self._clear_runtime_failure()

        lines = [
            f"## Web Search: {query}",
            "",
        ]
        if focus:
            lines.append(f"**Focus:** {focus}")
        if domain_filters:
            lines.append(f"**Domains:** {', '.join(domain_filters)}")
        lines.extend([
            f"**Results:** {len(results)}",
            "",
            "### Results",
        ])
        if recovery_notes:
            lines.extend(["### Recovery"])
            lines.extend(f"- {item}" for item in recovery_notes)
            lines.append("")

        for idx, item in enumerate(results, 1):
            meta = f" ({item.domain})" if item.domain else ""
            snippet = f" — {item.snippet}" if item.snippet else ""
            lines.append(f"{idx}. [{item.title}]({item.url}){meta}{snippet}")

        if fetch_pages:
            fetched = self._fetch_search_result_notes(results)
            if fetched:
                brief = self._compose_search_brief(
                    query=query,
                    focus=focus,
                    results=results,
                    fetched=fetched,
                )
                if brief:
                    lines.extend([
                        "",
                        "### Search Brief",
                        f"- Summary: {brief['summary']}",
                    ])
                    if brief["themes"]:
                        lines.extend(f"- Theme: {item}" for item in brief["themes"])
                    if brief["recommended_sources"]:
                        lines.extend(
                            f"- Start with: {item}" for item in brief["recommended_sources"]
                        )

                lines.extend(["", "### Source Glimpses"])
                for note in fetched:
                    lines.extend([
                        f"- **{note['title']}**",
                        f"  URL: {note['url']}",
                        f"  Glimpse: {note['excerpt']}",
                    ])

        return "\n".join(lines)

    def ask(
        self,
        query: str,
        *,
        max_concepts: int = 15,
        max_depth: int = 2,
    ) -> str:
        """Ask a question — get a cognitive projection and LLM answer.

        1. Extract seed concepts from the query
        2. Activate and project from the concept world
        3. Use LLM to synthesize an answer from the projection

        If no LLM is available, returns the raw projection render.
        """
        if not query.strip():
            return "Please provide a question."

        seeds = self._extract_seeds(query)
        if not seeds:
            return (
                "I couldn't identify relevant concepts in your question. "
                "Try using more specific terms, or check what concepts exist "
                "with the `status` command."
            )

        projection = self._world.project(
            seeds, task=query, max_concepts=max_concepts, max_depth=max_depth
        )

        if not projection.concepts:
            # Try broader search with individual words
            words = [w.lower().strip() for w in query.split() if len(w) > 3]
            if words:
                projection = self._world.project(
                    words, task=query, max_concepts=max_concepts, max_depth=max_depth
                )

        if not projection.concepts:
            return (
                f"No concepts found for: {', '.join(seeds)}.\n"
                "Your concept world may not have knowledge about this topic yet. "
                "Use `learn` to add relevant knowledge first."
            )

        rendered = projection.render()
        basis = self._render_projection_basis(projection, seeds)
        self._remember_projection_snapshot(query, seeds, projection)

        if not self._llm:
            return f"{rendered}\n\n{basis}"

        # Generate answer using LLM + projection context
        try:
            user_prompt = (
                f"## Cognitive Projection\n{rendered}\n\n"
                f"## User Question\n{query}"
            )
            response = self._llm.complete_json(
                self._prompt_with_language("agent.answer.system"),
                user_prompt,
            )
            # The response here is plain text, not necessarily JSON
            return f"{response}\n\n---\n{basis}"
        except LLMError:
            return f"{rendered}\n\n{basis}"

    def explore(self, concept_name: str) -> str:
        """Deep dive into a single concept and its neighborhood.

        Returns a detailed view of the concept, its relations, and
        connected concepts.
        """
        node = self._world.concepts.resolve(concept_name)
        if not node:
            return f"Concept '{concept_name}' not found."

        lines = [
            f"# {node.name}",
            f"",
            f"**Maturity:** {node.maturity.value}",
            f"**Confidence:** {node.confidence:.2f}",
            f"**Activated:** {node.activation_count} times",
            f"**Last active:** {node.last_activated.strftime('%Y-%m-%d %H:%M')} UTC",
            f"**Origin:** {node.origin or 'unknown'}",
        ]

        if node.description:
            lines.extend(["", f"**Description:** {node.description}"])

        if node.aliases:
            lines.extend(["", f"**Aliases:** {', '.join(node.aliases)}"])

        if node.tags:
            lines.extend(["", f"**Tags:** {', '.join(node.tags)}"])

        # Relations
        relations = self._world.relations.for_concept(node.id)
        if relations:
            lines.extend(["", "## Relations", ""])
            for rel in sorted(relations, key=lambda r: r.weight, reverse=True):
                other_id = rel.other_end(node.id)
                if not other_id:
                    continue
                other = self._world.concepts.get(other_id)
                if not other:
                    continue

                direction = "→" if rel.source_id == node.id else "←"
                lines.append(
                    f"- {direction} **{rel.relation_type.value}** → "
                    f"{other.name} (weight: {rel.weight:.2f}, "
                    f"reinforced {rel.reinforcement_count}x)"
                )

        # Recent reinforcement log
        if node.reinforcement_log:
            lines.extend(["", "## Recent Activity", ""])
            for entry in node.reinforcement_log[-5:]:
                task_str = f" [{entry.task}]" if entry.task else ""
                lines.append(
                    f"- {entry.timestamp.strftime('%Y-%m-%d %H:%M')}{task_str}"
                )

        return "\n".join(lines)

    def concept_card(self, concept_name: str) -> dict[str, Any] | None:
        """Return a structured concept card for UI inspection."""
        node = self._world.concepts.resolve(concept_name)
        if not node:
            return None

        relations = self._world.relations.for_concept(node.id)
        relation_cards: list[dict[str, Any]] = []
        related_names: list[str] = []
        for rel in sorted(relations, key=lambda r: r.weight, reverse=True):
            other_id = rel.other_end(node.id)
            if not other_id:
                continue
            other = self._world.concepts.get(other_id)
            if not other:
                continue
            direction = "outgoing" if rel.source_id == node.id else "incoming"
            relation_cards.append({
                "relation_type": rel.relation_type.value,
                "other_name": other.name,
                "other_id": other.id,
                "direction": direction,
                "weight": round(rel.weight, 4),
                "confidence": round(rel.confidence, 4),
                "reinforcement_count": rel.reinforcement_count,
                "is_explicit": rel.is_explicit,
                "provenance": rel.provenance,
                "task_history": list(rel.task_history),
                "last_reinforced": rel.last_reinforced.isoformat(),
            })
            related_names.append(other.name)

        recent_activity = [
            {
                "timestamp": entry.timestamp.isoformat(),
                "source": entry.source,
                "task": entry.task,
            }
            for entry in node.reinforcement_log[-8:]
        ]
        sources = sorted({entry.source for entry in node.reinforcement_log if entry.source})
        tasks = sorted({entry.task for entry in node.reinforcement_log if entry.task})

        return {
            "id": node.id,
            "name": node.name,
            "description": node.description,
            "aliases": list(node.aliases),
            "domain": node.domain,
            "tags": list(node.tags),
            "maturity": node.maturity.value,
            "confidence": round(node.confidence, 4),
            "activation_count": node.activation_count,
            "origin": node.origin,
            "created_at": node.created_at.isoformat(),
            "last_activated": node.last_activated.isoformat(),
            "relation_count": len(relation_cards),
            "related_names": related_names[:12],
            "relations": relation_cards[:24],
            "sources": sources[:12],
            "tasks": tasks[:12],
            "recent_activity": recent_activity,
        }

    def connect(
        self,
        source: str,
        target: str,
        relation_type: str = "related_to",
    ) -> str:
        """Manually create a typed relation between two concepts.

        Creates concepts if they don't exist.
        """
        try:
            rel_type = RelationType(relation_type)
        except ValueError:
            valid = ", ".join(rt.value for rt in RelationType)
            return (
                f"Invalid relation type: '{relation_type}'.\n"
                f"Valid types: {valid}"
            )

        obs = Observation(
            concepts=[source, target],
            relations=[(source, target, relation_type)],
            task="manual connection",
            source="pkm_manual",
        )
        result = self._world.ingest(obs)

        return (
            f"Connected: {source} → {rel_type.value} → {target}\n"
            f"New concepts: {result.new_concepts or 'none'}\n"
            f"New relations: {result.new_relations or 'none'}"
        )

    def reflect(self) -> str:
        """Run cognitive consolidation — decay, promote, prune."""
        result = self._world.reflect()

        lines = ["## Reflection Complete", ""]
        if result.promoted_concepts:
            lines.append(f"Promoted: {', '.join(result.promoted_concepts)}")
        if result.demoted_concepts:
            lines.append(f"Demoted: {', '.join(result.demoted_concepts)}")
        if result.decayed_concepts:
            lines.append(
                f"Decayed concepts: {len(result.decayed_concepts)}"
            )
        if result.decayed_relations:
            lines.append(
                f"Decayed relations: {len(result.decayed_relations)}"
            )
        if result.pruned_concepts:
            lines.append(f"Pruned concepts: {', '.join(result.pruned_concepts)}")
        if result.pruned_relations:
            lines.append(
                f"Pruned relations: {len(result.pruned_relations)}"
            )

        if len(lines) == 2:
            lines.append("No changes — your concept world is stable.")

        return "\n".join(lines)

    def status(self) -> str:
        """Overview of the concept world."""
        st = self._world.status()

        lines = [
            "## Knowledge World Status",
            "",
            f"**Concepts:** {st.total_concepts}",
            f"**Relations:** {st.total_relations}",
            f"**Avg confidence:** {st.avg_confidence:.2f}",
        ]

        if st.by_maturity:
            lines.extend(["", "### Maturity Distribution"])
            for maturity, count in sorted(st.by_maturity.items()):
                lines.append(f"- {maturity}: {count}")

        if st.last_reflect:
            lines.append(
                f"\n**Last reflection:** "
                f"{st.last_reflect.strftime('%Y-%m-%d %H:%M')} UTC"
            )
        else:
            lines.append("\n*No reflection performed yet.*")

        # Top concepts by activation count
        all_concepts = self._world.concepts.all()
        if all_concepts:
            top = sorted(
                all_concepts, key=lambda c: c.activation_count, reverse=True
            )[:10]
            lines.extend(["", "### Top Concepts"])
            for c in top:
                lines.append(
                    f"- **{c.name}** ({c.maturity.value}, "
                    f"confidence: {c.confidence:.2f}, "
                    f"activated: {c.activation_count}x)"
                )

        return "\n".join(lines)

    def search(self, query: str) -> str:
        """Search concepts by name substring."""
        query_lower = query.strip().lower()
        matches = [
            c
            for c in self._world.concepts.all()
            if query_lower in c.name.lower()
            or any(query_lower in a.lower() for a in c.aliases)
            or query_lower in c.description.lower()
        ]

        if not matches:
            return f"No concepts matching '{query}'."

        lines = [f"## Search: '{query}' ({len(matches)} results)", ""]
        for c in sorted(matches, key=lambda x: x.confidence, reverse=True):
            desc = f" — {c.description}" if c.description else ""
            lines.append(
                f"- **{c.name}** ({c.maturity.value}, "
                f"confidence: {c.confidence:.2f}){desc}"
            )

        return "\n".join(lines)

    def visualize(self, output: str | None = None) -> str:
        """Generate interactive visualization of the concept world."""
        path = self._world.visualize(
            output=output, open_browser=True
        )
        return f"Visualization saved to: {path}"

    # ── Interactive chat ──────────────────────────────────────────────

    def chat(self) -> None:
        """Start an interactive chat loop.

        Commands:
            /learn <text>       — Learn from text
            /ask <question>     — Ask a question
            /explore <concept>  — Explore a concept
            /connect <a> <b> [type] — Connect two concepts
            /search <query>     — Search concepts
            /web-search <query> — Search the public web
            /reflect            — Run consolidation
            /status             — Show status
            /viz                — Visualize
            /help               — Show help
            /quit               — Exit
        """
        print("=" * 60)
        print("  World 0 — Personal Knowledge Management Agent")
        print("=" * 60)
        print()
        print("Commands: /learn, /ask, /explore, /connect, /search, /web-search,")
        print("          /reflect, /status, /viz, /help, /quit")
        print()
        print("Or just type naturally — I'll treat it as a question.")
        print()

        while True:
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            response = self.handle_input(user_input)
            if response is None:
                break

            print()
            print(response)
            print()

    def handle_input(self, user_input: str) -> str | None:
        """Process a single user input. Returns None to signal exit."""
        if user_input.startswith("/"):
            return self._handle_command(user_input)

        # Default: treat as a question
        return self.ask(user_input)

    def _handle_command(self, cmd: str) -> str | None:
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command == "/quit" or command == "/exit":
            print("Goodbye!")
            return None

        if command == "/help":
            return self._help_text()

        if command == "/learn":
            if not arg:
                return "Usage: /learn <text to learn from>"
            return self.learn(arg)

        if command == "/ask":
            if not arg:
                return "Usage: /ask <question>"
            return self.ask(arg)

        if command == "/explore":
            if not arg:
                return "Usage: /explore <concept name>"
            return self.explore(arg)

        if command == "/connect":
            return self._parse_connect(arg)

        if command == "/search":
            if not arg:
                return "Usage: /search <query>"
            return self.search(arg)

        if command == "/web-search":
            if not arg:
                return "Usage: /web-search <query>"
            return self.search_web(arg, fetch_pages=True)

        if command == "/reflect":
            return self.reflect()

        if command == "/status":
            return self.status()

        if command == "/viz":
            return self.visualize()

        return f"Unknown command: {command}. Type /help for available commands."

    def _parse_connect(self, arg: str) -> str:
        """Parse /connect arguments: source target [type]"""
        if not arg:
            return (
                "Usage: /connect <source> <target> [relation_type]\n"
                f"Types: {', '.join(rt.value for rt in RelationType)}"
            )

        parts = arg.split()
        if len(parts) < 2:
            return "Need at least two concept names: /connect <source> <target>"

        source = parts[0]
        target = parts[1]
        rel_type = parts[2] if len(parts) > 2 else "related_to"

        return self.connect(source, target, rel_type)

    # ── Internal helpers ──────────────────────────────────────────────

    def _parse_domain_filters(
        self,
        domains: str | list[str] | tuple[str, ...] | None,
    ) -> list[str]:
        if domains is None:
            return []
        if isinstance(domains, str):
            raw_parts = re.split(r"[, \n]+", domains)
        else:
            raw_parts = [str(item) for item in domains]

        normalized: list[str] = []
        seen: set[str] = set()
        for part in raw_parts:
            clean = part.strip().lower()
            if not clean:
                continue
            clean = clean.removeprefix("https://").removeprefix("http://")
            clean = clean.split("/", 1)[0].removeprefix("www.")
            if clean and clean not in seen:
                normalized.append(clean)
                seen.add(clean)
        return normalized

    def _auto_sediment_agent_turn(
        self,
        user_input: str,
        assistant_output: str,
    ) -> dict[str, Any]:
        """Persist successful agent dialogue turns into World 0."""
        latest_turn = self.session.latest_turn_summary()
        if latest_turn and latest_turn.failure_class != "none":
            event = {
                "status": "skipped",
                "mode": "agent_chat",
                "task": self._dialogue_task_label(user_input, mode="agent_chat"),
                "source": "",
                "reason": (
                    "Skipped dialogue sedimentation because the latest turn "
                    f"ended with {latest_turn.failure_class}."
                ),
                "new_concepts": [],
                "reinforced_concepts": [],
                "new_relations": [],
                "reinforced_relations": [],
                "hebbian_relations": [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.session.metadata["last_dialogue_sediment"] = event
            return event

        return self._auto_sediment_dialogue_turn(
            user_input,
            assistant_output,
            mode="agent_chat",
        )

    def _auto_sediment_dialogue_turn(
        self,
        user_input: str,
        assistant_output: str,
        *,
        mode: str,
    ) -> dict[str, Any]:
        """Batch dialogue turns and sediment them on the configured interval."""
        session = self._ensure_session()
        interval = int(self._runtime_settings.get("dialogue_sediment_interval", 1))
        event = {
            "status": "skipped",
            "mode": mode,
            "task": self._dialogue_task_label(user_input, mode=mode),
            "source": "",
            "reason": "",
            "pending_turns": 0,
            "required_turns": interval,
            "new_concepts": [],
            "reinforced_concepts": [],
            "new_relations": [],
            "reinforced_relations": [],
            "hebbian_relations": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if not self._runtime_settings.get("auto_sediment_dialogue", True):
            event["reason"] = "Automatic dialogue sedimentation is disabled."
            session.metadata["last_dialogue_sediment"] = event
            return event

        if not self._llm:
            event["reason"] = "No LLM provider configured for dialogue sedimentation."
            session.metadata["last_dialogue_sediment"] = event
            return event

        queue = list(session.metadata.get("dialogue_sediment_queue") or [])
        queue.append({
            "user_input": user_input,
            "assistant_output": assistant_output,
            "mode": mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        session.metadata["dialogue_sediment_queue"] = queue

        if len(queue) < interval:
            event["status"] = "pending"
            event["pending_turns"] = len(queue)
            event["reason"] = (
                f"Waiting for dialogue sedimentation interval "
                f"({len(queue)}/{interval})."
            )
            session.metadata["last_dialogue_sediment"] = event
            return event

        latest_user_input = queue[-1]["user_input"]
        task_label = self._dialogue_task_label(latest_user_input, mode=mode)
        source_label = self._dialogue_source_label(
            session,
            mode=mode,
            batch_size=len(queue),
        )
        dialogue_text = self._render_dialogue_batch_for_learning(queue)
        result = self._ingest_dialogue_text(
            dialogue_text,
            mode=mode,
            task=task_label,
            source=source_label,
            pending_turns=len(queue),
            required_turns=interval,
        )
        if result["status"] == "ingested":
            session.metadata["dialogue_sediment_queue"] = []
        return result

    def _dialogue_task_label(self, user_input: str, *, mode: str) -> str:
        prompt = self._excerpt_text(user_input, limit=80) or "conversation"
        return f"{mode}: {prompt}"

    def _dialogue_source_label(self, session, *, mode: str, batch_size: int = 1) -> str:
        turn_count = len(session.turn_summaries) or max(1, session.message_count() // 2)
        return f"{mode}:{session.id}:{turn_count}:batch{batch_size}"

    def _render_dialogue_for_learning(
        self,
        user_input: str,
        assistant_output: str,
        *,
        mode: str,
    ) -> str:
        user_excerpt = self._excerpt_text(user_input, limit=2000)
        assistant_excerpt = self._excerpt_text(assistant_output, limit=3500)
        lines = [
            f"Dialogue mode: {mode}",
            "",
            "[User]",
            user_excerpt,
            "",
            "[Assistant]",
            assistant_excerpt,
        ]
        return "\n".join(lines).strip()

    def _render_dialogue_batch_for_learning(
        self,
        queue: list[dict[str, Any]],
    ) -> str:
        """Render multiple dialogue turns into one extraction document."""
        lines = [
            f"Dialogue mode: {queue[-1]['mode']}",
            f"Dialogue turns: {len(queue)}",
            "",
        ]
        for index, item in enumerate(queue, start=1):
            lines.extend([
                f"[Turn {index} User]",
                self._excerpt_text(item.get("user_input", ""), limit=2000),
                "",
                f"[Turn {index} Assistant]",
                self._excerpt_text(item.get("assistant_output", ""), limit=3500),
                "",
            ])
        return "\n".join(lines).strip()

    def _remember_projection_snapshot(
        self,
        query: str,
        seeds: list[str],
        projection: Projection,
    ) -> None:
        """Store a compact snapshot of the latest projection for feedback."""
        concept_items = [
            {
                "id": concept.id,
                "name": concept.name,
                "confidence": concept.confidence,
                "maturity": concept.maturity.value,
            }
            for concept in projection.concepts
        ]
        concept_names = {concept.id: concept.name for concept in projection.concepts}
        relation_items = []
        for relation in projection.relations:
            src = concept_names.get(relation.source_id, relation.source_id)
            tgt = concept_names.get(relation.target_id, relation.target_id)
            relation_items.append({
                "id": relation.id,
                "label": f"{src} -> {relation.relation_type.value} -> {tgt}",
                "source_id": relation.source_id,
                "target_id": relation.target_id,
            })
        self.session.metadata["last_projection"] = {
            "query": query,
            "task": projection.task,
            "seeds": seeds,
            "concepts": concept_items,
            "relations": relation_items,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _resolve_relation_feedback_label(self, label: str) -> str | None:
        """Resolve a human-readable relation label into a relation id."""
        match = re.match(
            r"^\s*(.+?)\s*->\s*([a-z_]+)\s*->\s*(.+?)\s*$",
            label,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        source_name, rel_type_name, target_name = match.groups()
        source = self._world.concepts.resolve(source_name)
        target = self._world.concepts.resolve(target_name)
        if not source or not target:
            return None
        try:
            rel_type = RelationType(rel_type_name.strip().lower())
        except ValueError:
            rel_type = None
        relation = self._world.relations.find_between(
            source.id,
            target.id,
            rel_type,
        )
        if relation is None and rel_type is not None:
            relation = self._world.relations.find_between(source.id, target.id, None)
        return relation.id if relation else None

    def _search_web_with_recovery(
        self,
        *,
        query: str,
        focus: str,
        max_results: int,
        domains: list[str] | tuple[str, ...] | None,
    ) -> tuple[list[Any], list[str], FailureReport | None]:
        """Search the web with basic recovery fallbacks for over-constrained queries."""
        domain_filters = list(domains or [])
        search_query = query if not focus else f"{query} {focus}"
        attempts: list[tuple[str, list[str], RecoveryAction | None]] = [
            (search_query, domain_filters, None),
        ]
        if domain_filters:
            attempts.append((search_query, [], RecoveryAction.RETRY_WITHOUT_DOMAINS))
        if focus:
            attempts.append((query, [], RecoveryAction.RETRY_WITHOUT_FOCUS))

        seen: set[tuple[str, tuple[str, ...]]] = set()
        recovery_notes: list[str] = []
        last_failure: FailureReport | None = None
        action_notes = {
            RecoveryAction.RETRY_WITHOUT_DOMAINS: "Retried search without domain filters.",
            RecoveryAction.RETRY_WITHOUT_FOCUS: "Retried search without the extra focus constraint.",
        }

        for attempt_query, attempt_domains, action in attempts:
            key = (attempt_query, tuple(attempt_domains))
            if key in seen:
                continue
            seen.add(key)
            try:
                results = research_utils.search_web(
                    attempt_query,
                    limit=max_results,
                    domains=attempt_domains or None,
                )
            except Exception as exc:
                last_failure = classify_exception(exc, context="search")
                if action is not None:
                    recovery_notes.append(action_notes[action])
                continue
            if results:
                if action is not None:
                    recovery_notes.append(action_notes[action])
                return results, recovery_notes, None
            last_failure = FailureReport(
                failure_class=FailureClass.SEARCH_FETCH_FAILED,
                message=f"No web results found for '{attempt_query}'.",
                retryable=True,
                context="search",
                recovery_actions=[
                    RecoveryAction.RETRY_WITHOUT_DOMAINS,
                    RecoveryAction.RETRY_WITHOUT_FOCUS,
                ],
            )
            if action is not None:
                recovery_notes.append(action_notes[action])

        return [], recovery_notes, last_failure

    def _fetch_search_result_notes(
        self,
        results: list[research_utils.SearchResult],
        *,
        max_pages: int = 3,
    ) -> list[dict[str, str]]:
        fetched: list[dict[str, str]] = []
        for item in results[:max_pages]:
            try:
                doc = research_utils.fetch_web_document(item.url, max_chars=4000)
            except Exception:
                continue
            excerpt = self._excerpt_text(doc.text, limit=320)
            if not excerpt:
                continue
            fetched.append({
                "title": doc.title or item.title,
                "url": item.url,
                "excerpt": excerpt,
            })
        return fetched

    def _compose_search_brief(
        self,
        *,
        query: str,
        focus: str,
        results: list[research_utils.SearchResult],
        fetched: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        fallback_sources = [item.title for item in results[:2] if item.title]
        fallback_themes = [
            item.snippet for item in results[:2] if item.snippet
        ]
        fallback = {
            "summary": (
                f"Search results for {query} cluster around "
                f"{'; '.join(fallback_themes[:2])}."
                if fallback_themes
                else f"Collected {len(results)} public web results for {query}."
            ),
            "themes": [self._excerpt_text(text, limit=120) for text in fallback_themes[:3]],
            "recommended_sources": fallback_sources,
        }

        if not self._llm:
            return fallback

        try:
            raw = self._llm.complete_json(
                self._prompt_with_language("agent.search_brief.system"),
                (
                    f"Query: {query}\n"
                    f"Focus: {focus or 'none'}\n\n"
                    "Search results:\n"
                    + "\n".join(
                        f"- {item.title} | {item.url} | {item.snippet}"
                        for item in results
                    )
                    + "\n\nFetched excerpts:\n"
                    + "\n".join(
                        f"- {item['title']} | {item['url']} | {item['excerpt']}"
                        for item in fetched
                    )
                ),
            )
            parsed = json.loads(self._extract_json(raw))
            summary = str(parsed.get("summary") or fallback["summary"]).strip()
            themes = [
                str(item).strip() for item in parsed.get("themes", [])
                if str(item).strip()
            ] or fallback["themes"]
            recommended_sources = [
                str(item).strip() for item in parsed.get("recommended_sources", [])
                if str(item).strip()
            ] or fallback["recommended_sources"]
            return {
                "summary": summary,
                "themes": themes[:3],
                "recommended_sources": recommended_sources[:3],
            }
        except (LLMError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return fallback

    def _excerpt_text(self, text: str, *, limit: int = 320) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return ""
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    def _distill_research_source(
        self,
        *,
        topic: str,
        focus: str,
        title: str,
        url: str,
        snippet: str,
        text: str,
    ) -> dict[str, Any]:
        fallback = {
            "summary": snippet or f"Source collected for {topic}.",
            "key_points": [snippet] if snippet else [],
            "concepts": self._extract_seeds(f"{topic} {focus}".strip()),
            "open_questions": [],
        }

        try:
            raw = self._llm.complete_json(
                self._prompt_with_language("agent.research_source.system"),
                (
                    f"Topic: {topic}\n"
                    f"Focus: {focus or 'none'}\n"
                    f"Title: {title}\n"
                    f"URL: {url}\n"
                    f"Search snippet: {snippet}\n\n"
                    f"Source text:\n{text[:7000]}"
                ),
            )
            data = json.loads(self._extract_json(raw))
            return {
                "summary": str(data.get("summary", fallback["summary"])).strip() or fallback["summary"],
                "key_points": self._string_list(data.get("key_points")) or fallback["key_points"],
                "concepts": self._string_list(data.get("concepts")) or fallback["concepts"],
                "open_questions": self._string_list(data.get("open_questions")),
            }
        except Exception:
            return fallback

    def _compose_research_brief(
        self,
        *,
        topic: str,
        focus: str,
        source_notes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        fallback_findings = []
        for note in source_notes[:3]:
            if note.get("summary"):
                fallback_findings.append(note["summary"])
        fallback = {
            "summary": fallback_findings[0] if fallback_findings else f"Research notes collected for {topic}.",
            "findings": fallback_findings,
            "gaps": [],
            "next_steps": [],
        }

        rendered_sources = []
        for note in source_notes:
            rendered_sources.append(
                "\n".join([
                    f"Title: {note.get('title', '')}",
                    f"URL: {note.get('url', '')}",
                    f"Summary: {note.get('summary', '')}",
                    f"Key points: {'; '.join(note.get('key_points', []))}",
                    f"Concepts: {', '.join(note.get('concepts', []))}",
                    f"Open questions: {'; '.join(note.get('open_questions', []))}",
                ])
            )

        try:
            raw = self._llm.complete_json(
                self._prompt_with_language("agent.research_report.system"),
                (
                    f"Topic: {topic}\n"
                    f"Focus: {focus or 'none'}\n\n"
                    f"Source notes:\n\n" + "\n\n---\n\n".join(rendered_sources)
                ),
            )
            data = json.loads(self._extract_json(raw))
            return {
                "summary": str(data.get("summary", fallback["summary"])).strip() or fallback["summary"],
                "findings": self._string_list(data.get("findings")) or fallback["findings"],
                "gaps": self._string_list(data.get("gaps")),
                "next_steps": self._string_list(data.get("next_steps")),
            }
        except Exception:
            return fallback

    def _prepare_session_for_agentic(self) -> None:
        """Compact older session context before agentic runs."""
        session = self._ensure_session()
        if not self._llm:
            return
        if not session.needs_compaction():
            return
        self._compact_session(session)

    def _compact_session(self, session) -> None:
        preserve_recent = 16
        covered_messages = max(0, len(session.messages) - preserve_recent)
        if covered_messages <= 0:
            return

        transcript = self._render_messages_for_compaction(
            session.messages[:covered_messages]
        )
        fallback = self._fallback_session_compaction(session, covered_messages)

        try:
            raw = self._llm.complete_json(
                self._prompt_with_language("agent.session_compaction.system"),
                transcript,
            )
            data = json.loads(self._extract_json(raw))
            summary = str(data.get("summary") or fallback["summary"]).strip()
            open_loops = self._string_list(data.get("open_loops")) or fallback["open_loops"]
            key_concepts = self._string_list(data.get("key_concepts")) or fallback["key_concepts"]
        except Exception:
            summary = fallback["summary"]
            open_loops = fallback["open_loops"]
            key_concepts = fallback["key_concepts"]

        from world0.agents.session import SessionCompaction

        session.set_compaction(SessionCompaction(
            summary=summary,
            open_loops=open_loops[:4],
            key_concepts=key_concepts[:6],
            covered_messages=covered_messages,
        ))

    def _render_messages_for_compaction(self, messages: list[Any]) -> str:
        rendered: list[str] = []
        for msg in messages:
            role = msg.role
            if role == "tool_call":
                rendered.append(f"[tool_call] {msg.content}")
                continue
            if role == "tool_result":
                rendered.append(f"[tool_result] {self._excerpt_text(msg.content, limit=280)}")
                continue
            rendered.append(f"[{role}] {self._excerpt_text(msg.content, limit=320)}")
        return "\n".join(rendered[-80:])

    def _fallback_session_compaction(
        self,
        session,
        covered_messages: int,
    ) -> dict[str, list[str] | str]:
        user_messages = [
            self._excerpt_text(msg.content, limit=90)
            for msg in session.messages[:covered_messages]
            if msg.role == "user"
        ]
        assistant_messages = [
            self._excerpt_text(msg.content, limit=120)
            for msg in session.messages[:covered_messages]
            if msg.role == "assistant"
        ]
        key_concepts = self._extract_seeds(" ".join(user_messages[-4:]))[:6]
        open_loops: list[str] = []
        latest_turn = session.latest_turn_summary()
        if latest_turn and latest_turn.failure_class != "none":
            open_loops.append(
                f"Latest earlier turn ended with {latest_turn.failure_class}."
            )
        summary_parts = []
        if user_messages:
            summary_parts.append(f"Earlier requests focused on: {'; '.join(user_messages[-3:])}.")
        if assistant_messages:
            summary_parts.append(f"Recent agent outcomes included: {'; '.join(assistant_messages[-2:])}.")
        summary = " ".join(summary_parts) or "Earlier agent context was compacted."
        return {
            "summary": summary,
            "open_loops": open_loops,
            "key_concepts": key_concepts,
        }

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _extract_seeds(self, query: str) -> list[str]:
        """Extract seed concept names from a query.

        Uses LLM if available, falls back to keyword extraction.
        """
        if self._llm:
            try:
                raw = self._llm.complete_json(
                    self._prompts.render("agent.query_extract.system"),
                    query,
                )
                cleaned = self._extract_json(raw)
                data = json.loads(cleaned)
                seeds = data.get("seeds", [])
                if isinstance(seeds, list) and seeds:
                    return [s.strip().lower() for s in seeds if isinstance(s, str)]
            except (LLMError, json.JSONDecodeError, KeyError):
                pass

        # Fallback: extract significant words
        words = re.findall(r'\b[a-zA-Z\u4e00-\u9fff]{2,}\b', query.lower())
        stopwords = {
            "the", "and", "for", "are", "but", "not", "you", "all",
            "can", "her", "was", "one", "our", "out", "how", "what",
            "why", "when", "where", "which", "who", "this", "that",
            "with", "from", "have", "has", "had", "will", "would",
            "could", "should", "about", "into", "does", "between",
        }
        return [w for w in words if w not in stopwords][:5]

    @staticmethod
    def _extract_json(text: str) -> str:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return text

    @staticmethod
    def _render_projection_basis(
        projection: Projection, seeds: list[str]
    ) -> str:
        """Render a compact explanation of how an answer was grounded."""
        lines = ["### Projection Basis", ""]
        seed_str = ", ".join(seeds) if seeds else "none"
        lines.append(f"- Seed concepts: {seed_str}")
        lines.append(
            f"- Activated concepts: {len(projection.concepts)}"
        )
        lines.append(
            f"- Included relations: {len(projection.relations)}"
        )

        top = projection.top_concepts(5)
        if top:
            ranked = ", ".join(
                f"{c.name} ({projection.activation_scores.get(c.id, 0.0):.2f})"
                for c in top
            )
            lines.append(f"- Top activated concepts: {ranked}")

        if projection.relations:
            concept_names = {c.id: c.name for c in projection.concepts}
            key_relations = []
            for rel in sorted(
                projection.relations, key=lambda r: r.weight, reverse=True
            )[:3]:
                src = concept_names.get(rel.source_id, rel.source_id)
                tgt = concept_names.get(rel.target_id, rel.target_id)
                key_relations.append(
                    f"{src} → {rel.relation_type.value} → {tgt}"
                )
            lines.append(f"- Key relation paths: {'; '.join(key_relations)}")

        return "\n".join(lines)

    def _language_instruction(self) -> str:
        if self._language == "zh":
            return "Respond in Simplified Chinese."
        return "Respond in English."

    def _prompt_with_language(self, prompt_id: str, **values: Any) -> str:
        prompt = self._prompts.render(prompt_id, **values)
        return f"{prompt}\n\n{self._language_instruction()}"

    @staticmethod
    def _help_text() -> str:
        return """\
## World 0 PKM Agent — Commands

| Command | Description |
|---------|-------------|
| `/learn <text>` | Ingest knowledge from text |
| `/ask <question>` | Query your concept world |
| `/explore <concept>` | Deep dive into a concept |
| `/connect <a> <b> [type]` | Create a relation |
| `/search <query>` | Search concepts by name |
| `/reflect` | Run consolidation (decay/promote/prune) |
| `/status` | Show world overview |
| `/viz` | Generate interactive visualization |
| `/help` | Show this help |
| `/quit` | Exit |

**Relation types:** contains, part_of, depends_on, supports, \
contrasts, similar_to, activates, precedes, derived_from, related_to

**Tips:**
- Just type naturally to ask a question (no /ask needed)
- Learn often — concepts strengthen with repeated exposure
- Use /reflect periodically to consolidate your knowledge
- Use /explore to see how a concept connects to others\
"""
