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
from world0.agents.provider import create_provider, default_model_for_provider
from world0.llm.base import LLMError, LLMProvider
from world0.schemas.relation import RelationType
from world0.schemas.types import Observation, Projection
from world0.world import World


# ── LLM prompts ──────────────────────────────────────────────────────

_ANSWER_SYSTEM_PROMPT = """\
You are a cognitive concept-world assistant powered by World 0. \
You help the user understand a task domain through concepts, relations, \
activation, and local projection.

You will receive:
1. A cognitive projection — a local view of the user's concept world \
relevant to their query. This includes concepts (with maturity and confidence), \
relations between them, and activation scores.
2. The user's question or request.

Your job:
- Answer based on the cognitive projection provided.
- Highlight connections between concepts the user might not have noticed.
- If the projection is sparse, say so honestly — suggest what observations \
or concept links would make the world clearer.
- Be concise but insightful. Focus on conceptual understanding, not trivia.
- When referencing concepts, mention their maturity level if it adds context \
(e.g., an "embryonic" concept is new and may need more reinforcement).

Do NOT fabricate knowledge that isn't in the projection or general knowledge. \
If the projection doesn't cover the query well, say so.\
"""

_QUERY_EXTRACT_PROMPT = """\
Extract the key concept names from this user query. These will be used as \
seed concepts to activate a cognitive projection.

Return ONLY a JSON object:
{"seeds": ["concept1", "concept2", ...]}

Extract 1-5 concept names that best capture what the user is asking about. \
Use lowercase, canonical forms. Respond ONLY with JSON, no explanation.\
"""

_LEARN_SUMMARY_PROMPT = """\
Summarize what was just learned in 1-2 sentences. The user submitted text \
and the system extracted concepts and relations. Here is the ingest result:

{ingest_result}

Be brief and informative. Mention the most interesting new concepts or \
relations discovered.\
"""

_RESEARCH_SOURCE_PROMPT = """\
You are distilling a web source into a compact research note for World 0.

Return ONLY a JSON object:
{
  "summary": "1-2 sentence summary",
  "key_points": ["point 1", "point 2"],
  "concepts": ["concept a", "concept b"],
  "open_questions": ["question 1"]
}

Rules:
- Focus on claims, mechanisms, boundaries, tradeoffs, and notable terms.
- Keep key_points to 2-4 items.
- Keep concepts to 2-6 concise concept names.
- Keep open_questions to 0-3 items.
- Respond with JSON only.\
"""

_RESEARCH_REPORT_PROMPT = """\
You are composing a concise research brief for World 0 from source notes.

Return ONLY a JSON object:
{
  "summary": "short overall summary",
  "findings": ["finding 1", "finding 2"],
  "gaps": ["gap 1"],
  "next_steps": ["step 1", "step 2"]
}

Rules:
- Findings should synthesize across sources, not repeat them verbatim.
- Gaps should identify uncertainty, missing evidence, or weakly covered areas.
- Next steps should be concrete research or learning actions.
- Keep each list to 2-5 items.
- Respond with JSON only.\
"""


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
    ) -> None:
        store_path = Path(store_path).expanduser()
        self._store_path = store_path
        self._world = World(store_path=store_path, llm=llm)
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
        }

        # Agentic components (lazy-initialized)
        self._tool_registry = None
        self._session_store = None
        self._current_session = None
        self._agent_loop = None
        self._chat_provider = None

        # MCP & Skill (lazy-initialized)
        self._mcp_manager = None
        self._skill_registry = None
        self._skill_executor = None

    @property
    def world(self) -> World:
        """Access the underlying World 0 instance."""
        return self._world

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
    ) -> None:
        """Update language and LLM runtime settings."""
        if language:
            self._language = language
            self._runtime_settings["language"] = language

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

        loop = AgentLoop(
            self._chat_provider,
            self._ensure_tools(),
            self._ensure_session(),
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            language=self._language,
        )
        return loop.run(user_input)

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
            }
            for s in sessions
        ]

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
    ) -> str:
        """Record a non-agentic UI interaction into the current session."""
        session = self._ensure_session()
        session.add_message("user", user_input, mode=mode)
        session.add_message("assistant", assistant_output, mode=mode)
        if save:
            return self.save_session()
        return session.id

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
            register_builtin_skills(self._skill_registry)
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
                    "You are a concise knowledge assistant. "
                    f"{self._language_instruction()} "
                    "Respond with a JSON object: {\"summary\": \"...\"}",
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

        try:
            results = research_utils.search_web(search_query, limit=source_limit)
        except Exception as e:
            return f"Web search failed: {e}"

        if not results:
            return f"No web results found for '{search_query}'."

        source_notes: list[dict[str, Any]] = []
        learning_log: list[str] = []

        for result in results[:source_limit]:
            try:
                doc = research_utils.fetch_web_document(result.url)
            except Exception:
                continue
            if not doc.text.strip():
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
            return (
                f"I found search results for '{search_query}', but couldn't extract readable source text."
            )

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

        if learning_log:
            lines.extend(["", "### World 0 Update"])
            lines.extend(learning_log)

        lines.extend(["", "### Projection Into World 0", "", projection])
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

        if not self._llm:
            return f"{rendered}\n\n{basis}"

        # Generate answer using LLM + projection context
        try:
            user_prompt = (
                f"## Cognitive Projection\n{rendered}\n\n"
                f"## User Question\n{query}"
            )
            response = self._llm.complete_json(
                f"{_ANSWER_SYSTEM_PROMPT}\n\n{self._language_instruction()}",
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
        print("Commands: /learn, /ask, /explore, /connect, /search,")
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
                f"{_RESEARCH_SOURCE_PROMPT}\n\n{self._language_instruction()}",
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
                f"{_RESEARCH_REPORT_PROMPT}\n\n{self._language_instruction()}",
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
                raw = self._llm.complete_json(_QUERY_EXTRACT_PROMPT, query)
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
