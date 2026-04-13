"""PKM-specific tools — registered into the ToolRegistry for agentic use.

Each function here is a tool handler that operates on a PKMAgent instance.
The build_pkm_tools() factory wires them into a ToolRegistry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from world0.agents import research as research_utils
from world0.agents.tools.registry import (
    Permission,
    Tool,
    ToolParam,
    ToolRegistry,
    ToolResult,
)
from world0.schemas.relation import RelationType
from world0.schemas.types import Observation

if TYPE_CHECKING:
    from world0.agents.pkm import PKMAgent


def build_pkm_tools(agent: PKMAgent) -> ToolRegistry:
    """Build the full tool registry for a PKM Agent instance."""
    registry = ToolRegistry()

    # ── Learn ─────────────────────────────────────────────────────────
    def learn(text: str, task: str = "knowledge intake", source: str = "") -> ToolResult:
        try:
            result = agent.learn(text, task=task, source=source)
            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(success=False, output=str(e))

    registry.register(Tool(
        name="learn",
        description=(
            "Learn knowledge from text. Extracts concepts and relations "
            "from the given text and ingests them into the concept world. "
            "Use this when the user provides new information, articles, "
            "notes, or any text to be understood and remembered."
        ),
        parameters=[
            ToolParam("text", "The text content to learn from", required=True),
            ToolParam("task", "Task context label (e.g., 'reading notes', 'meeting recap')", required=False),
            ToolParam("source", "Source label for provenance tracking", required=False),
        ],
        handler=learn,
        permission=Permission.WRITE,
    ))

    # ── Ask / Query ───────────────────────────────────────────────────
    def ask(query: str, max_concepts: int = 15) -> ToolResult:
        try:
            result = agent.ask(query, max_concepts=int(max_concepts))
            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(success=False, output=str(e))

    registry.register(Tool(
        name="ask",
        description=(
            "Query the concept world. Activates relevant concepts based "
            "on the query, generates a cognitive projection, and synthesizes "
            "an answer. Use this to answer questions about previously learned knowledge."
        ),
        parameters=[
            ToolParam("query", "The question to ask about the knowledge world", required=True),
            ToolParam("max_concepts", "Maximum concepts in projection (default: 15)", type="integer", required=False),
        ],
        handler=ask,
        permission=Permission.READ,
    ))

    # ── Explore ───────────────────────────────────────────────────────
    def explore(concept_name: str) -> ToolResult:
        result = agent.explore(concept_name)
        found = "not found" not in result.lower()
        return ToolResult(success=found, output=result)

    registry.register(Tool(
        name="explore",
        description=(
            "Deep-dive into a single concept. Shows its maturity, confidence, "
            "relations, activity history, and connected concepts. "
            "Use this when the user wants to understand a specific concept in detail."
        ),
        parameters=[
            ToolParam("concept_name", "Name of the concept to explore", required=True),
        ],
        handler=explore,
        permission=Permission.READ,
    ))

    # ── Search ────────────────────────────────────────────────────────
    def search(query: str) -> ToolResult:
        result = agent.search(query)
        return ToolResult(success=True, output=result)

    registry.register(Tool(
        name="search",
        description=(
            "Search concepts by name, alias, or description substring. "
            "Returns matching concepts with their maturity and confidence. "
            "Use this to find concepts when you're not sure of the exact name."
        ),
        parameters=[
            ToolParam("query", "Search query string", required=True),
        ],
        handler=search,
        permission=Permission.READ,
    ))

    # ── Connect ───────────────────────────────────────────────────────
    def connect(
        source: str, target: str, relation_type: str = "related_to"
    ) -> ToolResult:
        result = agent.connect(source, target, relation_type)
        ok = "invalid" not in result.lower()
        return ToolResult(success=ok, output=result)

    registry.register(Tool(
        name="connect",
        description=(
            "Create a typed relation between two concepts. "
            "Creates the concepts if they don't exist. "
            "Use this to manually establish meaningful connections."
        ),
        parameters=[
            ToolParam("source", "Source concept name", required=True),
            ToolParam("target", "Target concept name", required=True),
            ToolParam(
                "relation_type",
                "Type of relation",
                required=False,
                enum=[rt.value for rt in RelationType],
            ),
        ],
        handler=connect,
        permission=Permission.WRITE,
    ))

    # ── Status ────────────────────────────────────────────────────────
    def status() -> ToolResult:
        result = agent.status()
        return ToolResult(success=True, output=result)

    registry.register(Tool(
        name="status",
        description=(
            "Show overview of the concept world: total concepts/relations, "
            "maturity distribution, average confidence, and top concepts. "
            "Use this to give the user a summary of their knowledge state."
        ),
        parameters=[],
        handler=status,
        permission=Permission.READ,
    ))

    # ── Reflect ───────────────────────────────────────────────────────
    def reflect() -> ToolResult:
        result = agent.reflect()
        return ToolResult(success=True, output=result)

    registry.register(Tool(
        name="reflect",
        description=(
            "Run cognitive consolidation: decay unused concepts, "
            "promote/demote based on activity, prune deeply decayed items. "
            "Use this when the user asks to clean up or consolidate their knowledge."
        ),
        parameters=[],
        handler=reflect,
        permission=Permission.ADMIN,
    ))

    # ── List Concepts ─────────────────────────────────────────────────
    def list_concepts(maturity: str = "", limit: int = 20) -> ToolResult:
        all_c = agent.world.concepts.all()
        if maturity:
            all_c = [c for c in all_c if c.maturity.value == maturity]
        top = sorted(all_c, key=lambda c: c.activation_count, reverse=True)[:int(limit)]
        if not top:
            return ToolResult(success=True, output="No concepts found.")
        lines = [f"**{c.name}** ({c.maturity.value}, conf: {c.confidence:.2f}, activated: {c.activation_count}x)"
                 for c in top]
        return ToolResult(success=True, output="\n".join(lines))

    registry.register(Tool(
        name="list_concepts",
        description=(
            "List concepts in the knowledge world, optionally filtered by maturity. "
            "Returns concepts sorted by activation count."
        ),
        parameters=[
            ToolParam("maturity", "Filter by maturity stage: embryonic, developing, established, core, fading", required=False,
                      enum=["", "embryonic", "developing", "established", "core", "fading"]),
            ToolParam("limit", "Maximum number to return (default: 20)", type="integer", required=False),
        ],
        handler=list_concepts,
        permission=Permission.READ,
    ))

    # ── Batch Learn ───────────────────────────────────────────────────
    def batch_learn(texts: str, task: str = "batch intake") -> ToolResult:
        """Learn from multiple texts separated by ===."""
        chunks = [t.strip() for t in texts.split("===") if t.strip()]
        results = []
        for i, chunk in enumerate(chunks, 1):
            try:
                r = agent.learn(chunk, task=task, source=f"batch_{i}")
                results.append(f"Chunk {i}: {r}")
            except Exception as e:
                results.append(f"Chunk {i}: Error — {e}")
        return ToolResult(success=True, output="\n\n".join(results))

    registry.register(Tool(
        name="batch_learn",
        description=(
            "Learn from multiple text chunks at once. "
            "Chunks are separated by '==='. Useful for ingesting "
            "multiple notes or paragraphs in one operation."
        ),
        parameters=[
            ToolParam("texts", "Multiple text chunks separated by '==='", required=True),
            ToolParam("task", "Task context label for all chunks", required=False),
        ],
        handler=batch_learn,
        permission=Permission.WRITE,
    ))

    # ── Run Skill ─────────────────────────────────────────────────────
    def run_skill(skill_name: str, text: str = "", topic: str = "", concepts: str = "") -> ToolResult:
        """Execute a skill by name with parameters."""
        try:
            # Build params dict from non-empty values
            params = {}
            if text:
                params["text"] = text
            if topic:
                params["topic"] = topic
            if concepts:
                params["concepts"] = concepts

            result = agent.run_skill(skill_name, **params)
            return ToolResult(success=True, output=result)
        except RuntimeError as e:
            return ToolResult(success=False, output=str(e))
        except Exception as e:
            return ToolResult(success=False, output=f"Skill error: {e}")

    registry.register(Tool(
        name="run_skill",
        description=(
            "Execute a multi-step skill workflow. Skills are complex operations "
            "that combine multiple tools in a guided sequence. "
            "Available skills: digest_article (params: text), "
            "research_topic (params: topic, focus, sources_limit, save_findings), "
            "analyze_topic (params: topic), "
            "build_knowledge_map (params: concepts — comma-separated), "
            "review_and_connect (no params), "
            "summarize_world (no params), "
            "learn_and_quiz (params: text). "
            "Use this when a task requires a coordinated multi-step approach."
        ),
        parameters=[
            ToolParam("skill_name", "Name of the skill to run (use list_skills to see available)", required=True),
            ToolParam("text", "Text content (for digest_article, learn_and_quiz)", required=False),
            ToolParam("topic", "Topic name (for analyze_topic)", required=False),
            ToolParam("concepts", "Comma-separated concept names (for build_knowledge_map)", required=False),
        ],
        handler=run_skill,
        permission=Permission.WRITE,
    ))

    # ── List Skills ──────────────────────────────────────────────────
    def list_available_skills() -> ToolResult:
        """List all registered skills."""
        return ToolResult(success=True, output=agent.list_skills())

    registry.register(Tool(
        name="list_skills",
        description=(
            "List all available multi-step skills with their descriptions and parameters. "
            "Use this to discover what complex workflows are available."
        ),
        parameters=[],
        handler=list_available_skills,
        permission=Permission.READ,
    ))

    # ── Web Search ───────────────────────────────────────────────────
    def web_search(
        query: str,
        limit: int = 5,
        focus: str = "",
        domains: str = "",
        fetch_pages: bool = False,
    ) -> ToolResult:
        try:
            fetch_flag = fetch_pages
            if not isinstance(fetch_pages, bool):
                fetch_flag = str(fetch_pages).strip().lower() not in {
                    "false", "0", "no", "off", ""
                }
            rendered = agent.search_web(
                query,
                focus=focus,
                max_results=int(limit),
                domains=domains,
                fetch_pages=fetch_flag,
            )
            ok = "failed" not in rendered.lower()
            return ToolResult(success=ok, output=rendered)
        except Exception as e:
            return ToolResult(success=False, output=f"Search error: {e}")

    registry.register(Tool(
        name="web_search",
        description=(
            "Search the public web for a topic and return candidate sources with titles, "
            "URLs, snippets, and optional fetched page glimpses. Use this when the user asks "
            "for outside research or fresh sources."
        ),
        parameters=[
            ToolParam("query", "Search query string", required=True),
            ToolParam("limit", "Maximum number of results to return (default: 5)", type="integer", required=False),
            ToolParam("focus", "Optional search angle or constraint", required=False),
            ToolParam("domains", "Optional comma-separated domain filter list", required=False),
            ToolParam("fetch_pages", "Whether to fetch top results and include short excerpts", type="boolean", required=False),
        ],
        handler=web_search,
        permission=Permission.READ,
    ))

    # ── Web Fetch ────────────────────────────────────────────────────
    def web_fetch(url: str) -> ToolResult:
        """Fetch content from a URL and return text."""
        try:
            doc = research_utils.fetch_web_document(url, max_chars=8000)
            output = "\n".join([
                f"Title: {doc.title}",
                f"URL: {doc.url}",
                "",
                doc.text,
            ])
            return ToolResult(
                success=True,
                output=output,
                data={"title": doc.title, "url": doc.url},
            )
        except Exception as e:
            return ToolResult(success=False, output=f"Fetch error: {e}")

    registry.register(Tool(
        name="web_fetch",
        description=(
            "Fetch and extract text content from a URL. "
            "Use this to retrieve articles, documentation, or web pages "
            "that the user wants to learn from. Returns plain text."
        ),
        parameters=[
            ToolParam("url", "The URL to fetch", required=True),
        ],
        handler=web_fetch,
        permission=Permission.READ,
    ))

    # ── Research ─────────────────────────────────────────────────────
    def research_topic(
        topic: str,
        focus: str = "",
        max_sources: int = 4,
        save_findings: bool = True,
    ) -> ToolResult:
        try:
            save_flag = save_findings
            if not isinstance(save_findings, bool):
                save_flag = str(save_findings).strip().lower() not in {
                    "false", "0", "no", "off", ""
                }
            result = agent.research_topic(
                topic,
                focus=focus,
                max_sources=int(max_sources),
                save_findings=save_flag,
            )
            lowered = result.lower()
            ok = not any(marker in lowered for marker in (
                "please provide",
                "requires an llm provider",
                "web search failed",
                "no web results found",
                "couldn't extract readable source text",
            ))
            return ToolResult(success=ok, output=result)
        except Exception as e:
            return ToolResult(success=False, output=f"Research error: {e}")

    registry.register(Tool(
        name="research_topic",
        description=(
            "Run an end-to-end research workflow: search the web, fetch sources, distill key ideas, "
            "optionally learn them into World 0, and return a structured research brief."
        ),
        parameters=[
            ToolParam("topic", "The research topic or question", required=True),
            ToolParam("focus", "Optional focus or angle to investigate", required=False),
            ToolParam("max_sources", "Maximum sources to review (default: 4)", type="integer", required=False),
            ToolParam("save_findings", "Whether to learn the source findings into World 0", type="boolean", required=False),
        ],
        handler=research_topic,
        permission=Permission.WRITE,
    ))

    return registry
