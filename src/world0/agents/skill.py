"""Skill system — reusable multi-step knowledge processing workflows.

A Skill is a named, self-describing workflow that combines multiple
tool calls with a specialized prompt to accomplish a complex task.

Skills execute through the AgentLoop with a purpose-built system prompt
that guides the LLM to use the right tools in the right order.

Built-in skills:
- digest_article: Extract and ingest knowledge from article text
- research_topic: Research a topic from outside sources and synthesize it
- analyze_topic: Deep analysis of a topic across the concept world
- build_knowledge_map: Map connections between a set of concepts
- review_and_connect: Review recent knowledge and find new connections
- summarize_world: Generate a comprehensive summary of the knowledge state
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from world0.agents.pkm import PKMAgent


@dataclass
class SkillParam:
    """A parameter required by a skill."""
    name: str
    description: str
    required: bool = True
    default: str = ""


@dataclass
class Skill:
    """A reusable multi-step knowledge processing workflow.

    A skill is essentially a prompt template + parameter schema.
    When invoked, it generates a user prompt from the template and
    parameters, then runs it through the AgentLoop which decides
    which tools to call.
    """
    name: str
    description: str
    prompt_template: str
    parameters: list[SkillParam] = field(default_factory=list)
    system_override: str = ""  # Optional override for the system prompt
    tags: list[str] = field(default_factory=list)

    def render_prompt(self, **kwargs) -> str:
        """Render the prompt template with given parameters."""
        prompt = self.prompt_template
        for param in self.parameters:
            key = param.name
            value = kwargs.get(key, param.default)
            prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
        return prompt

    def validate_params(self, **kwargs) -> list[str]:
        """Validate that required parameters are provided. Returns error messages."""
        errors = []
        for param in self.parameters:
            if param.required and not kwargs.get(param.name):
                errors.append(f"Missing required parameter: {param.name}")
        return errors

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {"name": p.name, "description": p.description,
                 "required": p.required, "default": p.default}
                for p in self.parameters
            ],
            "tags": self.tags,
        }


class SkillRegistry:
    """Registry for available skills.

    Skills can be registered programmatically or loaded from JSON config.
    The registry provides discovery, validation, and invocation.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def names(self) -> list[str]:
        return list(self._skills.keys())

    def search(self, query: str) -> list[Skill]:
        """Search skills by name, description, or tags."""
        q = query.lower()
        return [
            s for s in self._skills.values()
            if q in s.name.lower()
            or q in s.description.lower()
            or any(q in t.lower() for t in s.tags)
        ]

    def load_from_file(self, path: str) -> int:
        """Load skills from a JSON file. Returns count loaded."""
        from pathlib import Path
        p = Path(path).expanduser()
        if not p.exists():
            return 0
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            skills = data.get("skills", [])
            count = 0
            for s in skills:
                skill = Skill(
                    name=s["name"],
                    description=s.get("description", ""),
                    prompt_template=s.get("prompt_template", ""),
                    parameters=[
                        SkillParam(
                            name=p["name"],
                            description=p.get("description", ""),
                            required=p.get("required", True),
                            default=p.get("default", ""),
                        )
                        for p in s.get("parameters", [])
                    ],
                    system_override=s.get("system_override", ""),
                    tags=s.get("tags", []),
                )
                self.register(skill)
                count += 1
            return count
        except (json.JSONDecodeError, KeyError):
            return 0

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills


class SkillExecutor:
    """Executes skills through the PKM Agent's agentic loop.

    Takes a skill + parameters, renders the prompt, and runs it
    through agent_chat() which autonomously decides which tools to call.
    """

    def __init__(self, agent: PKMAgent) -> None:
        self._agent = agent

    def execute(
        self,
        skill: Skill,
        on_tool_call: Callable | None = None,
        on_tool_result: Callable | None = None,
        **kwargs,
    ) -> str:
        """Execute a skill with the given parameters.

        Returns the agent's final response.
        """
        # Validate parameters
        errors = skill.validate_params(**kwargs)
        if errors:
            return f"Skill validation failed:\n" + "\n".join(f"- {e}" for e in errors)

        # Render the prompt
        prompt = skill.render_prompt(**kwargs)

        # Execute through agentic loop
        if not self._agent._chat_provider:
            raise RuntimeError(
                "Skill execution requires agentic mode. Call init_agentic() first."
            )

        from world0.agents.loop import AgentLoop
        self._agent._prepare_session_for_agentic()

        # Use skill's system override if provided
        loop = AgentLoop(
            self._agent._chat_provider,
            self._agent._ensure_tools(),
            self._agent._ensure_session(),
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            language=self._agent.language,
        )

        # If skill has a system override, we inject it as context
        if skill.system_override:
            full_prompt = f"[Skill: {skill.name}]\n{skill.system_override}\n\n{prompt}"
        else:
            full_prompt = f"[Skill: {skill.name}]\n{prompt}"

        return loop.run(full_prompt)

    def execute_by_name(
        self,
        skill_name: str,
        registry: SkillRegistry,
        on_tool_call: Callable | None = None,
        on_tool_result: Callable | None = None,
        **kwargs,
    ) -> str:
        """Execute a skill by looking it up in the registry."""
        skill = registry.get(skill_name)
        if not skill:
            available = ", ".join(registry.names())
            return f"Unknown skill: '{skill_name}'. Available: {available}"
        return self.execute(skill, on_tool_call, on_tool_result, **kwargs)


# ── Built-in Skills ──────────────────────────────────────────────────

def builtin_skills() -> list[Skill]:
    """Return all built-in PKM skills."""
    return [
        Skill(
            name="digest_article",
            description=(
                "Extract and ingest knowledge from an article or long text. "
                "Identifies key concepts, establishes relations, and provides "
                "a summary of what was learned."
            ),
            prompt_template=(
                "Please digest the following article/text and add it to my knowledge world.\n\n"
                "Steps:\n"
                "1. Use the `learn` tool to ingest the text\n"
                "2. Use `list_concepts` to see what was extracted\n"
                "3. Use `explore` on the most important new concepts\n"
                "4. If you notice concepts that should be connected but aren't, use `connect`\n"
                "5. Give me a brief summary of what was learned and what connections were made\n\n"
                "Article text:\n{{text}}"
            ),
            parameters=[
                SkillParam("text", "The article or text to digest"),
            ],
            tags=["learn", "article", "ingest"],
        ),
        Skill(
            name="research_topic",
            description=(
                "Research a topic from outside sources, distill the strongest findings, "
                "bring them into World 0, and highlight open questions."
            ),
            prompt_template=(
                "Please research the topic '{{topic}}' for me.\n\n"
                "Focus: {{focus}}\n"
                "Sources limit: {{sources_limit}}\n"
                "Learn findings into World 0: {{save_findings}}\n\n"
                "Steps:\n"
                "1. Use the `research_topic` tool with the topic, focus, and source limit\n"
                "2. If the brief surfaces important concepts, use `explore` on the strongest ones\n"
                "3. If needed, use `ask` to project what World 0 now knows about the topic\n"
                "4. Return a concise research brief with findings, gaps, next steps, and source links"
            ),
            parameters=[
                SkillParam("topic", "The topic or question to research"),
                SkillParam("focus", "Optional angle or lens for the research", required=False, default=""),
                SkillParam("sources_limit", "How many sources to review", required=False, default="4"),
                SkillParam("save_findings", "Whether to learn the findings into World 0", required=False, default="true"),
            ],
            tags=["research", "web", "sources"],
        ),
        Skill(
            name="analyze_topic",
            description=(
                "Deep analysis of a topic: what you know, what's connected, "
                "what gaps exist, and suggestions for what to learn next."
            ),
            prompt_template=(
                "Please perform a deep analysis of the topic '{{topic}}' in my knowledge world.\n\n"
                "Steps:\n"
                "1. Use `search` to find related concepts\n"
                "2. Use `explore` on each relevant concept found\n"
                "3. Use `ask` to query what I know about this topic\n"
                "4. Identify gaps — what concepts are missing or weak (embryonic)?\n"
                "5. Suggest what I should learn next to strengthen this area\n\n"
                "Be thorough but concise in your analysis."
            ),
            parameters=[
                SkillParam("topic", "The topic to analyze"),
            ],
            tags=["analyze", "research", "gaps"],
        ),
        Skill(
            name="build_knowledge_map",
            description=(
                "Map connections between a set of concepts. Explores each one, "
                "identifies missing links, and creates new relations."
            ),
            prompt_template=(
                "Please build a knowledge map around these concepts: {{concepts}}\n\n"
                "Steps:\n"
                "1. Use `explore` on each concept to understand current connections\n"
                "2. Identify concepts that should be related but aren't connected\n"
                "3. Use `connect` to create meaningful typed relations\n"
                "4. Use `search` to find other relevant concepts in the world\n"
                "5. Give me a summary of the map: what's well-connected and what's isolated\n\n"
                "Use specific relation types (depends_on, supports, contrasts, etc.) rather than generic related_to."
            ),
            parameters=[
                SkillParam("concepts", "Comma-separated list of concept names"),
            ],
            tags=["map", "connections", "relations"],
        ),
        Skill(
            name="review_and_connect",
            description=(
                "Review recent knowledge and find new connections. "
                "Looks at recently added concepts, identifies patterns, "
                "and creates cross-domain links."
            ),
            prompt_template=(
                "Please review my knowledge world and find new connections.\n\n"
                "Steps:\n"
                "1. Use `status` to see the current state\n"
                "2. Use `list_concepts` to see all concepts, especially embryonic ones\n"
                "3. Look for concepts from different domains that could be connected\n"
                "4. Use `connect` to create meaningful cross-domain relations\n"
                "5. Use `reflect` to consolidate the knowledge\n"
                "6. Give me a summary of what connections you found and why they matter\n\n"
                "Focus on surprising or non-obvious connections across different domains."
            ),
            parameters=[],
            tags=["review", "cross-domain", "synthesis"],
        ),
        Skill(
            name="summarize_world",
            description=(
                "Generate a comprehensive summary of the entire knowledge world: "
                "key themes, strongest concepts, most important relations, and overall health."
            ),
            prompt_template=(
                "Please give me a comprehensive summary of my knowledge world.\n\n"
                "Steps:\n"
                "1. Use `status` for the overview\n"
                "2. Use `list_concepts` to see all concepts\n"
                "3. Identify the top 3-5 knowledge themes/domains\n"
                "4. For each theme, use `explore` on the core concepts\n"
                "5. Summarize:\n"
                "   - What are my strongest knowledge areas?\n"
                "   - What are the main themes and how do they connect?\n"
                "   - What concepts are fading and might need reinforcement?\n"
                "   - What's the overall health of my knowledge world?"
            ),
            parameters=[],
            tags=["summary", "overview", "health"],
        ),
        Skill(
            name="learn_and_quiz",
            description=(
                "Learn from text and then generate quiz questions to test understanding. "
                "Helps reinforce learning through active recall."
            ),
            prompt_template=(
                "Please help me learn and test my understanding.\n\n"
                "Steps:\n"
                "1. Use `learn` to ingest the following text\n"
                "2. Use `explore` on the most important concepts extracted\n"
                "3. Generate 3-5 quiz questions that test understanding of the key concepts and relations\n"
                "4. Include questions about how concepts connect to each other\n\n"
                "Text to learn:\n{{text}}"
            ),
            parameters=[
                SkillParam("text", "The text to learn and be quizzed on"),
            ],
            tags=["learn", "quiz", "recall"],
        ),
    ]


def register_builtin_skills(registry: SkillRegistry) -> None:
    """Register all built-in skills into a registry."""
    for skill in builtin_skills():
        registry.register(skill)
