"""Agentic conversation loop — inspired by claw-code's ConversationRuntime.

The agent loop lets the LLM autonomously decide which tools to call,
executing a think→act→observe cycle until the task is complete.

This transforms the PKM Agent from command→response into a true agent
that can plan multi-step knowledge operations.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable

from world0.agents.provider import ChatProvider, ChatResponse, ToolCall
from world0.agents.session import Session, TurnSummary
from world0.agents.tools.registry import Permission, ToolRegistry, ToolResult
from world0.llm.base import LLMError
from world0.prompts import PromptRegistry

# Maximum tool calls per turn to prevent infinite loops
MAX_TOOL_ROUNDS = 10


@dataclass
class AgentTurnOutcome:
    stop_reason: str
    failure_class: str
    rounds: int
    tool_count: int
    failed_tools: list[str]
    user_input: str
    assistant_output: str


def _build_dynamic_prompt(
    tools: ToolRegistry,
    language: str = "en",
    prompt_registry: PromptRegistry | None = None,
) -> str:
    """Build system prompt with dynamic context about available tools."""
    registry = prompt_registry or PromptRegistry()
    prompt = registry.render("agent.loop.system")

    # Add MCP tools section if any are registered
    mcp_tools = [t for t in tools.all() if t.name.startswith("mcp__")]
    if mcp_tools:
        lines = ["\n\n## MCP External Tools\n\nThe following external tools are available via MCP servers:"]
        for t in mcp_tools:
            lines.append(f"- `{t.name}`: {t.description}")
        lines.append("\nUse these tools when the user's request matches their capability.")
        prompt += "\n".join(lines)

    if language == "zh":
        prompt += "\n\nAlways respond in Simplified Chinese."
    else:
        prompt += "\n\nAlways respond in English."

    return prompt


class AgentLoop:
    """Agentic conversation loop with tool use.

    Runs a think→act→observe cycle:
    1. Send conversation + tools to LLM
    2. If LLM requests tool calls → execute them, feed results back
    3. Repeat until LLM produces a final text response
    4. Persist everything to the session

    Usage::

        loop = AgentLoop(chat_provider, tool_registry, session)
        response = loop.run("What do I know about machine learning?")
    """

    def __init__(
        self,
        provider: ChatProvider,
        tools: ToolRegistry,
        session: Session,
        *,
        permission: Permission = Permission.ADMIN,
        max_rounds: int = MAX_TOOL_ROUNDS,
        on_tool_call: Callable[[str, dict], None] | None = None,
        on_tool_result: Callable[[str, ToolResult], None] | None = None,
        language: str = "en",
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._session = session
        self._permission = permission
        self._max_rounds = max_rounds
        self._on_tool_call = on_tool_call
        self._on_tool_result = on_tool_result
        self._language = language
        self._prompts = prompt_registry or PromptRegistry()
        self._last_outcome: AgentTurnOutcome | None = None

    @property
    def session(self) -> Session:
        return self._session

    @property
    def last_outcome(self) -> AgentTurnOutcome | None:
        return self._last_outcome

    def run(self, user_input: str) -> str:
        """Execute one user turn through the agentic loop.

        Returns the agent's final text response.
        """
        # Record user message
        self._session.add_message("user", user_input)

        # Build LLM messages from session history
        messages = self._build_messages()

        # Build tool specs based on provider
        if self._provider.provider_name == "anthropic":
            tool_specs = self._tools.anthropic_specs(self._permission)
        else:
            tool_specs = self._tools.openai_specs(self._permission)

        # Agent loop
        rounds = 0
        tool_count = 0
        failed_tools: list[str] = []
        response = ChatResponse(content=None, tool_calls=[], stop_reason="end_turn")
        while rounds < self._max_rounds:
            rounds += 1

            try:
                system_prompt = _build_dynamic_prompt(
                    self._tools,
                    language=self._language,
                    prompt_registry=self._prompts,
                )
                response = self._provider.chat(
                    messages,
                    system=system_prompt,
                    tools=tool_specs if tool_specs else None,
                )
            except LLMError as e:
                error_msg = f"LLM error: {e}"
                self._session.add_message("assistant", error_msg)
                self._record_turn_outcome(AgentTurnOutcome(
                    stop_reason="llm_error",
                    failure_class="llm_error",
                    rounds=rounds,
                    tool_count=tool_count,
                    failed_tools=failed_tools,
                    user_input=user_input,
                    assistant_output=error_msg,
                ))
                return error_msg

            # If no tool calls → final response
            if not response.tool_calls:
                final = response.content or ""
                self._session.add_message("assistant", final)
                failure_class = "tool_runtime" if failed_tools else "none"
                self._record_turn_outcome(AgentTurnOutcome(
                    stop_reason=response.stop_reason or "end_turn",
                    failure_class=failure_class,
                    rounds=rounds,
                    tool_count=tool_count,
                    failed_tools=failed_tools,
                    user_input=user_input,
                    assistant_output=final,
                ))
                return final

            # Process tool calls
            tool_count += len(response.tool_calls)
            # First, record the assistant's response (may include text + tool calls)
            if self._provider.provider_name == "anthropic":
                messages, turn_failed = self._process_anthropic_tools(
                    messages, response
                )
            else:
                messages, turn_failed = self._process_openai_tools(
                    messages, response
                )
            failed_tools.extend(turn_failed)

        # Exceeded max rounds
        fallback = response.content or "I've reached the maximum number of tool calls for this turn."
        self._session.add_message("assistant", fallback)
        self._record_turn_outcome(AgentTurnOutcome(
            stop_reason="max_rounds",
            failure_class="tool_round_limit",
            rounds=rounds,
            tool_count=tool_count,
            failed_tools=failed_tools,
            user_input=user_input,
            assistant_output=fallback,
        ))
        return fallback

    def _build_messages(self) -> list[dict]:
        """Build LLM-compatible messages from session history."""
        return self._session.to_llm_messages(max_messages=40)

    def _execute_tool(self, tc: ToolCall) -> ToolResult:
        """Execute a tool call and fire callbacks."""
        if self._on_tool_call:
            self._on_tool_call(tc.name, tc.arguments)

        result = self._tools.execute(
            tc.name, tc.arguments, self._permission
        )

        # Record in session
        self._session.add_message(
            "tool_call",
            json.dumps({"name": tc.name, "arguments": tc.arguments}),
            tool_id=tc.id,
        )
        self._session.add_message(
            "tool_result",
            result.output,
            tool_id=tc.id, tool_name=tc.name, success=result.success,
        )

        if self._on_tool_result:
            self._on_tool_result(tc.name, result)

        return result

    def _process_anthropic_tools(
        self, messages: list[dict], response: ChatResponse
    ) -> tuple[list[dict], list[str]]:
        """Process tool calls for Anthropic format."""
        # Build assistant content blocks
        assistant_content: list[dict[str, Any]] = []
        if response.content:
            assistant_content.append({"type": "text", "text": response.content})
        for tc in response.tool_calls:
            assistant_content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })
        messages.append({"role": "assistant", "content": assistant_content})

        # Execute tools and build result message
        tool_results: list[dict[str, Any]] = []
        failed_tools: list[str] = []
        for tc in response.tool_calls:
            result = self._execute_tool(tc)
            if not result.success:
                failed_tools.append(tc.name)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": result.output,
            })
        messages.append({"role": "user", "content": tool_results})

        return messages, failed_tools

    def _process_openai_tools(
        self, messages: list[dict], response: ChatResponse
    ) -> tuple[list[dict], list[str]]:
        """Process tool calls for OpenAI format."""
        # Assistant message with tool calls
        tc_dicts = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in response.tool_calls
        ]
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": tc_dicts,
        }
        if response.content:
            assistant_msg["content"] = response.content
        messages.append(assistant_msg)

        # Tool result messages
        failed_tools: list[str] = []
        for tc in response.tool_calls:
            result = self._execute_tool(tc)
            if not result.success:
                failed_tools.append(tc.name)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result.output,
            })

        return messages, failed_tools

    def _record_turn_outcome(self, outcome: AgentTurnOutcome) -> None:
        self._last_outcome = outcome
        self._session.add_turn_summary(TurnSummary(
            stop_reason=outcome.stop_reason,
            failure_class=outcome.failure_class,
            rounds=outcome.rounds,
            tool_count=outcome.tool_count,
            failed_tools=outcome.failed_tools,
            user_input_preview=self._preview(outcome.user_input),
            assistant_output_preview=self._preview(outcome.assistant_output),
        ))

    @staticmethod
    def _preview(text: str, limit: int = 140) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."
