"""Multi-provider LLM routing — inspired by claw-code's provider dispatch.

Supports model-prefix routing:
  - "anthropic/claude-sonnet-4-6"        → Anthropic provider
  - "openai/gpt-5.4"                     → OpenAI provider
  - "claude-sonnet-4-6"                  → Anthropic (auto-detected)
  - "gpt-5.4"                            → OpenAI (auto-detected)

Also supports model aliases:
  - "sonnet" → "claude-sonnet-4-6"
  - "opus"   → "claude-opus-4-6"
  - "haiku"  → "claude-haiku-4-5-20251001"

Provides a ChatProvider that supports multi-turn conversation
with tool use (extending the base LLMProvider).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from world0.llm.base import LLMError, LLMProvider

# ── Provider aliases ─────────────────────────────────────────────────
# Logical agent names map onto canonical provider ids, so callers can say
# "claude"/"codex" (the external coding agents) and resolve to the
# underlying LLM provider.

PROVIDER_ALIASES: dict[str, str] = {
    "claude": "anthropic",
    "codex": "openai",
}

# ── Model aliases (claw-code style) ─────────────────────────────────

MODEL_ALIASES: dict[str, str] = {
    "claude": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    # OpenAI's latest-model guide states gpt-5.4 is the newest model
    # powering Codex and Codex CLI; use that as the generic Codex alias.
    "codex": "gpt-5.4",
    "gpt5": "gpt-5.4",
    "gpt5-pro": "gpt-5.4-pro",
    "gpt5-mini": "gpt-5.4-mini",
    "gpt5-nano": "gpt-5.4-nano",
    "gpt4o": "gpt-4o",
    "gpt4o-mini": "gpt-4o-mini",
}

PROVIDER_MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "openai": {
        "default": "gpt-5.4",
        "models": [
            "gpt-5.4",
            "gpt-5.4-pro",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
        ],
    },
    "anthropic": {
        "default": "claude-sonnet-4-6",
        "models": [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ],
    },
    "azure-openai": {
        "default": "gpt-5.4",
        "models": [
            "gpt-5.4",
            "gpt-5.4-pro",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.3-codex",
            "gpt-5.2",
            "gpt-5.3-chat",
            "gpt-5.2-chat",
            "Kimi-K2.5",
            "Kimi-K2-Thinking",
        ],
    },
}


def normalize_provider_name(provider: str) -> str:
    """Normalize provider aliases to canonical provider ids."""
    clean = provider.strip().lower()
    return PROVIDER_ALIASES.get(clean, clean)


def default_model_for_provider(provider: str) -> str:
    """Return the default model name for a provider."""
    canonical = normalize_provider_name(provider)
    return PROVIDER_MODEL_CATALOG.get(canonical, {}).get("default", "")


def suggested_models_for_provider(provider: str) -> list[str]:
    """Return suggested model names for a provider."""
    canonical = normalize_provider_name(provider)
    return list(PROVIDER_MODEL_CATALOG.get(canonical, {}).get("models", []))

# ── Provider detection ───────────────────────────────────────────────

_ANTHROPIC_PREFIXES = ("claude-", "anthropic/")
_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "openai/", "chatgpt-")
_AZURE_OPENAI_PREFIXES = ("azure-openai/", "azure/")


def detect_provider(model: str) -> tuple[str, str]:
    """Detect provider from model name. Returns (provider, clean_model_name)."""
    # Explicit prefix routing (highest priority, like claw-code)
    if "/" in model:
        provider, name = model.split("/", 1)
        return normalize_provider_name(provider), name

    # Alias resolution
    resolved = MODEL_ALIASES.get(model.lower(), model)

    # Auto-detect from model name patterns
    lower = resolved.lower()
    for prefix in _AZURE_OPENAI_PREFIXES:
        if lower.startswith(prefix):
            return "azure-openai", resolved.split("/", 1)[1] if "/" in resolved else resolved
    for prefix in _ANTHROPIC_PREFIXES:
        if lower.startswith(prefix):
            return "anthropic", resolved
    for prefix in _OPENAI_PREFIXES:
        if lower.startswith(prefix):
            return "openai", resolved

    # Fallback: check env vars
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", resolved
    if os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY"):
        return "azure-openai", resolved
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", resolved

    return "anthropic", resolved


def create_provider(
    model: str = "sonnet",
    api_key: str | None = None,
    base_url: str | None = None,
    azure_endpoint: str | None = None,
    api_version: str | None = None,
) -> LLMProvider:
    """Create an LLM provider with auto-detection from model name."""
    provider_name, clean_model = detect_provider(model)
    provider_name = normalize_provider_name(provider_name)

    if provider_name == "anthropic":
        from world0.llm.anthropic import AnthropicProvider
        kwargs: dict[str, Any] = {"model": clean_model}
        if api_key:
            kwargs["api_key"] = api_key
        return AnthropicProvider(**kwargs)

    if provider_name == "openai":
        from world0.llm.openai import OpenAIProvider
        kwargs = {"model": clean_model}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAIProvider(**kwargs)

    if provider_name == "azure-openai":
        from world0.llm.azure_openai import AzureOpenAIProvider
        kwargs = {"model": clean_model}
        if api_key:
            kwargs["api_key"] = api_key
        if azure_endpoint:
            kwargs["azure_endpoint"] = azure_endpoint
        if api_version:
            kwargs["api_version"] = api_version
        return AzureOpenAIProvider(**kwargs)

    raise ValueError(f"Unknown provider: {provider_name}")


# ── Chat provider with tool use ──────────────────────────────────────

@dataclass
class ToolCall:
    """A tool call requested by the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """Response from a chat completion, may include tool calls."""
    content: str | None
    tool_calls: list[ToolCall]
    stop_reason: str  # "end_turn", "tool_use", "max_tokens"


class ChatProvider:
    """Extended LLM provider with multi-turn conversation and tool use.

    Wraps either Anthropic or OpenAI with a unified interface for
    the agentic loop.
    """

    def __init__(
        self,
        model: str = "sonnet",
        api_key: str | None = None,
        base_url: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
    ) -> None:
        self._provider_name, self._model = detect_provider(model)
        self._api_key = api_key
        self._base_url = base_url
        self._azure_endpoint = azure_endpoint
        self._api_version = api_version or "2024-10-21"
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        if self._provider_name == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError("pip install world0[anthropic]")
            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = Anthropic(**kwargs)

        elif self._provider_name == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("pip install world0[openai]")
            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)

        elif self._provider_name == "azure-openai":
            try:
                from openai import AzureOpenAI
            except ImportError:
                raise ImportError("pip install world0[openai]")
            kwargs = {"api_version": self._api_version}
            resolved_api_key = (
                self._api_key
                or os.environ.get("AZURE_OPENAI_API_KEY")
                or os.environ.get("AZURE_OPENAI_KEY")
            )
            resolved_endpoint = (
                self._azure_endpoint
                or os.environ.get("AZURE_OPENAI_ENDPOINT")
            )
            if resolved_api_key:
                kwargs["api_key"] = resolved_api_key
            if resolved_endpoint:
                kwargs["azure_endpoint"] = resolved_endpoint
            self._client = AzureOpenAI(**kwargs)

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        messages: list[dict],
        *,
        system: str = "",
        tools: list[dict] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        """Send a multi-turn chat with optional tool definitions."""
        if self._provider_name == "anthropic":
            return self._chat_anthropic(
                messages, system=system, tools=tools,
                temperature=temperature, max_tokens=max_tokens,
            )
        elif self._provider_name in ("openai", "azure-openai"):
            return self._chat_openai(
                messages, system=system, tools=tools,
                temperature=temperature, max_tokens=max_tokens,
            )
        raise LLMError(f"Unsupported provider: {self._provider_name}")

    def _chat_anthropic(
        self, messages, *, system, tools, temperature, max_tokens
    ) -> ChatResponse:
        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system:
                kwargs["system"] = system
            if tools:
                kwargs["tools"] = tools

            response = self._client.messages.create(**kwargs)

            content = None
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    content = block.text
                elif block.type == "tool_use":
                    tool_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    ))

            stop = "tool_use" if tool_calls else "end_turn"
            return ChatResponse(content=content, tool_calls=tool_calls, stop_reason=stop)

        except Exception as e:
            raise LLMError(f"Anthropic chat failed: {e}") from e

    def _chat_openai(
        self, messages, *, system, tools, temperature, max_tokens
    ) -> ChatResponse:
        try:
            full_messages = []
            if system:
                full_messages.append({"role": "system", "content": system})
            full_messages.extend(messages)

            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": full_messages,
                "temperature": temperature,
            }
            if tools:
                kwargs["tools"] = tools

            token_param = self._openai_token_param_name(self._model)
            kwargs[token_param] = max_tokens

            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception as e:
                alt_param = (
                    "max_completion_tokens"
                    if token_param == "max_tokens"
                    else "max_tokens"
                )
                if not self._is_unsupported_token_param_error(e, token_param):
                    raise
                kwargs.pop(token_param, None)
                kwargs[alt_param] = max_tokens
                response = self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message

            tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    ))

            stop = "tool_use" if tool_calls else "end_turn"
            return ChatResponse(
                content=msg.content,
                tool_calls=tool_calls,
                stop_reason=stop,
            )

        except Exception as e:
            raise LLMError(f"OpenAI chat failed: {e}") from e

    @staticmethod
    def _openai_token_param_name(model: str) -> str:
        lower = model.lower()
        if lower.startswith(("gpt-5", "o1", "o3", "o4")):
            return "max_completion_tokens"
        return "max_tokens"

    @staticmethod
    def _is_unsupported_token_param_error(error: Exception, param_name: str) -> bool:
        text = str(error)
        return (
            "Unsupported parameter" in text
            and param_name in text
        )
