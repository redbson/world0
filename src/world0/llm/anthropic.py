"""Anthropic (Claude) provider implementation."""

from __future__ import annotations

from world0.llm.base import LLMError, LLMProvider


class AnthropicProvider(LLMProvider):
    """LLM provider backed by the Anthropic API.

    Requires the ``anthropic`` package::

        pip install world0[anthropic]

    Args:
        model: Model name (default: ``claude-sonnet-4-6``).
        api_key: Optional API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' package. "
                "Install with: pip install world0[anthropic]"
            )

        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key

        self._client = Anthropic(**kwargs)
        self._model = model

    def complete_json(self, system: str, user: str) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=0.1,
            )
            return response.content[0].text
        except Exception as e:
            raise LLMError(f"Anthropic API call failed: {e}") from e
