"""OpenAI provider implementation."""

from __future__ import annotations

from world0.llm.base import LLMError, LLMProvider


class OpenAIProvider(LLMProvider):
    """LLM provider backed by the OpenAI API.

    Requires the ``openai`` package::

        pip install world0[openai]

    Args:
        model: Model name (default: ``gpt-5.4``).
        api_key: Optional API key. Falls back to ``OPENAI_API_KEY`` env var.
        base_url: Optional custom base URL for compatible APIs.
    """

    def __init__(
        self,
        model: str = "gpt-5.4",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI provider requires the 'openai' package. "
                "Install with: pip install world0[openai]"
            )

        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._client = OpenAI(**kwargs)
        self._model = model

    def complete_json(self, system: str, user: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise LLMError(f"OpenAI API call failed: {e}") from e
