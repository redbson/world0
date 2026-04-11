"""Azure OpenAI provider implementation."""

from __future__ import annotations

import os

from world0.llm.base import LLMError, LLMProvider


class AzureOpenAIProvider(LLMProvider):
    """LLM provider backed by Azure OpenAI.

    Requires the ``openai`` package::

        pip install world0[openai]

    Args:
        model: Azure deployment name.
        api_key: Optional API key. Falls back to ``AZURE_OPENAI_API_KEY`` or
            ``AZURE_OPENAI_KEY`` env vars.
        azure_endpoint: Optional Azure endpoint.
        api_version: API version string.
    """

    def __init__(
        self,
        model: str = "gpt-5.4",
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str = "2024-10-21",
    ) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError(
                "Azure OpenAI provider requires the 'openai' package. "
                "Install with: pip install world0[openai]"
            )

        resolved_api_key = (
            api_key
            or os.environ.get("AZURE_OPENAI_API_KEY")
            or os.environ.get("AZURE_OPENAI_KEY")
        )
        resolved_endpoint = (
            azure_endpoint
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
        )

        kwargs: dict = {"api_version": api_version}
        if resolved_api_key:
            kwargs["api_key"] = resolved_api_key
        if resolved_endpoint:
            kwargs["azure_endpoint"] = resolved_endpoint

        self._client = AzureOpenAI(**kwargs)
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
            raise LLMError(f"Azure OpenAI API call failed: {e}") from e
