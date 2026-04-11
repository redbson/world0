"""Abstract LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Minimal interface for LLM calls used by World 0.

    World 0 only needs structured JSON output from a single prompt.
    This keeps the interface tight — no streaming, no tool use, no chat history.
    """

    @abstractmethod
    def complete_json(self, system: str, user: str) -> str:
        """Send a prompt and return the raw response text.

        The caller is responsible for parsing the JSON from the response.
        Implementations should configure the model to prefer JSON output
        where possible.

        Args:
            system: System-level instruction.
            user: User-level prompt content.

        Returns:
            The model's text response (expected to contain JSON).

        Raises:
            LLMError: On API or network failure.
        """
        ...


class LLMError(Exception):
    """Raised when an LLM API call fails."""
