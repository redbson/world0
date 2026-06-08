"""Tests for multi-provider routing."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from world0.agents.provider import (
    ChatProvider,
    MODEL_ALIASES,
    default_model_for_provider,
    detect_provider,
    normalize_provider_name,
    suggested_models_for_provider,
)


class TestDetectProvider:
    def test_explicit_prefix_anthropic(self):
        provider, model = detect_provider("anthropic/claude-sonnet-4-6")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-6"

    def test_explicit_prefix_openai(self):
        provider, model = detect_provider("openai/gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_explicit_prefix_azure_openai(self):
        provider, model = detect_provider("azure-openai/gpt-4o")
        assert provider == "azure-openai"
        assert model == "gpt-4o"

    def test_auto_detect_claude(self):
        provider, model = detect_provider("claude-sonnet-4-6")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-6"

    def test_auto_detect_gpt(self):
        provider, model = detect_provider("gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_alias_sonnet(self):
        provider, model = detect_provider("sonnet")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-6"

    def test_alias_opus(self):
        provider, model = detect_provider("opus")
        assert provider == "anthropic"
        assert model == "claude-opus-4-6"

    def test_alias_haiku(self):
        provider, model = detect_provider("haiku")
        assert provider == "anthropic"
        assert model == "claude-haiku-4-5-20251001"

    def test_alias_haiku_api_alias(self):
        provider, model = detect_provider("claude-haiku-4-5")
        assert provider == "anthropic"
        assert model == "claude-haiku-4-5-20251001"

    def test_alias_gpt4o(self):
        provider, model = detect_provider("gpt4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_alias_claude(self):
        provider, model = detect_provider("claude")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-6"

    def test_alias_codex(self):
        provider, model = detect_provider("codex")
        assert provider == "openai"
        assert model == "gpt-5.4"

    def test_o1_model(self):
        provider, model = detect_provider("o1-preview")
        assert provider == "openai"

    def test_unknown_model_fallback(self):
        # With no env vars, defaults to anthropic
        env_backup = {}
        for key in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_KEY",
        ):
            env_backup[key] = os.environ.pop(key, None)
        try:
            provider, model = detect_provider("some-custom-model")
            assert provider == "anthropic"  # default fallback
            assert model == "some-custom-model"
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val


class TestModelAliases:
    def test_all_aliases_defined(self):
        assert "claude" in MODEL_ALIASES
        assert "codex" in MODEL_ALIASES
        assert "opus" in MODEL_ALIASES
        assert "sonnet" in MODEL_ALIASES
        assert "haiku" in MODEL_ALIASES

    def test_aliases_resolve_to_valid_models(self):
        for alias, model in MODEL_ALIASES.items():
            assert len(model) > 0
            assert "/" not in model  # aliases should not include provider prefix


class TestProviderModelCatalog:
    def test_normalize_provider_name(self):
        assert normalize_provider_name("claude") == "anthropic"
        assert normalize_provider_name("codex") == "openai"
        assert normalize_provider_name("openai") == "openai"

    def test_default_model_for_provider(self):
        assert default_model_for_provider("openai") == "gpt-5.4"
        assert default_model_for_provider("codex") == "gpt-5.4"
        assert default_model_for_provider("anthropic") == "claude-sonnet-4-6"
        assert default_model_for_provider("claude") == "claude-sonnet-4-6"
        assert default_model_for_provider("azure-openai") == "gpt-5.4"

    def test_suggested_models_for_provider(self):
        assert "gpt-5.4" in suggested_models_for_provider("openai")
        assert "gpt-5.4" in suggested_models_for_provider("codex")
        assert "gpt-5.4-pro" in suggested_models_for_provider("openai")
        assert "gpt-5.4-mini" in suggested_models_for_provider("openai")
        assert "gpt-5.4-nano" in suggested_models_for_provider("openai")
        assert "claude-opus-4-6" in suggested_models_for_provider("anthropic")
        assert "claude-sonnet-4-6" in suggested_models_for_provider("claude")
        assert "claude-sonnet-4-6" in suggested_models_for_provider("anthropic")
        assert "claude-haiku-4-5-20251001" in suggested_models_for_provider("anthropic")
        assert "gpt-5.4" in suggested_models_for_provider("azure-openai")
        assert "Kimi-K2.5" in suggested_models_for_provider("azure-openai")


class TestChatProviderTokenParams:
    def test_gpt5_uses_max_completion_tokens(self):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_choice.message.tool_calls = []
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": mock_openai}):
            provider = ChatProvider(model="openai/gpt-5.4", api_key="test-key")
            resp = provider.chat([{"role": "user", "content": "hi"}])
            assert resp.content == "ok"
            kwargs = mock_client.chat.completions.create.call_args[1]
            assert "max_completion_tokens" in kwargs
            assert "max_tokens" not in kwargs

    def test_gpt4o_uses_max_tokens(self):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_choice.message.tool_calls = []
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": mock_openai}):
            provider = ChatProvider(model="openai/gpt-4o", api_key="test-key")
            resp = provider.chat([{"role": "user", "content": "hi"}])
            assert resp.content == "ok"
            kwargs = mock_client.chat.completions.create.call_args[1]
            assert "max_tokens" in kwargs
            assert "max_completion_tokens" not in kwargs
