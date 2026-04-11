"""Tests for LLM provider interface and error handling.

Uses mocks to avoid real API calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from world0.llm.base import LLMError, LLMProvider


class TestLLMProviderInterface:
    def test_base_class_is_abstract(self):
        with pytest.raises(TypeError):
            LLMProvider()


class TestOpenAIProvider:
    def test_import_error_without_package(self):
        with patch.dict("sys.modules", {"openai": None}):
            from importlib import reload

            import world0.llm.openai as mod

            reload(mod)
            with pytest.raises(ImportError, match="openai"):
                mod.OpenAIProvider()

    def test_complete_json_calls_api(self):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"concepts": []}'
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from importlib import reload

            import world0.llm.openai as mod

            reload(mod)
            provider = mod.OpenAIProvider(model="gpt-4o-mini", api_key="test-key")
            result = provider.complete_json("system", "user")

            assert result == '{"concepts": []}'
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["model"] == "gpt-4o-mini"
            assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_api_error_raises_llm_error(self):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from importlib import reload

            import world0.llm.openai as mod

            reload(mod)
            provider = mod.OpenAIProvider(api_key="test-key")
            with pytest.raises(LLMError, match="OpenAI API call failed"):
                provider.complete_json("sys", "usr")


class TestAnthropicProvider:
    def test_import_error_without_package(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            from importlib import reload

            import world0.llm.anthropic as mod

            reload(mod)
            with pytest.raises(ImportError, match="anthropic"):
                mod.AnthropicProvider()

    def test_complete_json_calls_api(self):
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = '{"concepts": []}'
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from importlib import reload

            import world0.llm.anthropic as mod

            reload(mod)
            provider = mod.AnthropicProvider(
                model="claude-sonnet-4-20250514", api_key="test-key"
            )
            result = provider.complete_json("system", "user")

            assert result == '{"concepts": []}'
            mock_client.messages.create.assert_called_once()
            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["model"] == "claude-sonnet-4-20250514"
            assert call_kwargs["system"] == "system"

    def test_api_error_raises_llm_error(self):
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = RuntimeError("API down")

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from importlib import reload

            import world0.llm.anthropic as mod

            reload(mod)
            provider = mod.AnthropicProvider(api_key="test-key")
            with pytest.raises(LLMError, match="Anthropic API call failed"):
                provider.complete_json("sys", "usr")


class TestAzureOpenAIProvider:
    def test_import_error_without_package(self):
        with patch.dict("sys.modules", {"openai": None}):
            from importlib import reload

            import world0.llm.azure_openai as mod

            reload(mod)
            with pytest.raises(ImportError, match="openai"):
                mod.AzureOpenAIProvider()

    def test_complete_json_calls_api(self):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.AzureOpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"concepts": []}'
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from importlib import reload

            import world0.llm.azure_openai as mod

            reload(mod)
            provider = mod.AzureOpenAIProvider(
                model="gpt-4o-mini",
                api_key="test-key",
                azure_endpoint="https://example.openai.azure.com/",
                api_version="2024-10-21",
            )
            result = provider.complete_json("system", "user")

            assert result == '{"concepts": []}'
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["model"] == "gpt-4o-mini"

    def test_api_error_raises_llm_error(self):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.AzureOpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from importlib import reload

            import world0.llm.azure_openai as mod

            reload(mod)
            provider = mod.AzureOpenAIProvider(
                api_key="test-key",
                azure_endpoint="https://example.openai.azure.com/",
            )
            with pytest.raises(LLMError, match="Azure OpenAI API call failed"):
                provider.complete_json("sys", "usr")

    def test_uses_azure_openai_key_alias(self, monkeypatch):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.AzureOpenAI.return_value = mock_client

        monkeypatch.setenv("AZURE_OPENAI_KEY", "alias-key")
        monkeypatch.setenv(
            "AZURE_OPENAI_ENDPOINT",
            "https://example.openai.azure.com/",
        )

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from importlib import reload

            import world0.llm.azure_openai as mod

            reload(mod)
            mod.AzureOpenAIProvider(model="gpt-4o-mini")
            call_kwargs = mock_openai.AzureOpenAI.call_args[1]
            assert call_kwargs["api_key"] == "alias-key"
            assert call_kwargs["azure_endpoint"] == "https://example.openai.azure.com/"
