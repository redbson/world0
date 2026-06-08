"""Tests for per-operation model configuration."""

import json

from world0.agents.pkm import PKMAgent
from world0.llm.base import LLMProvider
from world0.models import (
    OperationModelConfig,
    OperationModelSpec,
    load_operation_model_config,
    model_config_path,
    save_operation_model_config,
)


class ConfiguredExtractionLLM(LLMProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return json.dumps({
            "concepts": [{"name": "operation model config"}],
            "relations": [],
        })


def test_operation_model_config_round_trip(tmp_path):
    config = OperationModelConfig()
    config.set(OperationModelSpec(
        operation="extraction",
        provider="openai",
        model="gpt-5.4-mini",
        notes="cheap structured extraction",
    ))

    path = model_config_path(tmp_path)
    save_operation_model_config(path, config)
    loaded = load_operation_model_config(path)
    spec = loaded.get("extraction")

    assert spec is not None
    assert spec.provider == "openai"
    assert spec.model == "gpt-5.4-mini"
    assert spec.provider_model() == "openai/gpt-5.4-mini"
    assert spec.notes == "cheap structured extraction"


def test_pkm_uses_extraction_model_override(tmp_path, monkeypatch):
    config = OperationModelConfig()
    config.set(OperationModelSpec(
        operation="extraction",
        provider="openai",
        model="gpt-5.4-mini",
    ))
    save_operation_model_config(model_config_path(tmp_path), config)

    created: list[str] = []
    fake_llm = ConfiguredExtractionLLM()

    def fake_create_provider(**kwargs):
        created.append(kwargs["model"])
        return fake_llm

    monkeypatch.setattr("world0.agents.pkm.create_provider", fake_create_provider)

    agent = PKMAgent(store_path=tmp_path, llm=None)
    result = agent.learn("Model configuration can route extraction.")

    assert created == ["openai/gpt-5.4-mini"]
    assert fake_llm.calls
    assert "operation model config" in result
    assert agent.world.concepts.resolve("operation model config") is not None
