"""FastAPI backend for the PKM Agent GUI.

Provides REST API endpoints for all PKM Agent operations,
plus serves the embedded single-page frontend.
"""

from __future__ import annotations

import json
import os
from datetime import timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from world0.agents.pkm import PKMAgent
from world0.agents.provider import (
    default_model_for_provider,
    suggested_models_for_provider,
)
from world0.agents.state import AgentLifecycleStatus
from world0.llm.base import LLMProvider
from world0.schemas.relation import RelationType, semantic_relation_names

# Lazy import — FastAPI is an optional dependency
_app = None
_agent: PKMAgent | None = None


# ── Request/Response models ──────────────────────────────────────────

class LearnRequest(BaseModel):
    text: str
    task: str = "knowledge intake"
    source: str = ""


class AskRequest(BaseModel):
    query: str
    max_concepts: int = 15
    max_depth: int = 2


class ResearchRequest(BaseModel):
    topic: str
    focus: str = ""
    max_sources: int = 4
    save_findings: bool = True


class ConnectRequest(BaseModel):
    source: str
    target: str
    relation_type: str = "generic_relation"


class SearchRequest(BaseModel):
    query: str


class AgentChatRequest(BaseModel):
    message: str
    model: str = "sonnet"


class SettingsRequest(BaseModel):
    language: str = "en"
    provider: str = "none"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    azure_endpoint: str = ""
    api_version: str = "2024-10-21"
    auto_sediment_dialogue: bool = True
    dialogue_sediment_interval: int = 1


class PromptOverrideRequest(BaseModel):
    template: str


class SessionResumeRequest(BaseModel):
    session_id: str


class SessionRenameRequest(BaseModel):
    session_id: str = ""
    title: str


class SessionCompactRequest(BaseModel):
    session_id: str = ""


class ProjectionFeedbackRequest(BaseModel):
    useful: bool | None = None
    missing_concepts: list[str] = Field(default_factory=list)
    noisy_concepts: list[str] = Field(default_factory=list)
    weak_relations: list[str] = Field(default_factory=list)
    notes: str = ""


class SkillRequest(BaseModel):
    skill_name: str
    params: dict = Field(default_factory=dict)


class McpServerRequest(BaseModel):
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class SpaceCreateRequest(BaseModel):
    name: str
    description: str = ""
    activate: bool = True


class SpaceSwitchRequest(BaseModel):
    target: str  # name or id


class SpaceRenameRequest(BaseModel):
    target: str
    new_name: str


class SpaceDeleteRequest(BaseModel):
    target: str
    purge: bool = False


class MessageResponse(BaseModel):
    message: str
    type: str = "text"  # text, markdown, error
    tool_calls: list[dict] = Field(default_factory=list)


# ── App factory ──────────────────────────────────────────────────────

def create_app(
    store_path: str | Path = "~/.pkm_world",
    llm: LLMProvider | None = None,
    model: str = "sonnet",
    space_id: str | None = None,
) -> Any:
    """Create the FastAPI application with a PKMAgent backend."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    global _agent
    _agent = PKMAgent(store_path=store_path, llm=llm, space_id=space_id)
    runtime_provider = "none"
    runtime_model = ""
    if llm is not None:
        class_name = llm.__class__.__name__
        if class_name == "OpenAIProvider":
            runtime_provider = "openai"
        elif class_name == "AnthropicProvider":
            runtime_provider = "anthropic"
        elif class_name == "AzureOpenAIProvider":
            runtime_provider = "azure-openai"
        runtime_model = getattr(llm, "_model", "")
    if runtime_provider != "none" or llm is None:
        _agent.configure_runtime(
            language="en",
            provider=runtime_provider,
            model=runtime_model or (
                model if runtime_provider != "none"
                else ""
            ) or default_model_for_provider(runtime_provider),
        )
    else:
        _agent._language = "en"
        _agent._runtime_settings.update({
            "language": "en",
            "provider": "custom",
            "model": runtime_model,
        })

    # Initialize agentic mode if LLM is available
    _agentic_ready = False
    try:
        if runtime_provider != "none":
            runtime_model_name = runtime_model or model
            provider_model = f"{runtime_provider}/{runtime_model_name}"
            _agent.init_agentic(model=provider_model)
            _agentic_ready = True
    except Exception:
        pass  # Agentic mode unavailable (no API key, etc.)

    app = FastAPI(title="World 0 Concept World", version="0.2.0")

    def _space_payload() -> dict[str, Any]:
        registry = _agent.space_registry
        active = registry.active()
        return {
            "spaces": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "path": str(registry.path_for(s.id)),
                    "created_at": s.created_at.isoformat(),
                    "last_active_at": s.last_active_at.isoformat(),
                    "is_active": active is not None and active.id == s.id,
                }
                for s in registry.list()
            ],
            "active_space_id": active.id if active else None,
        }

    def _prompt_config_paths() -> list[Path]:
        from world0.prompts import prompt_config_path

        root = _agent._root_path
        paths = [prompt_config_path(root)]
        if _agent._store_path != root:
            paths.append(prompt_config_path(_agent._store_path))
        return paths

    def _prompt_config_target() -> Path:
        from world0.prompts import prompt_config_path

        return prompt_config_path(_agent._store_path)

    def _reload_prompt_registry() -> None:
        """Reload prompt config and rewire prompt consumers in the agent."""
        from world0.prompts import load_prompt_registry

        registry = load_prompt_registry(*_prompt_config_paths())
        _agent._prompts = registry
        _agent.world._prompts = registry
        extractor = getattr(_agent.world, "_extractor", None)
        if extractor is not None:
            extractor._prompts = registry
        _agent._skill_registry = None
        _agent._skill_executor = None

    def _target_prompt_overrides() -> dict[str, Any]:
        from world0.prompts import load_prompt_registry

        return load_prompt_registry(_prompt_config_target()).overrides()

    def _prompt_payload(prompt_id: str) -> dict[str, Any]:
        active_overrides = _target_prompt_overrides()
        spec = _agent.prompts.get(prompt_id)
        base = _agent.prompts.default(prompt_id)
        active = active_overrides.get(prompt_id)
        return {
            **spec.to_dict(),
            "default_template": base.template,
            "is_overridden": _agent.prompts.is_overridden(prompt_id),
            "has_active_override": active is not None,
            "active_override_template": active.template if active else "",
        }

    def _prompt_list_payload() -> dict[str, Any]:
        return {
            "config_path": str(_prompt_config_target()),
            "prompts": [_prompt_payload(spec.id) for spec in _agent.prompts.all()],
        }

    def _rebuild_agent(target_space_id: str) -> None:
        """Swap the process-global ``_agent`` to a different space.

        Preserves current runtime settings (language, provider, model,
        credentials) and re-initialises agentic mode if LLM is ready.
        """
        nonlocal _agentic_ready, llm
        global _agent

        prior_settings = dict(_agent.runtime_settings())
        prior_llm = _agent._llm
        prior_language = prior_settings.get("language", "en")
        prior_provider = prior_settings.get("provider", "none")
        prior_model = prior_settings.get("model", "")

        _agent = PKMAgent(
            store_path=store_path,
            llm=prior_llm,
            space_id=target_space_id,
        )
        if prior_provider and prior_provider != "custom":
            try:
                _agent.configure_runtime(
                    language=prior_language,
                    provider=prior_provider,
                    model=prior_model,
                    api_key=prior_settings.get("api_key") or None,
                    base_url=prior_settings.get("base_url") or None,
                    azure_endpoint=prior_settings.get("azure_endpoint") or None,
                    api_version=prior_settings.get("api_version") or None,
                    auto_sediment_dialogue=prior_settings.get(
                        "auto_sediment_dialogue", True
                    ),
                    dialogue_sediment_interval=prior_settings.get(
                        "dialogue_sediment_interval", 1
                    ),
                )
            except Exception:
                pass
        llm = _agent._llm

        _agentic_ready = False
        try:
            if prior_provider not in ("none", "custom"):
                effective_model = (
                    _agent.runtime_settings().get("model") or prior_model
                )
                if effective_model:
                    _agent.init_agentic(
                        model=f"{prior_provider}/{effective_model}"
                    )
                    _agentic_ready = _agent._chat_provider is not None
        except Exception:
            _agentic_ready = False

    def _session_payload(session) -> dict[str, Any]:
        session_state = _agent.session_state(session)
        latest_failure = _agent.latest_failure() if session.id == _agent.session.id else None
        return {
            "id": session.id,
            "title": session.title or "Untitled",
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "message_count": session.message_count(),
            "state": session_state.model_dump(mode="json"),
            "latest_failure": latest_failure.model_dump(mode="json") if latest_failure else None,
            "latest_projection_feedback": (
                session.latest_projection_feedback().model_dump(mode="json")
                if session.latest_projection_feedback()
                else None
            ),
            "turn_summaries": [
                {
                    "timestamp": item.timestamp.isoformat(),
                    "stop_reason": item.stop_reason,
                    "failure_class": item.failure_class,
                    "rounds": item.rounds,
                    "tool_count": item.tool_count,
                    "failed_tools": item.failed_tools,
                    "user_input_preview": item.user_input_preview,
                    "assistant_output_preview": item.assistant_output_preview,
                }
                for item in session.turn_summaries
            ],
            "compaction": (
                {
                    "updated_at": session.compaction.updated_at.isoformat(),
                    "summary": session.compaction.summary,
                    "open_loops": session.compaction.open_loops,
                    "key_concepts": session.compaction.key_concepts,
                    "covered_messages": session.compaction.covered_messages,
                }
                if session.compaction
                else None
            ),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata,
                }
                for msg in session.messages
            ],
        }

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _FRONTEND_HTML

    @app.post("/api/learn", response_model=MessageResponse)
    async def learn(req: LearnRequest):
        try:
            result = _agent.learn(req.text, task=req.task, source=req.source)
            _agent.record_direct_turn(
                req.text,
                result,
                mode="observe",
            )
            return MessageResponse(message=result, type="markdown")
        except Exception as e:
            return MessageResponse(message=str(e), type="error")

    @app.post("/api/ask", response_model=MessageResponse)
    async def ask(req: AskRequest):
        try:
            result = _agent.ask(
                req.query,
                max_concepts=req.max_concepts,
                max_depth=req.max_depth,
            )
            _agent.record_direct_turn(
                req.query,
                result,
                mode="project",
            )
            return MessageResponse(message=result, type="markdown")
        except Exception as e:
            return MessageResponse(message=str(e), type="error")

    @app.get("/api/explore/{concept_name}", response_model=MessageResponse)
    async def explore(concept_name: str):
        try:
            result = _agent.explore(concept_name)
            _agent.record_direct_turn(
                concept_name,
                result,
                mode="inspect",
            )
            return MessageResponse(message=result, type="markdown")
        except Exception as e:
            return MessageResponse(message=str(e), type="error")

    @app.post("/api/connect", response_model=MessageResponse)
    async def connect(req: ConnectRequest):
        try:
            result = _agent.connect(
                req.source, req.target, req.relation_type
            )
            _agent.record_direct_turn(
                f"{req.source} -> {req.relation_type} -> {req.target}",
                result,
                mode="relate",
            )
            return MessageResponse(message=result, type="markdown")
        except Exception as e:
            return MessageResponse(message=str(e), type="error")

    @app.post("/api/research", response_model=MessageResponse)
    async def research(req: ResearchRequest):
        try:
            result = _agent.research_topic(
                req.topic,
                focus=req.focus,
                max_sources=req.max_sources,
                save_findings=req.save_findings,
            )
            _agent.record_direct_turn(
                req.topic if not req.focus else f"{req.topic} [{req.focus}]",
                result,
                mode="research",
            )
            return MessageResponse(message=result, type="markdown")
        except Exception as e:
            return MessageResponse(message=str(e), type="error")

    @app.get("/api/search")
    async def search(q: str = ""):
        if not q.strip():
            return {"results": []}
        concepts = _agent.world.concepts.all()
        q_lower = q.strip().lower()
        matches = [
            c for c in concepts
            if q_lower in c.name.lower()
            or any(q_lower in a.lower() for a in c.aliases)
            or q_lower in c.description.lower()
        ]
        return {
            "results": [
                {
                    "id": c.id,
                    "representation": c.representation(),
                    "feature": c.representation_feature(),
                    "name": c.name,
                    "description": c.description,
                    "maturity": c.maturity.value,
                    "confidence": round(c.confidence, 3),
                    "activation_count": c.activation_count,
                }
                for c in sorted(
                    matches, key=lambda x: x.confidence, reverse=True
                )
            ]
        }

    @app.post("/api/reflect", response_model=MessageResponse)
    async def reflect():
        try:
            result = _agent.reflect()
            _agent.record_direct_turn(
                "Consolidate the current concept world.",
                result,
                mode="consolidate",
            )
            return MessageResponse(message=result, type="markdown")
        except Exception as e:
            return MessageResponse(message=str(e), type="error")

    @app.get("/api/status")
    async def status():
        st = _agent.world.status()
        all_concepts = _agent.world.concepts.all()
        top = sorted(
            all_concepts, key=lambda c: c.activation_count, reverse=True
        )[:20]

        return {
            "total_concepts": st.total_concepts,
            "total_relations": st.total_relations,
            "avg_confidence": round(st.avg_confidence, 3),
            "by_maturity": st.by_maturity,
            "last_reflect": (
                st.last_reflect.strftime("%Y-%m-%d %H:%M")
                if st.last_reflect
                else None
            ),
            "top_concepts": [
                {
                    "id": c.id,
                    "representation": c.representation(),
                    "feature": c.representation_feature(),
                    "name": c.name,
                    "maturity": c.maturity.value,
                    "confidence": round(c.confidence, 3),
                    "activation_count": c.activation_count,
                    "description": c.description,
                }
                for c in top
            ],
        }

    @app.get("/api/concepts")
    async def concepts():
        all_c = _agent.world.concepts.all()
        return {
            "concepts": [
                {
                    "id": c.id,
                    "representation": c.representation(),
                    "feature": c.representation_feature(),
                    "name": c.name,
                    "maturity": c.maturity.value,
                    "confidence": round(c.confidence, 3),
                    "activation_count": c.activation_count,
                    "description": c.description,
                    "aliases": c.aliases,
                }
                for c in sorted(all_c, key=lambda x: x.confidence, reverse=True)
            ]
        }

    @app.get("/api/concepts/{concept_name}/card")
    async def concept_card(concept_name: str):
        card = _agent.concept_card(concept_name)
        if not card:
            return JSONResponse(
                {"error": f"Concept '{concept_name}' not found."},
                status_code=404,
            )
        return {"card": card}

    @app.get("/api/graph")
    async def graph():
        concepts = _agent.world.concepts.all()
        relations = _agent.world.relations.all()
        id_to_name = {c.id: c.name for c in concepts}
        id_to_representation = {c.id: c.representation() for c in concepts}

        nodes = []
        for c in concepts:
            connections = len(_agent.world.relations.for_concept(c.id))
            nodes.append({
                "id": c.id,
                "representation": c.representation(),
                "feature": c.representation_feature(),
                "name": c.name,
                "confidence": round(c.confidence, 4),
                "maturity": c.maturity.value,
                "activation_count": c.activation_count,
                "connections": connections,
                "description": c.description,
            })

        edges = []
        for r in relations:
            if r.source_id in id_to_name and r.target_id in id_to_name:
                edges.append({
                    "source": r.source_id,
                    "target": r.target_id,
                    "relation_type": r.relation_type.value,
                    "semantic_relation": r.semantic_relation,
                    "structural_strength": round(r.structural_strength, 4),
                    "propagation_strength": round(r.propagation_strength, 4),
                    "weight": round(r.weight, 4),
                    "probability": round(r.probability, 4),
                    "reinforcement_count": r.reinforcement_count,
                    "source_name": id_to_name[r.source_id],
                    "target_name": id_to_name[r.target_id],
                    "source_representation": id_to_representation[r.source_id],
                    "target_representation": id_to_representation[r.target_id],
                })

        return {"nodes": nodes, "edges": edges}

    @app.get("/api/relation_types")
    async def relation_types():
        return {
            "types": semantic_relation_names(),
            "axes": [rt.value for rt in RelationType],
        }

    # ── Agentic endpoints ─────────────────────────────────────────

    @app.post("/api/agent/chat", response_model=MessageResponse)
    async def agent_chat(req: AgentChatRequest):
        """Agentic chat — LLM autonomously decides which tools to call."""
        current_state = _agent.agent_state()
        if not current_state.agentic_ready:
            return MessageResponse(
                message=current_state.reason or "Agentic mode unavailable.",
                type="error",
            )
        tool_log: list[dict] = []

        def on_tool_call(name, args):
            tool_log.append({"tool": name, "args": args, "phase": "call"})

        def on_tool_result(name, result):
            tool_log.append({
                "tool": name,
                "success": result.success,
                "output_preview": result.output[:200],
                "phase": "result",
            })

        try:
            response = _agent.agent_chat(
                req.message,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
            )
            _agent.save_session()
            return MessageResponse(
                message=response,
                type="markdown",
                tool_calls=tool_log,
            )
        except Exception as e:
            return MessageResponse(message=str(e), type="error")

    @app.get("/api/agent/status")
    async def agent_status():
        """Return agentic mode status and available tools."""
        tools = _agent.tool_registry
        current_session = _agent.session
        settings = _agent.runtime_settings()
        current_state = _agent.agent_state()
        if current_state.status in (
            AgentLifecycleStatus.BLOCKED,
            AgentLifecycleStatus.DEGRADED,
            AgentLifecycleStatus.FAILED,
        ):
            unavailable_reason = current_state.reason
        else:
            unavailable_reason = None
        return {
            "agentic_ready": current_state.agentic_ready,
            "llm_enabled": current_state.llm_enabled,
            "store_path": str(Path(store_path).expanduser()),
            "language": _agent.language,
            "provider": current_state.provider,
            "model": current_state.model,
            "unavailable_reason": unavailable_reason,
            "settings": settings,
            "state": current_state.model_dump(mode="json"),
            "latest_failure": (
                _agent.latest_failure().model_dump(mode="json")
                if _agent.latest_failure()
                else None
            ),
            "providers": [
                {"id": "none", "label": "None"},
                {"id": "claude", "label": "Claude (Anthropic)"},
                {"id": "codex", "label": "Codex (OpenAI)"},
                {"id": "openai", "label": "OpenAI"},
                {"id": "anthropic", "label": "Anthropic"},
                {"id": "azure-openai", "label": "Azure OpenAI"},
            ],
            "provider_env": {
                "codex": {
                    "api_key_env": "OPENAI_API_KEY",
                    "available": bool(os.environ.get("OPENAI_API_KEY")),
                },
                "claude": {
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "available": bool(os.environ.get("ANTHROPIC_API_KEY")),
                },
                "openai": {
                    "api_key_env": "OPENAI_API_KEY",
                    "available": bool(os.environ.get("OPENAI_API_KEY")),
                },
                "anthropic": {
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "available": bool(os.environ.get("ANTHROPIC_API_KEY")),
                },
                "azure-openai": {
                    "api_key_env": "AZURE_OPENAI_API_KEY / AZURE_OPENAI_KEY",
                    "endpoint_env": "AZURE_OPENAI_ENDPOINT",
                    "available": bool(
                        os.environ.get("AZURE_OPENAI_API_KEY")
                        or os.environ.get("AZURE_OPENAI_KEY")
                    ),
                },
            },
            "languages": [
                {"id": "zh", "label": "中文"},
                {"id": "en", "label": "English"},
            ],
            "suggested_models": {
                "codex": suggested_models_for_provider("codex"),
                "claude": suggested_models_for_provider("claude"),
                "openai": suggested_models_for_provider("openai"),
                "anthropic": suggested_models_for_provider("anthropic"),
                "azure-openai": suggested_models_for_provider("azure-openai"),
            },
            "external_agents": _agent.available_external_agents(),
            "last_external_consultation": _agent.session.metadata.get(
                "last_external_consultation"
            ),
            "current_session": {
                "id": current_session.id,
                "title": current_session.title or "Untitled",
                "message_count": current_session.message_count(),
                "state": _agent.session_state(current_session).model_dump(mode="json"),
                "last_projection": _agent.latest_projection_snapshot(),
                "last_dialogue_sediment": _agent.latest_dialogue_sediment(),
                "latest_turn": (
                    {
                        "stop_reason": current_session.latest_turn_summary().stop_reason,
                        "failure_class": current_session.latest_turn_summary().failure_class,
                        "rounds": current_session.latest_turn_summary().rounds,
                        "tool_count": current_session.latest_turn_summary().tool_count,
                    }
                    if current_session.latest_turn_summary()
                    else None
                ),
                "has_compaction": current_session.compaction is not None,
            },
            "recent_session_count": len(_agent.list_session_summaries(limit=20)),
            "tools": [
                {"name": t.name, "description": t.description, "permission": t.permission.value}
                for t in tools.all()
            ],
            "tool_count": len(tools),
        }

    @app.get("/api/settings")
    async def get_settings():
        return _agent.runtime_settings()

    @app.post("/api/settings")
    async def update_settings(req: SettingsRequest):
        nonlocal _agentic_ready, llm
        try:
            _agent.configure_runtime(
                language=req.language,
                provider=req.provider,
                model=req.model,
                api_key=req.api_key or None,
                base_url=req.base_url or None,
                azure_endpoint=req.azure_endpoint or None,
                api_version=req.api_version or None,
                auto_sediment_dialogue=req.auto_sediment_dialogue,
                dialogue_sediment_interval=req.dialogue_sediment_interval,
            )
            llm = _agent._llm
            _agentic_ready = _agent._chat_provider is not None
            current_state = _agent.agent_state()
            return {
                "success": True,
                "settings": _agent.runtime_settings(),
                "agentic_ready": current_state.agentic_ready,
                "state": current_state.model_dump(mode="json"),
            }
        except Exception as e:
            return JSONResponse(
                {"success": False, "error": str(e)},
                status_code=400,
            )

    # ── Prompt configuration endpoints ─────────────────────────────

    @app.get("/api/prompts")
    async def list_prompts():
        return _prompt_list_payload()

    @app.get("/api/prompts/export")
    async def export_prompts():
        from world0.prompts import export_prompt_config

        return export_prompt_config(_agent.prompts)

    @app.get("/api/prompts/{prompt_id}")
    async def get_prompt(prompt_id: str):
        try:
            return {"prompt": _prompt_payload(prompt_id)}
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    @app.post("/api/prompts/{prompt_id}")
    async def update_prompt(prompt_id: str, req: PromptOverrideRequest):
        from world0.prompts import (
            PromptSpec,
            load_prompt_registry,
            save_prompt_overrides,
        )

        template = req.template
        if not template.strip():
            return JSONResponse(
                {"success": False, "error": "prompt template must not be empty"},
                status_code=400,
            )

        target = _prompt_config_target()
        target_registry = load_prompt_registry(target)
        try:
            base = target_registry.default(prompt_id)
        except KeyError as exc:
            return JSONResponse({"success": False, "error": str(exc)}, status_code=404)

        data = base.to_dict(include_id=False)
        data["template"] = template
        data["variables"] = []
        target_registry.set_override(PromptSpec.from_dict(prompt_id, data))
        save_prompt_overrides(target, target_registry)
        _reload_prompt_registry()
        return {
            "success": True,
            "config_path": str(target),
            "prompt": _prompt_payload(prompt_id),
        }

    @app.post("/api/prompts/{prompt_id}/reset")
    async def reset_prompt(prompt_id: str):
        from world0.prompts import load_prompt_registry, save_prompt_overrides

        target = _prompt_config_target()
        target_registry = load_prompt_registry(target)
        try:
            target_registry.default(prompt_id)
        except KeyError as exc:
            return JSONResponse({"success": False, "error": str(exc)}, status_code=404)
        target_registry.clear_override(prompt_id)
        save_prompt_overrides(target, target_registry)
        _reload_prompt_registry()
        return {
            "success": True,
            "config_path": str(target),
            "prompt": _prompt_payload(prompt_id),
        }

    # ── Session endpoints ─────────────────────────────────────────

    @app.get("/api/sessions")
    async def list_sessions():
        return {"sessions": _agent.list_session_summaries(limit=20)}

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        session = _agent.get_session(session_id)
        if not session:
            return JSONResponse(
                {"error": f"Session '{session_id}' not found."},
                status_code=404,
            )
        return {"session": _session_payload(session)}

    @app.post("/api/sessions/save")
    async def save_session():
        sid = _agent.save_session()
        return {"session_id": sid}

    @app.post("/api/sessions/resume")
    async def resume_session(req: SessionResumeRequest):
        ok = _agent.resume_session(req.session_id)
        payload = None
        if ok:
            session = _agent.session
            payload = _session_payload(session)
        return {
            "success": ok,
            "session_id": req.session_id,
            "session": payload,
        }

    @app.post("/api/sessions/rename")
    async def rename_session(req: SessionRenameRequest):
        try:
            sid = _agent.rename_session(
                req.title,
                session_id=req.session_id or None,
            )
            session = _agent.get_session(sid)
            return {"success": True, "session": _session_payload(session)}
        except Exception as e:
            return JSONResponse(
                {"success": False, "error": str(e)},
                status_code=400,
            )

    @app.post("/api/sessions/compact")
    async def compact_session(req: SessionCompactRequest):
        try:
            result = _agent.compact_session(req.session_id or None)
            session = _agent.get_session(result["session_id"])
            return {
                "success": True,
                "compacted": result["compacted"],
                "covered_messages": result["covered_messages"],
                "session": _session_payload(session),
            }
        except Exception as e:
            return JSONResponse(
                {"success": False, "error": str(e)},
                status_code=400,
            )

    @app.get("/api/agent/latest_failure")
    async def latest_session_failure():
        failure = _agent.latest_failure()
        return {
            "failure": failure.model_dump(mode="json") if failure else None,
        }

    @app.get("/api/projection/last")
    async def last_projection():
        return {
            "projection": _agent.latest_projection_snapshot(),
            "latest_feedback": (
                _agent.latest_projection_feedback().model_dump(mode="json")
                if _agent.latest_projection_feedback()
                else None
            ),
        }

    @app.post("/api/projection/feedback")
    async def projection_feedback(req: ProjectionFeedbackRequest):
        try:
            result = _agent.apply_projection_feedback(
                useful=req.useful,
                missing_concepts=req.missing_concepts,
                noisy_concepts=req.noisy_concepts,
                weak_relations=req.weak_relations,
                notes=req.notes,
            )
            return {"success": True, **result}
        except Exception as e:
            return JSONResponse(
                {"success": False, "error": str(e)},
                status_code=400,
            )

    @app.post("/api/sessions/new")
    async def new_session():
        sid = _agent.new_session()
        return {"session_id": sid}

    # ── Skill endpoints ───────────────────────────────────────────

    @app.get("/api/skills")
    async def list_skills():
        skills = _agent.skills.all()
        return {
            "skills": [s.to_dict() for s in skills],
        }

    @app.post("/api/skills/run", response_model=MessageResponse)
    async def run_skill(req: SkillRequest):
        current_state = _agent.agent_state()
        if not current_state.agentic_ready:
            return MessageResponse(
                message=current_state.reason or "Skill execution requires agentic mode.",
                type="error",
            )
        tool_log: list[dict] = []

        def on_tool_call(name, args):
            tool_log.append({"tool": name, "args": args, "phase": "call"})

        def on_tool_result(name, result):
            tool_log.append({
                "tool": name, "success": result.success,
                "output_preview": result.output[:200], "phase": "result",
            })

        try:
            result = _agent.run_skill(
                req.skill_name,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
                **req.params,
            )
            _agent.save_session()
            return MessageResponse(
                message=result, type="markdown", tool_calls=tool_log,
            )
        except Exception as e:
            return MessageResponse(message=str(e), type="error")

    # ── MCP endpoints ─────────────────────────────────────────────

    @app.get("/api/mcp/status")
    async def mcp_status():
        try:
            statuses = _agent.mcp.server_statuses()
            return {
                "servers": [
                    {
                        "name": s.name,
                        "status": s.status,
                        "tool_count": s.tool_count,
                        "resource_count": s.resource_count,
                        "error": s.error,
                    }
                    for s in statuses
                ],
                "total_tools": _agent.mcp.total_tools,
                "connected": _agent.mcp.connected_count,
            }
        except Exception as e:
            return {"servers": [], "total_tools": 0, "connected": 0, "error": str(e)}

    @app.post("/api/mcp/add", response_model=MessageResponse)
    async def mcp_add(req: McpServerRequest):
        try:
            result = _agent.add_mcp_server(
                req.name, req.command, req.args, req.env,
            )
            return MessageResponse(message=result, type="markdown")
        except Exception as e:
            return MessageResponse(message=str(e), type="error")

    @app.post("/api/mcp/load", response_model=MessageResponse)
    async def mcp_load():
        try:
            result = _agent.load_mcp_config()
            return MessageResponse(message=result, type="markdown")
        except Exception as e:
            return MessageResponse(message=str(e), type="error")

    # ── Space endpoints ───────────────────────────────────────────

    @app.get("/api/spaces")
    async def list_spaces():
        return _space_payload()

    @app.post("/api/spaces/create")
    async def create_space(req: SpaceCreateRequest):
        try:
            space = _agent.space_registry.create(
                req.name, description=req.description
            )
        except ValueError as exc:
            return JSONResponse(
                {"success": False, "error": str(exc)}, status_code=400
            )
        if req.activate:
            _rebuild_agent(space.id)
        return {
            "success": True,
            "space": {
                "id": space.id,
                "name": space.name,
                "description": space.description,
            },
            **_space_payload(),
        }

    @app.post("/api/spaces/use")
    async def use_space(req: SpaceSwitchRequest):
        space = _agent.space_registry.resolve(req.target)
        if space is None:
            return JSONResponse(
                {"success": False, "error": f"space {req.target!r} not found"},
                status_code=404,
            )
        _rebuild_agent(space.id)
        return {"success": True, **_space_payload()}

    @app.post("/api/spaces/rename")
    async def rename_space(req: SpaceRenameRequest):
        registry = _agent.space_registry
        space = registry.resolve(req.target)
        if space is None:
            return JSONResponse(
                {"success": False, "error": f"space {req.target!r} not found"},
                status_code=404,
            )
        try:
            registry.rename(space.id, req.new_name)
        except ValueError as exc:
            return JSONResponse(
                {"success": False, "error": str(exc)}, status_code=400
            )
        return {"success": True, **_space_payload()}

    @app.post("/api/spaces/delete")
    async def delete_space(req: SpaceDeleteRequest):
        registry = _agent.space_registry
        space = registry.resolve(req.target)
        if space is None:
            return JSONResponse(
                {"success": False, "error": f"space {req.target!r} not found"},
                status_code=404,
            )
        active = registry.active()
        was_active = active is not None and active.id == space.id
        registry.delete(space.id, purge_data=req.purge)
        if was_active:
            new_active = registry.active()
            if new_active is not None:
                _rebuild_agent(new_active.id)
        return {"success": True, **_space_payload()}

    return app


# ── Embedded Frontend ────────────────────────────────────────────────

_FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>World 0 — Concept World</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
:root {
  --bg-primary: #0a0e15;
  --bg-secondary: #10151d;
  --bg-tertiary: #161c25;
  --bg-card: #1c232e;
  --bg-elevated: #232b38;
  --border: #232b38;
  --border-strong: #2f3a49;
  --border-light: #1a2029;
  --text-primary: #e6edf3;
  --text-secondary: #9aa5b1;
  --text-muted: #5b6573;
  --accent: #6ea8ff;
  --accent-strong: #4a8fff;
  --accent-dim: rgba(110, 168, 255, 0.14);
  --green: #4ec96b;
  --green-dim: rgba(78, 201, 107, 0.14);
  --orange: #e0a93b;
  --orange-dim: rgba(224, 169, 59, 0.14);
  --purple: #c69dff;
  --purple-dim: rgba(198, 157, 255, 0.14);
  --red: #ff6a5b;
  --pink: #f78fc1;
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.35);
  --shadow-md: 0 6px 18px rgba(0, 0, 0, 0.4);
  --shadow-lg: 0 16px 40px rgba(0, 0, 0, 0.55);
  --radius-sm: 4px;
  --radius-md: 7px;
  --radius-lg: 10px;
  --radius-xl: 14px;
  --mode-color: var(--accent);
  --sidebar-width: 280px;
  --titlebar-height: 38px;
  --input-height: 60px;
}

* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { height: 100%; overflow: hidden; }

body {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue",
               "PingFang SC", "Microsoft YaHei", sans-serif;
  background: var(--bg-primary);
  background-image:
    radial-gradient(ellipse 800px 500px at 15% 0%, rgba(110, 168, 255, 0.05), transparent 60%),
    radial-gradient(ellipse 800px 500px at 85% 100%, rgba(198, 157, 255, 0.04), transparent 60%);
  background-attachment: fixed;
  color: var(--text-primary);
  font-size: 13px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
  letter-spacing: 0.1px;
}

/* ── Titlebar (draggable for native window) ── */
#titlebar {
  height: var(--titlebar-height);
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
  -webkit-app-region: drag;
  user-select: none;
  position: relative;
  z-index: 100;
}
#titlebar .title {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  letter-spacing: 0.3px;
}
#titlebar .traffic-spacer { width: 72px; }

/* ── Main layout ── */
#app {
  display: flex;
  height: calc(100vh - var(--titlebar-height));
}

/* ── Sidebar ── */
#sidebar {
  width: var(--sidebar-width);
  min-width: var(--sidebar-width);
  background: var(--bg-secondary);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.sidebar-header {
  padding: 16px 16px 10px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.sidebar-header h2 {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: var(--text-secondary);
  flex: 1;
}
.sidebar-badge {
  font-size: 10px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 10px;
  background: var(--accent-dim);
  color: var(--accent);
  font-variant-numeric: tabular-nums;
}

#sidebar-search {
  margin: 0 12px 10px;
  padding: 7px 12px;
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  color: var(--text-primary);
  font-size: 12px;
  outline: none;
  width: calc(100% - 24px);
  transition: border-color 0.15s, background 0.15s;
}
#sidebar-search:focus {
  border-color: var(--accent);
  background: var(--bg-card);
}
#sidebar-search::placeholder { color: var(--text-muted); }

#concept-list {
  flex: 1;
  overflow-y: auto;
  padding: 2px 8px 8px;
}

.concept-item {
  padding: 8px 10px;
  border-radius: var(--radius-md);
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 10px;
  transition: background 0.15s, transform 0.15s;
  margin-bottom: 2px;
}
.concept-item:hover {
  background: var(--bg-tertiary);
  transform: translateX(1px);
}
.concept-item.active {
  background: var(--bg-card);
  box-shadow: inset 0 0 0 1px var(--border-strong);
}

.concept-dot {
  width: 3px;
  height: 16px;
  border-radius: 2px;
  flex-shrink: 0;
}
.concept-name {
  flex: 1;
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.concept-meta {
  font-size: 10px;
  color: var(--text-muted);
  flex-shrink: 0;
  font-variant-numeric: tabular-nums;
}

/* Maturity colors */
.mat-embryonic { background: var(--orange); }
.mat-developing { background: var(--accent); }
.mat-established { background: var(--green); }
.mat-core { background: var(--purple); }
.mat-fading { background: var(--text-muted); }

/* ── Stats bar ── */
#stats-bar {
  padding: 12px 14px;
  border-top: 1px solid var(--border);
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 0;
  background: var(--bg-secondary);
}
.stat-box {
  text-align: center;
  padding: 4px 6px;
  position: relative;
}
.stat-box + .stat-box::before {
  content: "";
  position: absolute;
  left: 0;
  top: 20%;
  bottom: 20%;
  width: 1px;
  background: var(--border);
}
.stat-val {
  font-size: 17px;
  font-weight: 700;
  color: var(--accent);
  font-variant-numeric: tabular-nums;
  line-height: 1.1;
}
.stat-lbl {
  font-size: 9px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-top: 3px;
  font-weight: 600;
}

/* ── Environment ── */
#environment-card {
  margin: 0 12px 10px;
  padding: 10px;
  border-radius: var(--radius-lg);
  background: linear-gradient(180deg, var(--bg-card), var(--bg-tertiary));
  border: 1px solid var(--border);
  box-shadow: var(--shadow-sm);
}
.environment-title {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.9px;
  color: var(--text-secondary);
  margin-bottom: 8px;
  font-weight: 700;
}
.environment-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.environment-item {
  padding: 8px;
  border-radius: var(--radius-md);
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--border-light);
}
.environment-item strong {
  display: block;
  font-size: 10px;
  color: var(--text-muted);
  margin-bottom: 4px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
}
.environment-item span {
  display: block;
  font-size: 12px;
  color: var(--text-primary);
}
.environment-path {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--border-light);
  color: var(--text-secondary);
  font-size: 11px;
  line-height: 1.45;
  word-break: break-all;
}
.environment-hint {
  margin-top: 8px;
  color: var(--orange);
  font-size: 11px;
  line-height: 1.45;
}

/* ── Sidebar actions ── */
#sidebar-actions {
  padding: 10px 12px 12px;
  border-top: 1px solid var(--border);
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 6px;
}
.sidebar-btn {
  padding: 7px 4px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--bg-tertiary);
  color: var(--text-secondary);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.3px;
  cursor: pointer;
  text-align: center;
  transition: all 0.15s ease;
}
.sidebar-btn:hover {
  background: var(--bg-card);
  color: var(--accent);
  border-color: var(--accent);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}
.sidebar-btn:active { transform: translateY(0); }

/* ── Main content area ── */
#main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

/* ── Tab bar ── */
#tab-bar {
  display: flex;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  padding: 0 16px;
  gap: 4px;
}
.tab {
  padding: 11px 18px 10px;
  font-size: 12px;
  font-weight: 500;
  color: var(--text-muted);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.15s ease;
  user-select: none;
  position: relative;
  letter-spacing: 0.2px;
}
.tab:hover { color: var(--text-secondary); }
.tab.active {
  color: var(--text-primary);
  border-bottom-color: var(--accent);
  font-weight: 600;
}

/* ── Chat panel ── */
#chat-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

#messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px 28px;
  scroll-behavior: smooth;
}

.message {
  margin-bottom: 18px;
  display: flex;
  gap: 12px;
  animation: fadeIn 0.25s ease;
  max-width: 880px;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

.message-avatar {
  width: 30px; height: 30px;
  border-radius: 8px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 700;
  box-shadow: var(--shadow-sm);
}
.msg-user .message-avatar {
  background: linear-gradient(135deg, var(--accent-strong), var(--accent));
  color: #fff;
}
.msg-agent .message-avatar {
  background: linear-gradient(135deg, #6c4eb3, var(--purple));
  color: #fff;
}
.msg-system .message-avatar {
  background: var(--bg-card);
  color: var(--text-muted);
  border: 1px solid var(--border);
}

.message-content {
  flex: 1;
  min-width: 0;
}
.message-header {
  font-size: 11px;
  color: var(--text-muted);
  margin-bottom: 4px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.message-header .name { font-weight: 600; color: var(--text-secondary); }
.message-body {
  font-size: 13px;
  line-height: 1.6;
  color: var(--text-primary);
  word-wrap: break-word;
}
.message-body pre {
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 12px;
  overflow-x: auto;
  font-size: 12px;
  margin: 6px 0;
  font-family: "SF Mono", "Fira Code", Menlo, monospace;
}
.message-body code {
  font-family: "SF Mono", "Fira Code", Menlo, monospace;
  background: var(--bg-tertiary);
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 12px;
}
.message-body strong { color: var(--accent); font-weight: 600; }
.message-body h1, .message-body h2, .message-body h3 {
  margin: 10px 0 6px;
  color: var(--text-primary);
}
.message-body h1 { font-size: 16px; }
.message-body h2 { font-size: 14px; }
.message-body h3 { font-size: 13px; color: var(--text-secondary); }
.message-body ul, .message-body ol {
  padding-left: 18px;
  margin: 4px 0;
}
.message-body li { margin: 2px 0; }
.message-body hr {
  border: none;
  border-top: 1px solid var(--border);
  margin: 10px 0;
}
.message-body em { color: var(--text-secondary); }
.message-body .concept-tag {
  display: inline-block;
  padding: 1px 7px;
  border-radius: 4px;
  font-size: 11px;
  background: var(--accent-dim);
  color: var(--accent);
  cursor: pointer;
  margin: 1px 2px;
}
.message-body .concept-tag:hover { background: var(--accent); color: #fff; }
.msg-error .message-body { color: var(--red); }

/* Typing indicator */
.typing-indicator {
  display: flex; gap: 4px; padding: 4px 0;
}
.typing-indicator span {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--text-muted);
  animation: typing 1.2s infinite;
}
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
@keyframes typing {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
  30% { transform: translateY(-4px); opacity: 1; }
}

/* ── Input area ── */
#input-area {
  padding: 14px 24px 16px;
  background: linear-gradient(to bottom, transparent, var(--bg-secondary) 30%);
  border-top: 1px solid var(--border);
  --mode-color: var(--accent);
}
#input-area[data-mode="agent"]   { --mode-color: var(--purple); }
#input-area[data-mode="skill"]   { --mode-color: var(--green); }
#input-area[data-mode="ask"]     { --mode-color: var(--accent); }
#input-area[data-mode="learn"]   { --mode-color: var(--orange); }
#input-area[data-mode="explore"] { --mode-color: var(--pink); }
#input-area[data-mode="connect"] { --mode-color: #39d2c0; }

#input-row {
  display: flex;
  gap: 0;
  align-items: stretch;
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 4px;
  transition: border-color 0.2s, box-shadow 0.2s;
  box-shadow: var(--shadow-sm);
}
#input-row:focus-within {
  border-color: var(--mode-color);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--mode-color) 18%, transparent),
              var(--shadow-sm);
}

#mode-select {
  padding: 6px 26px 6px 12px;
  background: transparent;
  border: none;
  border-right: 1px solid var(--border);
  border-radius: 0;
  color: var(--mode-color);
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  outline: none;
  cursor: pointer;
  -webkit-appearance: none;
  min-width: 92px;
  background-image: linear-gradient(45deg, transparent 50%, currentColor 50%),
                    linear-gradient(135deg, currentColor 50%, transparent 50%);
  background-position: calc(100% - 14px) calc(50% - 2px),
                       calc(100% - 9px) calc(50% - 2px);
  background-size: 5px 5px, 5px 5px;
  background-repeat: no-repeat;
  margin-right: 6px;
}

#user-input {
  flex: 1;
  padding: 8px 10px;
  background: transparent;
  border: none;
  border-radius: 0;
  color: var(--text-primary);
  font-size: 13px;
  outline: none;
  resize: none;
  min-height: 32px;
  max-height: 140px;
  font-family: inherit;
  line-height: 1.5;
}
#user-input::placeholder { color: var(--text-muted); }

#send-btn {
  padding: 0 18px;
  background: var(--mode-color);
  border: none;
  border-radius: var(--radius-md);
  color: #0a0e15;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.4px;
  cursor: pointer;
  transition: filter 0.15s, transform 0.1s;
  white-space: nowrap;
  margin-left: 6px;
}
#send-btn:hover:not(:disabled) { filter: brightness(1.1); }
#send-btn:active:not(:disabled) { transform: scale(0.97); }
#send-btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
  background: var(--bg-elevated);
  color: var(--text-muted);
}

#input-hints {
  margin-top: 8px;
  padding: 0 4px;
  font-size: 10px;
  color: var(--text-muted);
  display: flex;
  gap: 14px;
  justify-content: flex-end;
}
.hint-key {
  background: var(--bg-card);
  border: 1px solid var(--border);
  padding: 1px 5px;
  border-radius: 3px;
  font-family: "SF Mono", Menlo, monospace;
  font-size: 9px;
  color: var(--text-secondary);
}

#mode-context {
  margin-bottom: 10px;
}
.mode-note {
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--bg-tertiary);
  color: var(--text-secondary);
  font-size: 12px;
  line-height: 1.5;
}
.mode-fields {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 8px;
}
.field-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.field-group label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  color: var(--text-muted);
  font-weight: 700;
}
.field-group input,
.field-group select,
.field-group textarea {
  width: 100%;
  padding: 8px 10px;
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
  background: var(--bg-tertiary);
  color: var(--text-primary);
  font-size: 12px;
  outline: none;
}
.field-group textarea {
  min-height: 180px;
  resize: vertical;
  line-height: 1.5;
  font-family: "SF Mono", Menlo, monospace;
}
.field-group input:focus,
.field-group select:focus,
.field-group textarea:focus {
  border-color: var(--mode-color);
}
.field-help {
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.45;
}
.prompt-settings {
  display: grid;
  grid-template-columns: minmax(180px, 240px) 1fr;
  gap: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
}
.prompt-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.prompt-list select {
  min-height: 220px;
}
.prompt-editor {
  display: grid;
  gap: 8px;
}
.prompt-meta-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: center;
}
.prompt-badge {
  display: inline-flex;
  align-items: center;
  min-height: 22px;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--bg-tertiary);
  color: var(--text-secondary);
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.6px;
}
.prompt-badge.active {
  border-color: var(--orange);
  color: var(--orange);
}
.prompt-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
@media (max-width: 760px) {
  .prompt-settings {
    grid-template-columns: 1fr;
  }
  .prompt-list select {
    min-height: 120px;
  }
}

/* ── Graph panel ── */
#graph-panel {
  flex: 1;
  display: none;
  position: relative;
  overflow: hidden;
}
#graph-panel.active { display: block; }
#graph-panel svg { width: 100%; height: 100%; }
.graph-link { stroke-opacity: 0.45; }
.graph-link-label { font-size: 8px; fill: var(--text-muted); pointer-events: none; }
.graph-node { cursor: pointer; transition: stroke-width 0.2s; }
.graph-node:hover { stroke-width: 3px; }
.graph-label {
  font-size: 10px; fill: var(--text-secondary);
  pointer-events: none; text-anchor: middle;
}
#graph-tooltip {
  position: absolute;
  padding: 6px 10px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 11px;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.15s;
  z-index: 50;
  color: var(--text-primary);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Welcome screen ── */
.welcome {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  text-align: center;
  padding: 40px;
  color: var(--text-secondary);
}
.welcome-logo {
  font-size: 56px;
  margin-bottom: 20px;
  background: linear-gradient(135deg, var(--accent), var(--purple));
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  opacity: 0.85;
  filter: drop-shadow(0 4px 16px rgba(110, 168, 255, 0.2));
}
.welcome h2 {
  font-size: 22px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 10px;
  letter-spacing: -0.2px;
}
.welcome p {
  font-size: 13px;
  line-height: 1.7;
  max-width: 440px;
  color: var(--text-secondary);
}
.welcome-actions {
  display: flex;
  gap: 8px;
  margin-top: 28px;
  flex-wrap: wrap;
  justify-content: center;
}
.welcome-action {
  padding: 8px 16px;
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  background: var(--bg-tertiary);
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.18s ease;
}
.welcome-action:hover {
  border-color: var(--accent);
  color: var(--accent);
  background: var(--accent-dim);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}

/* ── Agent mode indicator ── */
#mode-select option[value="agent"] { color: var(--purple); font-weight: 600; }

/* ── Empty sidebar ── */
.sidebar-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  color: var(--text-muted);
  font-size: 12px;
  text-align: center;
  padding: 20px;
}

.modal-shell {
  position: fixed;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.6);
  z-index: 1000;
}
.modal-card {
  width: min(720px, 92vw);
  max-height: 78vh;
  overflow: hidden;
  border-radius: 14px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  box-shadow: var(--shadow-lg);
}
.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 18px;
  border-bottom: 1px solid var(--border);
}
.modal-header h3 {
  font-size: 15px;
  color: var(--text-primary);
}
.modal-close {
  border: 1px solid var(--border);
  background: var(--bg-tertiary);
  color: var(--text-secondary);
  border-radius: 8px;
  padding: 6px 10px;
  cursor: pointer;
}
.modal-body {
  padding: 14px 18px 18px;
  overflow-y: auto;
  max-height: calc(78vh - 60px);
}
.session-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 10px;
  padding: 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--bg-tertiary);
  margin-bottom: 8px;
}
.session-row strong {
  display: block;
  color: var(--text-primary);
  margin-bottom: 4px;
}
.session-row span {
  display: block;
  color: var(--text-secondary);
  font-size: 12px;
}
.session-actions {
  display: flex;
  gap: 8px;
  align-items: center;
}
.session-actions button {
  padding: 7px 10px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--bg-card);
  color: var(--text-secondary);
  cursor: pointer;
}
.session-actions button:hover,
.modal-close:hover {
  border-color: var(--accent);
  color: var(--accent);
}
.concept-card-grid {
  display: grid;
  gap: 14px;
}
.concept-card-hero {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  padding: 14px;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--bg-soft);
}
.concept-card-title {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.concept-card-title h4 {
  margin: 0;
  font-size: 20px;
}
.concept-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.concept-chip {
  padding: 5px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--bg-card);
  font-size: 12px;
  color: var(--text-secondary);
}
.concept-card-metrics {
  display: grid;
  grid-template-columns: repeat(2, minmax(92px, 1fr));
  gap: 10px;
  min-width: 220px;
}
.concept-metric {
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: var(--bg-card);
}
.concept-metric strong {
  display: block;
  font-size: 18px;
}
.concept-metric span {
  color: var(--text-muted);
  font-size: 11px;
}
.concept-card-section {
  padding: 14px;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--bg-card);
}
.concept-card-section h4 {
  margin: 0 0 10px 0;
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
}
.concept-card-list {
  display: grid;
  gap: 8px;
}
.concept-card-list .empty {
  color: var(--text-muted);
  font-size: 13px;
}
.concept-relation-row {
  display: flex;
  gap: 8px;
  align-items: baseline;
  padding: 8px 10px;
  border-radius: 10px;
  background: var(--bg-soft);
}
.concept-relation-row .weight {
  margin-left: auto;
  color: var(--text-muted);
  font-size: 11px;
}
.concept-card-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.concept-card-actions button {
  padding: 7px 12px;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--bg-card);
  color: var(--text-primary);
  cursor: pointer;
}
.concept-card-actions button:hover {
  border-color: var(--accent);
  color: var(--accent);
}
@media (max-width: 900px) {
  .concept-card-hero {
    flex-direction: column;
  }
  .concept-card-metrics {
    width: 100%;
    min-width: 0;
  }
}
</style>
</head>
<body>

<div id="titlebar">
  <div class="traffic-spacer"></div>
  <span class="title" id="app-title">World 0 — Concept World</span>
</div>

<div id="app">
  <!-- Sidebar -->
  <div id="sidebar">
    <div class="sidebar-header">
      <h2 id="sidebar-title">Concept World</h2>
      <span class="sidebar-badge" id="concept-count">0</span>
    </div>
    <input type="text" id="sidebar-search" placeholder="Search concepts..." />
    <div id="environment-card">
      <div class="environment-title" id="environment-title">Environment</div>
      <div class="environment-grid">
        <div class="environment-item">
          <strong id="env-label-agentic">Agentic</strong>
          <span id="env-agentic">Checking…</span>
        </div>
        <div class="environment-item">
          <strong id="env-label-skills">Skills</strong>
          <span id="env-skills">0</span>
        </div>
        <div class="environment-item">
          <strong id="env-label-mcp">MCP</strong>
          <span id="env-mcp">0 connected</span>
        </div>
        <div class="environment-item">
          <strong id="env-label-sessions">Sessions</strong>
          <span id="env-sessions">0 recent</span>
        </div>
      </div>
      <div class="environment-path" id="env-store">Store: —</div>
      <div class="environment-hint" id="env-hint" style="display:none"></div>
    </div>
    <div id="concept-list">
      <div class="sidebar-empty" id="sidebar-empty">
        <div style="font-size: 24px; opacity: 0.3; margin-bottom: 8px;">&#9673;</div>
        <div>No concepts in this world yet</div>
        <div style="font-size: 11px; margin-top: 4px;">Start with an observation</div>
      </div>
    </div>
    <div id="stats-bar">
      <div class="stat-box">
        <div class="stat-val" id="stat-concepts">0</div>
        <div class="stat-lbl" id="stat-label-concepts">Concepts</div>
      </div>
      <div class="stat-box">
        <div class="stat-val" id="stat-relations">0</div>
        <div class="stat-lbl" id="stat-label-relations">Relations</div>
      </div>
      <div class="stat-box">
        <div class="stat-val" id="stat-confidence">0</div>
        <div class="stat-lbl" id="stat-label-confidence">Avg Conf</div>
      </div>
    </div>
    <div id="sidebar-actions">
      <button class="sidebar-btn" id="btn-consolidate" onclick="doReflect()">Consolidate</button>
      <button class="sidebar-btn" id="btn-projection" onclick="switchTab('graph')">Projection</button>
      <button class="sidebar-btn" id="btn-sessions" onclick="showSessionsModal()">Sessions</button>
      <button class="sidebar-btn" id="btn-save" onclick="saveSession()">Save</button>
      <button class="sidebar-btn" id="btn-new" onclick="newSession()">New</button>
      <button class="sidebar-btn" id="btn-settings" onclick="showSettingsModal()">Settings</button>
    </div>
  </div>

  <!-- Main -->
  <div id="main">
    <div id="tab-bar">
      <div class="tab active" id="tab-chat" data-tab="chat" onclick="switchTab('chat')">Chat</div>
      <div class="tab" id="tab-graph" data-tab="graph" onclick="switchTab('graph')">Projection Graph</div>
    </div>

    <!-- Chat -->
    <div id="chat-panel">
      <div id="messages">
        <div class="welcome" id="welcome-screen">
          <div class="welcome-logo">&#9673;</div>
          <h2 id="welcome-title">World 0</h2>
          <p id="welcome-body">Build a concept-world, not a note vault.<br>
          Submit observations, activate relevant concepts, and generate local cognitive projections for the task at hand.</p>
          <div class="welcome-actions">
            <div class="welcome-action" id="welcome-agent" style="border-color:var(--purple);color:var(--purple)" onclick="setMode('agent'); focusInput()">Agent Mode</div>
            <div class="welcome-action" id="welcome-research" style="border-color:var(--accent);color:var(--accent)" onclick="setMode('research'); focusInput()">Research</div>
            <div class="welcome-action" id="welcome-skill" style="border-color:var(--green);color:var(--green)" onclick="showSkillPicker()">Run Skill</div>
            <div class="welcome-action" id="welcome-observe" onclick="setMode('learn'); focusInput()">Observe</div>
            <div class="welcome-action" id="welcome-project" onclick="setMode('ask'); focusInput()">Project</div>
            <div class="welcome-action" id="welcome-inspect" onclick="setMode('explore'); focusInput()">Inspect</div>
            <div class="welcome-action" id="welcome-relate" onclick="setMode('connect'); focusInput()">Relate</div>
          </div>
        </div>
      </div>
      <div id="input-area">
        <div id="mode-context"></div>
        <div id="input-row">
          <select id="mode-select" onchange="handleModeChange()">
            <option value="agent" id="mode-option-agent">Agent</option>
            <option value="research" id="mode-option-research">Research</option>
            <option value="skill" id="mode-option-skill">Skill</option>
            <option value="ask" id="mode-option-ask">Project</option>
            <option value="learn" id="mode-option-learn">Observe</option>
            <option value="explore" id="mode-option-explore">Inspect</option>
            <option value="connect" id="mode-option-connect">Relate</option>
          </select>
          <textarea id="user-input" rows="1" placeholder="Ask a question about your knowledge world..."
                    onkeydown="handleKeydown(event)"></textarea>
          <button id="send-btn" onclick="sendMessage()">Send</button>
        </div>
        <div id="input-hints">
          <span id="hint-send"><span class="hint-key">Enter</span> send</span>
          <span id="hint-newline"><span class="hint-key">Shift+Enter</span> newline</span>
          <span id="hint-switch"><span class="hint-key">Tab</span> switch mode</span>
        </div>
      </div>
    </div>

    <!-- Graph (hidden by default) -->
    <div id="graph-panel">
      <svg id="graph-svg"></svg>
      <div id="graph-tooltip"></div>
    </div>
  </div>
</div>

<script>
// ── State ──
let isProcessing = false;
let graphInitialized = false;
let relationTypes = [];
let availableSkills = [];
let environmentStatus = null;
window._selectedSkill = null;
let currentLanguage = "en";
let isComposingInput = false;

const I18N = {
  en: {
    appTitle: "World 0 — Concept World",
    sidebarTitle: "Concept World",
    searchPlaceholder: "Search concepts...",
    environmentTitle: "Environment",
    agentic: "Agentic",
    skills: "Skills",
    sessions: "Sessions",
    concepts: "Concepts",
    relations: "Relations",
    avgConfidence: "Avg Conf",
    consolidate: "Consolidate",
    projection: "Projection",
    save: "Save",
    newSession: "New",
    settings: "Settings",
    chat: "Chat",
    projectionGraph: "Projection Graph",
    welcomeTitle: "World 0",
    welcomeBody: "Build a concept-world, not a note vault.<br>Submit observations, activate relevant concepts, and generate local cognitive projections for the task at hand.",
    agentMode: "Agent Mode",
    researchMode: "Research",
    runSkill: "Run Skill",
    observe: "Observe",
    project: "Project",
    inspect: "Inspect",
    relate: "Relate",
    send: "send",
    newline: "newline",
    switchMode: "switch mode",
    modeAgent: "Agent",
    modeResearch: "Research",
    modeSkill: "Skill",
    modeAsk: "Project",
    modeLearn: "Observe",
    modeExplore: "Inspect",
    modeConnect: "Relate",
    placeholderAgent: "Describe the task — the agent will decide which operations to run...",
    placeholderResearch: "Describe the topic you want researched...",
    placeholderSkill: "Optional free-form context for the selected skill...",
    placeholderAsk: "Describe the task you want a local projection for...",
    placeholderLearn: "Paste an observation, note, or source text to ingest...",
    placeholderExplore: "Optional: add extra context about what role to inspect...",
    placeholderConnect: "Optional note about why this relation matters...",
    noConceptsWorld: "No concepts in this world yet",
    startObservation: "Start with an observation",
    environmentStore: "Store",
    checking: "Checking…",
    connected: "connected",
    recent: "recent",
    llmOff: "LLM off",
    llmConfigured: "LLM configured",
    noConceptGraph: "No concept-world yet — submit an observation first",
    settingsTitle: "Settings",
    language: "Language",
    provider: "Provider",
    model: "Model",
    modelPreset: "Available Models",
    customModel: "Custom Model",
    apiKey: "API Key",
    baseUrl: "Base URL",
    azureEndpoint: "Azure Endpoint",
    apiVersion: "API Version",
    autoSedimentDialogue: "Auto-sediment dialogue into World 0",
    autoSedimentDialogueHelp: "When enabled, chat turns are periodically extracted into concepts and relations.",
    dialogueSedimentInterval: "Dialogue sediment interval",
    dialogueSedimentIntervalHelp: "Extract once every 1-20 turns.",
    promptRegistry: "Prompt Registry",
    promptSelect: "Prompt",
    promptTemplate: "Override Template",
    promptDefault: "Default",
    promptEffective: "Effective",
    promptOverridden: "Overridden",
    promptActiveOverride: "Active Space Override",
    savePrompt: "Save Override",
    resetPrompt: "Reset",
    exportPrompts: "Export JSON",
    promptSaved: "Prompt override saved: {id}",
    promptReset: "Prompt override reset: {id}",
    promptExported: "Prompt registry exported.",
    promptActionFailed: "Prompt update failed: {message}",
    noPromptsAvailable: "No configurable prompts.",
    saveSettings: "Save Settings",
    close: "Close",
    sessionsTitle: "Recent Sessions",
    resume: "Resume",
    noSavedSessions: "No saved sessions yet. Sessions are created automatically as you interact.",
    selectSkill: "Select a Skill",
    use: "Use",
    noParams: "No parameters",
    skillSelected: "Skill selected: **{name}**. Fill the fields below and press Send.",
    sessionSaved: "Session saved: {id}",
    sessionResumed: "Resumed session: {title} ({id})",
    newSessionReady: "Session {id} is ready. Submit an observation, inspect a concept, or request a projection.",
    reflectFailed: "Consolidation failed: {message}",
    loadSkillsFailed: "Failed to load skills: {message}",
    loadSessionsFailed: "Failed to load sessions: {message}",
    resumeSessionFailed: "Failed to resume session: {message}",
    saveSessionFailed: "Failed to save session",
    createSessionFailed: "Failed to create session",
    requestFailed: "Request failed: {message}",
    chooseSkillFirst: "Select a skill first.",
    noSkillsAvailable: "No skills available.",
    settingsUpdated: "Settings updated.",
    updateSettingsFailed: "Failed to update settings: {message}",
    apiKeyHint: "Leave empty to use the system environment variable.",
    envDetected: "Detected in environment",
    envNotDetected: "Not found in environment",
    autoSelectModel: "Use selected model",
    responseLangZh: "Respond in Simplified Chinese",
    responseLangEn: "Respond in English",
    saveFindings: "Learn findings into World 0",
    researchFocus: "Research Focus",
    researchSources: "Sources",
    conceptCard: "Concept Card",
    viewCard: "View Card",
    inspectConcept: "Inspect Concept",
    projectFromConcept: "Project From Concept",
    descriptionLabel: "Description",
    aliasesLabel: "Aliases",
    tagsLabel: "Tags",
    sourcesLabel: "Sources",
    tasksLabel: "Tasks",
    relationsLabel: "Relations",
    recentActivityLabel: "Recent Activity",
    createdAtLabel: "Created",
    lastActivatedLabel: "Last Activated",
    activationCountLabel: "Activations",
    confidenceLabel: "Confidence",
    maturityLabel: "Maturity",
    relationCountLabel: "Relations",
    noDescription: "No description yet.",
    noActivity: "No recent activity yet.",
    noRelations: "No relations yet.",
    conceptCardLoadFailed: "Failed to load concept card: {message}",
  },
  zh: {
    appTitle: "World 0 — 概念世界",
    sidebarTitle: "概念世界",
    searchPlaceholder: "搜索概念...",
    environmentTitle: "环境",
    agentic: "智能模式",
    skills: "技能",
    sessions: "会话",
    concepts: "概念",
    relations: "关系",
    avgConfidence: "平均置信度",
    consolidate: "巩固",
    projection: "投影",
    save: "保存",
    newSession: "新建",
    settings: "设置",
    chat: "对话",
    projectionGraph: "投影视图图谱",
    welcomeTitle: "World 0",
    welcomeBody: "构建概念世界，而不是笔记仓库。<br>提交观察，激活相关概念，并为当前任务生成局部认知投影。",
    agentMode: "智能模式",
    researchMode: "研究",
    runSkill: "运行技能",
    observe: "观察",
    project: "投影",
    inspect: "检查",
    relate: "建关系",
    send: "发送",
    newline: "换行",
    switchMode: "切换模式",
    modeAgent: "智能",
    modeResearch: "研究",
    modeSkill: "技能",
    modeAsk: "投影",
    modeLearn: "观察",
    modeExplore: "检查",
    modeConnect: "建关系",
    placeholderAgent: "描述任务，Agent 会自行决定要执行哪些操作...",
    placeholderResearch: "描述你要研究的主题...",
    placeholderSkill: "可选：补充技能的自由文本上下文...",
    placeholderAsk: "描述你想为其生成局部投影的任务...",
    placeholderLearn: "粘贴要摄取的观察、笔记或源文本...",
    placeholderExplore: "可选：补充你想检查的概念角色...",
    placeholderConnect: "可选：补充这条关系为什么重要...",
    noConceptsWorld: "这个世界里还没有概念",
    startObservation: "先提交一条观察",
    environmentStore: "存储路径",
    checking: "检查中…",
    connected: "已连接",
    recent: "最近",
    llmOff: "LLM 已关闭",
    llmConfigured: "LLM 已配置",
    noConceptGraph: "概念世界还是空的，先提交一条观察",
    settingsTitle: "设置",
    language: "语言",
    provider: "提供方",
    model: "模型",
    modelPreset: "可用模型",
    customModel: "自定义模型",
    apiKey: "API Key",
    baseUrl: "Base URL",
    azureEndpoint: "Azure Endpoint",
    apiVersion: "API 版本",
    autoSedimentDialogue: "自动将对话沉淀进 World 0",
    autoSedimentDialogueHelp: "开启后，系统会按间隔把对话轮次抽取为概念与关系。",
    dialogueSedimentInterval: "对话沉淀间隔",
    dialogueSedimentIntervalHelp: "每 1-20 轮对话执行一次抽取。",
    promptRegistry: "Prompt Registry",
    promptSelect: "Prompt",
    promptTemplate: "覆盖模板",
    promptDefault: "默认",
    promptEffective: "当前生效",
    promptOverridden: "已覆盖",
    promptActiveOverride: "当前空间覆盖",
    savePrompt: "保存覆盖",
    resetPrompt: "重置",
    exportPrompts: "导出 JSON",
    promptSaved: "Prompt 覆盖已保存：{id}",
    promptReset: "Prompt 覆盖已重置：{id}",
    promptExported: "Prompt registry 已导出。",
    promptActionFailed: "Prompt 更新失败：{message}",
    noPromptsAvailable: "当前没有可配置 prompt。",
    saveSettings: "保存设置",
    close: "关闭",
    sessionsTitle: "最近会话",
    resume: "恢复",
    noSavedSessions: "还没有保存的会话。你在界面中的交互会自动形成会话。",
    selectSkill: "选择技能",
    use: "使用",
    noParams: "无参数",
    skillSelected: "已选择技能：**{name}**。请填写下方字段后发送。",
    sessionSaved: "会话已保存：{id}",
    sessionResumed: "已恢复会话：{title}（{id}）",
    newSessionReady: "会话 {id} 已就绪。可以提交观察、检查概念或请求投影。",
    reflectFailed: "巩固失败：{message}",
    loadSkillsFailed: "加载技能失败：{message}",
    loadSessionsFailed: "加载会话失败：{message}",
    resumeSessionFailed: "恢复会话失败：{message}",
    saveSessionFailed: "保存会话失败",
    createSessionFailed: "创建会话失败",
    requestFailed: "请求失败：{message}",
    chooseSkillFirst: "请先选择一个技能。",
    noSkillsAvailable: "当前没有可用技能。",
    settingsUpdated: "设置已更新。",
    updateSettingsFailed: "更新设置失败：{message}",
    apiKeyHint: "留空时默认使用系统环境变量。",
    envDetected: "系统环境变量中已检测到",
    envNotDetected: "系统环境变量中未检测到",
    autoSelectModel: "使用所选模型",
    responseLangZh: "使用简体中文回答",
    responseLangEn: "使用英文回答",
    saveFindings: "将研究结果学习进 World 0",
    researchFocus: "研究焦点",
    researchSources: "来源数量",
    conceptCard: "概念卡片",
    viewCard: "查看卡片",
    inspectConcept: "检查概念",
    projectFromConcept: "从该概念投影",
    descriptionLabel: "描述",
    aliasesLabel: "别名",
    tagsLabel: "标签",
    sourcesLabel: "来源",
    tasksLabel: "任务",
    relationsLabel: "关系",
    recentActivityLabel: "最近活动",
    createdAtLabel: "创建时间",
    lastActivatedLabel: "最近激活",
    activationCountLabel: "激活次数",
    confidenceLabel: "置信度",
    maturityLabel: "成熟度",
    relationCountLabel: "关系数",
    noDescription: "还没有描述。",
    noActivity: "还没有最近活动。",
    noRelations: "还没有关系。",
    conceptCardLoadFailed: "加载概念卡片失败：{message}",
  },
};

const MATURITY_COLORS = {
  embryonic:   "#d29922",
  developing:  "#58a6ff",
  established: "#3fb950",
  core:        "#bc8cff",
  fading:      "#484f58",
};

const RELATION_COLORS = {
  positive: "#3fb950",
  negative: "#f85149",
  parallel: "#39d2c0",
};

// ── Helpers ──
function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function focusInput() { $("#user-input").focus(); }

function t(key, vars = {}) {
  const table = I18N[currentLanguage] || I18N.en;
  let text = table[key] || I18N.en[key] || key;
  for (const [name, value] of Object.entries(vars)) {
    text = text.replaceAll(`{${name}}`, value);
  }
  return text;
}

function applyTranslations() {
  document.documentElement.lang = currentLanguage === "zh" ? "zh" : "en";
  document.title = t("appTitle");
  $("#app-title").textContent = t("appTitle");
  $("#sidebar-title").textContent = t("sidebarTitle");
  $("#sidebar-search").placeholder = t("searchPlaceholder");
  $("#environment-title").textContent = t("environmentTitle");
  $("#env-label-agentic").textContent = t("agentic");
  $("#env-label-skills").textContent = t("skills");
  $("#env-label-mcp").textContent = "MCP";
  $("#env-label-sessions").textContent = t("sessions");
  $("#stat-label-concepts").textContent = t("concepts");
  $("#stat-label-relations").textContent = t("relations");
  $("#stat-label-confidence").textContent = t("avgConfidence");
  $("#btn-consolidate").textContent = t("consolidate");
  $("#btn-projection").textContent = t("projection");
  $("#btn-sessions").textContent = t("sessions");
  $("#btn-save").textContent = t("save");
  $("#btn-new").textContent = t("newSession");
  $("#btn-settings").textContent = t("settings");
  $("#tab-chat").textContent = t("chat");
  $("#tab-graph").textContent = t("projectionGraph");
  $("#welcome-title").textContent = t("welcomeTitle");
  $("#welcome-body").innerHTML = t("welcomeBody");
  $("#welcome-agent").textContent = t("agentMode");
  $("#welcome-research").textContent = t("researchMode");
  $("#welcome-skill").textContent = t("runSkill");
  $("#welcome-observe").textContent = t("observe");
  $("#welcome-project").textContent = t("project");
  $("#welcome-inspect").textContent = t("inspect");
  $("#welcome-relate").textContent = t("relate");
  $("#mode-option-agent").textContent = t("modeAgent");
  $("#mode-option-research").textContent = t("modeResearch");
  $("#mode-option-skill").textContent = t("modeSkill");
  $("#mode-option-ask").textContent = t("modeAsk");
  $("#mode-option-learn").textContent = t("modeLearn");
  $("#mode-option-explore").textContent = t("modeExplore");
  $("#mode-option-connect").textContent = t("modeConnect");
  $("#send-btn").textContent = currentLanguage === "zh" ? "发送" : "Send";
  $("#hint-send").innerHTML = `<span class="hint-key">Enter</span> ${t("send")}`;
  $("#hint-newline").innerHTML = `<span class="hint-key">Shift+Enter</span> ${t("newline")}`;
  $("#hint-switch").innerHTML = `<span class="hint-key">Tab</span> ${t("switchMode")}`;
  updatePlaceholder();
  renderModeContext();
}

function setMode(mode) {
  $("#mode-select").value = mode;
  handleModeChange();
}

function handleModeChange() {
  updatePlaceholder();
  renderModeContext();
}

function updatePlaceholder() {
  const mode = $("#mode-select").value;
  const input = $("#user-input");
  const placeholders = {
    agent: t("placeholderAgent"),
    research: t("placeholderResearch"),
    skill: t("placeholderSkill"),
    ask: t("placeholderAsk"),
    learn: t("placeholderLearn"),
    explore: t("placeholderExplore"),
    connect: t("placeholderConnect"),
  };
  input.placeholder = placeholders[mode] || "";
}

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatDateTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return escapeHtml(value);
  return date.toLocaleString(currentLanguage === "zh" ? "zh-CN" : "en-US");
}

function selectedSkill() {
  return availableSkills.find(s => s.name === window._selectedSkill) || null;
}

function fieldValue(id) {
  const el = document.getElementById(id);
  return el ? el.value.trim() : "";
}

function rawFieldValue(id) {
  const el = document.getElementById(id);
  return el ? el.value : "";
}

function renderModeContext() {
  const mode = $("#mode-select").value;
  const host = $("#mode-context");
  const skill = selectedSkill();
  const relationOptions = (relationTypes.length ? relationTypes : ["parallel"])
    .map(type => `<option value="${escapeHtml(type)}">${escapeHtml(type)}</option>`)
    .join("");

  if (mode === "learn") {
    host.innerHTML = `
      <div class="mode-fields">
        <div class="field-group">
          <label for="learn-task">${currentLanguage === "zh" ? "任务上下文" : "Task Context"}</label>
          <input id="learn-task" placeholder="${currentLanguage === "zh" ? "例如：认证迁移" : "e.g. auth migration"}" />
        </div>
        <div class="field-group">
          <label for="learn-source">${currentLanguage === "zh" ? "来源标签" : "Source Label"}</label>
          <input id="learn-source" placeholder="${currentLanguage === "zh" ? "例如：session_001 或 article_url" : "e.g. session_001 or article_url"}" />
        </div>
      </div>
    `;
    return;
  }

  if (mode === "research") {
    host.innerHTML = `
      <div class="mode-fields">
        <div class="field-group">
          <label for="research-focus">${t("researchFocus")}</label>
          <input id="research-focus" placeholder="${currentLanguage === "zh" ? "例如：架构、趋势、风险、竞品" : "e.g. architecture, trends, risks, competitors"}" />
        </div>
        <div class="field-group">
          <label for="research-sources">${t("researchSources")}</label>
          <input id="research-sources" type="number" min="1" max="8" value="4" />
        </div>
        <div class="field-group" style="justify-content:flex-end">
          <label for="research-save" style="display:flex;align-items:center;gap:8px">
            <input id="research-save" type="checkbox" checked />
            <span>${t("saveFindings")}</span>
          </label>
        </div>
      </div>
      <div class="field-help">${currentLanguage === "zh"
        ? "研究模式会搜索公开网页、阅读来源、提炼结论，并可把结果写入概念世界。"
        : "Research mode searches the public web, reads sources, distills findings, and can write them into the concept-world."}</div>
    `;
    return;
  }

  if (mode === "connect") {
    host.innerHTML = `
      <div class="mode-fields">
        <div class="field-group">
          <label for="connect-source">${currentLanguage === "zh" ? "源概念" : "Source Concept"}</label>
          <input id="connect-source" placeholder="e.g. FastAPI" />
        </div>
        <div class="field-group">
          <label for="connect-type">${currentLanguage === "zh" ? "关系类型" : "Relation Type"}</label>
          <select id="connect-type">${relationOptions}</select>
        </div>
        <div class="field-group">
          <label for="connect-target">${currentLanguage === "zh" ? "目标概念" : "Target Concept"}</label>
          <input id="connect-target" placeholder="e.g. Python" />
        </div>
      </div>
      <div class="field-help">${currentLanguage === "zh" ? "使用明确的关系类型来塑造局部概念世界。" : "Use typed relations to shape the local concept-world explicitly."}</div>
    `;
    return;
  }

  if (mode === "explore") {
    host.innerHTML = `
      <div class="mode-fields">
        <div class="field-group">
          <label for="explore-concept">${currentLanguage === "zh" ? "概念" : "Concept"}</label>
          <input id="explore-concept" placeholder="e.g. PostgreSQL" />
        </div>
      </div>
      <div class="field-help">${currentLanguage === "zh" ? "检查一个概念在世界中的轴向关系：吸引、排斥或共振。" : "Inspect a concept's axis links: attraction, repulsion, or resonance."}</div>
    `;
    return;
  }

  if (mode === "skill") {
    if (!skill) {
      host.innerHTML = `
      <div class="mode-note">
        ${currentLanguage === "zh"
          ? `尚未选择技能。<a href="#" onclick="showSkillPicker(); return false;" style="color:var(--green)">选择一个技能</a> 以显示其参数字段。`
          : `No skill selected. <a href="#" onclick="showSkillPicker(); return false;" style="color:var(--green)">Choose a skill</a> to render its parameter fields here.`}
      </div>
    `;
    return;
  }

    const paramFields = skill.parameters.map(param => `
      <div class="field-group">
        <label for="skill-param-${escapeHtml(param.name)}">${escapeHtml(param.name)}${param.required ? " *" : ""}</label>
        <input id="skill-param-${escapeHtml(param.name)}"
               placeholder="${escapeHtml(param.description || param.default || param.name)}"
               value="${escapeHtml(param.default || "")}" />
      </div>
    `).join("");

    host.innerHTML = `
      <div class="mode-note">
        ${currentLanguage === "zh" ? "已选择技能" : "Selected skill"}:
        <strong style="color:var(--green)">${escapeHtml(skill.name)}</strong><br>
        ${escapeHtml(skill.description)}
      </div>
      ${paramFields ? `<div class="mode-fields">${paramFields}</div>` : ""}
    `;
    return;
  }

  if (mode === "ask") {
    host.innerHTML = `
      <div class="mode-note">
        ${currentLanguage === "zh"
          ? "为你下方描述的任务生成一个紧凑的局部概念投影。"
          : "Project a compact local view of the concept-world for the task you describe below."}
      </div>
    `;
    return;
  }

  if (mode === "agent") {
    host.innerHTML = `
      <div class="mode-note">
        ${currentLanguage === "zh"
          ? "智能模式可以自主串联观察、研究、关系塑形、技能和 MCP 工具。"
          : "Agent mode can chain observation, research, relation-building, skills, and MCP tools autonomously."}
      </div>
    `;
    return;
  }

  host.innerHTML = "";
}

function handleKeydown(e) {
  if (e.isComposing || isComposingInput || e.keyCode === 229) {
    return;
  }

  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
  if (e.key === "Tab") {
    e.preventDefault();
    const sel = $("#mode-select");
    const opts = [...sel.options];
    const idx = opts.findIndex(o => o.value === sel.value);
    sel.value = opts[(idx + 1) % opts.length].value;
    handleModeChange();
  }
}

// ── Auto-resize textarea ──
const userInputEl = $("#user-input");
userInputEl.addEventListener("compositionstart", () => {
  isComposingInput = true;
});
userInputEl.addEventListener("compositionend", () => {
  isComposingInput = false;
});
userInputEl.addEventListener("input", function() {
  this.style.height = "auto";
  this.style.height = Math.min(this.scrollHeight, 120) + "px";
});

// ── Markdown rendering (lightweight) ──
function renderMarkdown(text) {
  if (!text) return "";
  let html = text
    // Code blocks
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Headers
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // Bold
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    // Italic
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    // HR
    .replace(/^---+$/gm, '<hr>')
    // Lists
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    // Links
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" style="color:var(--accent)">$1</a>')
    // Line breaks (but not inside pre blocks)
    .replace(/\n/g, '<br>');

  // Wrap consecutive <li> in <ul>
  html = html.replace(/((?:<li>.*?<\/li><br>?)+)/g, '<ul>$1</ul>');
  html = html.replace(/<ul>(.*?)<\/ul>/gs, (m, inner) =>
    '<ul>' + inner.replace(/<br>/g, '') + '</ul>'
  );

  return html;
}

// ── Messages ──
function formatMessageTime(value) {
  if (!value) {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function addMessage(role, content, type = "text", options = {}) {
  // Hide welcome screen
  const welcome = $("#welcome-screen");
  if (welcome) welcome.style.display = "none";

  const container = $("#messages");
  const div = document.createElement("div");
  div.className = `message msg-${role}`;

  const avatarMap = { user: "U", agent: "W", system: "S", error: "!" };
  const nameMap = { user: "You", agent: "World 0", system: "System", error: "Error" };

  const timeLabel = formatMessageTime(options.timestamp);

  const bodyHTML = type === "markdown" || role === "agent"
    ? renderMarkdown(content)
    : content.replace(/</g, "&lt;").replace(/\n/g, "<br>");

  div.innerHTML = `
    <div class="message-avatar">${avatarMap[role] || "?"}</div>
    <div class="message-content">
      <div class="message-header">
        <span class="name">${nameMap[role] || role}</span>
        <span>${timeLabel}</span>
      </div>
      <div class="message-body">${bodyHTML}</div>
    </div>
  `;

  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

function addTypingIndicator() {
  const container = $("#messages");
  const div = document.createElement("div");
  div.className = "message msg-agent";
  div.id = "typing";
  div.innerHTML = `
    <div class="message-avatar">W</div>
    <div class="message-content">
      <div class="typing-indicator"><span></span><span></span><span></span></div>
    </div>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function removeTypingIndicator() {
  const el = $("#typing");
  if (el) el.remove();
}

// ── API calls ──
async function apiCall(url, options = {}) {
  const resp = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.error || data.message || `HTTP ${resp.status}`);
  }
  return data;
}

function parseConnect(text) {
  const sourceField = fieldValue("connect-source");
  const targetField = fieldValue("connect-target");
  const relationField = fieldValue("connect-type");
  if (sourceField && targetField) {
    return {
      source: sourceField,
      target: targetField,
      relation_type: relationField || "generic_relation",
    };
  }

  const arrowMatch = text.match(/^(.+?)\s*(?:→|->)+\s*(.+?)(?:\s+([\w_]+))?\s*$/);
  if (arrowMatch) {
    return {
      source: arrowMatch[1].trim(),
      target: arrowMatch[2].trim(),
      relation_type: arrowMatch[3] || "generic_relation",
    };
  }
  const parts = text.split(/\s+/);
  if (parts.length >= 2) {
    return {
      source: parts[0],
      target: parts[1],
      relation_type: parts[2] || "generic_relation",
    };
  }
  return {
    error: currentLanguage === "zh"
      ? '请提供源概念和目标概念，或使用“source → target [type]”格式。'
      : 'Provide source and target concepts, or use "source → target [type]".',
  };
}

function buildSkillParams(fallbackText) {
  const skill = selectedSkill();
  if (!skill) return { error: t("chooseSkillFirst") };

  const params = {};
  for (const param of skill.parameters) {
    const value = fieldValue(`skill-param-${param.name}`);
    if (value) params[param.name] = value;
  }

  if (fallbackText && skill.parameters.length === 1 && !Object.keys(params).length) {
    params[skill.parameters[0].name] = fallbackText;
  }

  const missing = skill.parameters
    .filter(param => param.required && !params[param.name])
    .map(param => param.name);
  if (missing.length) {
    return {
      error: currentLanguage === "zh"
        ? `缺少必填技能字段：${missing.join(", ")}`
        : `Missing required skill fields: ${missing.join(", ")}`,
    };
  }

  return { skill, params };
}

function buildModeRequest(mode, text) {
  if (mode === "agent") {
    if (!text) return { error: currentLanguage === "zh" ? "请描述要交给 Agent 的任务。" : "Describe the task for the agent." };
    return {
      endpoint: "/api/agent/chat",
      request: { message: text },
      userDisplay: text,
    };
  }

  if (mode === "research") {
    if (!text) return { error: currentLanguage === "zh" ? "请描述要研究的主题。" : "Describe the topic to research." };
    return {
      endpoint: "/api/research",
      request: {
        topic: text,
        focus: fieldValue("research-focus"),
        max_sources: Number(fieldValue("research-sources") || 4),
        save_findings: !!document.getElementById("research-save")?.checked,
      },
      userDisplay: text,
    };
  }

  if (mode === "ask") {
    if (!text) return { error: currentLanguage === "zh" ? "请描述你想生成投影的任务。" : "Describe the task you want a projection for." };
    return {
      endpoint: "/api/ask",
      request: { query: text },
      userDisplay: text,
    };
  }

  if (mode === "learn") {
    if (!text) return { error: currentLanguage === "zh" ? "请粘贴要摄取的观察或源文本。" : "Paste an observation or source text to ingest." };
    return {
      endpoint: "/api/learn",
      request: {
        text,
        task: fieldValue("learn-task") || "knowledge intake",
        source: fieldValue("learn-source"),
      },
      userDisplay: text,
    };
  }

  if (mode === "explore") {
    const concept = fieldValue("explore-concept") || text;
    if (!concept) return { error: currentLanguage === "zh" ? "请选择一个要检查的概念。" : "Choose a concept to inspect." };
    return {
      endpoint: `/api/explore/${encodeURIComponent(concept)}`,
      method: "GET",
      userDisplay: concept,
    };
  }

  if (mode === "connect") {
    const parsed = parseConnect(text);
    if (parsed.error) return parsed;
    return {
      endpoint: "/api/connect",
      request: parsed,
      userDisplay: `${parsed.source} → ${parsed.relation_type} → ${parsed.target}`,
    };
  }

  if (mode === "skill") {
    if (!window._selectedSkill) {
      return { showSkillPicker: true };
    }
    const built = buildSkillParams(text);
    if (built.error) return built;
    return {
      endpoint: "/api/skills/run",
      request: { skill_name: built.skill.name, params: built.params },
      userDisplay: `[Skill] ${built.skill.name}`,
      clearSelectedSkill: true,
    };
  }

  return { error: currentLanguage === "zh" ? `不支持的模式：${mode}` : `Unsupported mode: ${mode}` };
}

async function sendMessage() {
  if (isProcessing) return;
  const input = $("#user-input");
  const submittedInput = input.value;
  const text = submittedInput.trim();
  const mode = $("#mode-select").value;
  const built = buildModeRequest(mode, text);

  if (built.showSkillPicker) {
    showSkillPicker();
    return;
  }
  if (built.error) {
    addMessage("error", built.error);
    return;
  }

  isProcessing = true;
  $("#send-btn").disabled = true;
  addMessage("user", built.userDisplay);
  addTypingIndicator();

  try {
    const requestOptions = built.method === "GET"
      ? { method: "GET" }
      : {
          method: built.method || "POST",
          body: JSON.stringify(built.request),
        };

    const data = await apiCall(built.endpoint, requestOptions);
    removeTypingIndicator();

    if (data.tool_calls && data.tool_calls.length > 0) {
      showToolCalls(data.tool_calls);
    }

    addMessage("agent", data.message, data.type || "markdown");

    if (built.clearSelectedSkill) {
      window._selectedSkill = null;
      updateSkillBadge();
    }

    if (input.value === submittedInput) {
      input.value = "";
      input.style.height = "auto";
    }
  } catch (err) {
    removeTypingIndicator();
    addMessage("error", t("requestFailed", { message: err.message }));
  } finally {
    isProcessing = false;
    $("#send-btn").disabled = false;
    focusInput();
    await refreshWorkspace();
  }
}

function buildToolCallPairs(calls) {
  const container = $("#messages");
  const callPairs = [];
  for (let i = 0; i < calls.length; i += 2) {
    const call = calls[i];
    const result = calls[i + 1];
    if (call && call.phase === "call") {
      const args = Object.entries(call.args || {})
        .map(([k, v]) => `${k}: ${typeof v === 'string' ? v.slice(0, 80) : v}`)
        .join(", ");
      const status = result ? (result.success ? "ok" : "err") : "...";
      const statusColor = result?.success ? "var(--green)" : "var(--orange)";
      callPairs.push(
        `<div style="display:flex;align-items:center;gap:6px;padding:3px 0">` +
        `<span style="color:var(--purple);font-weight:600">${call.tool}</span>` +
        `<span style="color:var(--text-muted);font-size:11px">${args}</span>` +
        `<span style="margin-left:auto;color:${statusColor};font-size:10px;font-weight:600">${status}</span>` +
        `</div>`
      );
    }
  }

  return callPairs;
}

function appendToolCallMessage(calls, options = {}) {
  const container = $("#messages");
  const div = document.createElement("div");
  div.className = "message msg-system";
  const callPairs = buildToolCallPairs(calls);

  if (callPairs.length === 0) return;
  const subtitle = options.subtitle || `${callPairs.length} tool${callPairs.length > 1 ? 's' : ''} used`;

  div.innerHTML = `
    <div class="message-avatar" style="background:var(--bg-card);color:var(--purple)">T</div>
    <div class="message-content">
      <div class="message-header">
        <span class="name">Tool Calls</span>
        <span>${options.timestamp ? formatMessageTime(options.timestamp) : subtitle}</span>
      </div>
      <div class="message-body" style="font-size:12px;font-family:'SF Mono',Menlo,monospace">
        ${callPairs.join("")}
      </div>
    </div>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

// ── Tool call display ──
function showToolCalls(calls) {
  appendToolCallMessage(calls);
}

// ── Session management ──
function renderWelcome(title = "World 0", body = "Ready for a fresh concept-world session.") {
  const container = $("#messages");
  container.innerHTML = `
    <div class="welcome" id="welcome-screen">
      <div class="welcome-logo">&#9673;</div>
      <h2>${escapeHtml(title)}</h2>
      <p>${escapeHtml(body)}</p>
    </div>
  `;
}

function renderStoredMessage(msg) {
  const roleMap = {
    user: "user",
    assistant: "agent",
    system: "system",
    tool_call: "system",
    tool_result: "system",
  };
  const role = roleMap[msg.role] || "system";
  const type = role === "user" ? "text" : "markdown";
  addMessage(role, msg.content, type, { timestamp: msg.timestamp });
}

function collectStoredToolCalls(messages, startIndex) {
  const calls = [];
  let index = startIndex;
  let firstTimestamp = null;

  while (index < messages.length) {
    const current = messages[index];
    if (!current || current.role !== "tool_call") break;

    let parsed = {};
    try {
      parsed = JSON.parse(current.content || "{}");
    } catch (_) {
      parsed = {};
    }

    const callEntry = {
      phase: "call",
      tool: parsed.name || current.metadata?.tool_name || "tool",
      args: parsed.arguments || {},
    };
    calls.push(callEntry);
    if (!firstTimestamp) firstTimestamp = current.timestamp;

    const next = messages[index + 1];
    if (
      next &&
      next.role === "tool_result" &&
      (
        !current.metadata?.tool_id ||
        !next.metadata?.tool_id ||
        current.metadata.tool_id === next.metadata.tool_id
      )
    ) {
      calls.push({
        phase: "result",
        success: next.metadata?.success !== false,
        output_preview: (next.content || "").slice(0, 200),
      });
      index += 2;
      continue;
    }

    index += 1;
  }

  return {
    calls,
    nextIndex: index,
    timestamp: firstTimestamp,
  };
}

function renderSession(session) {
  if (!session || !session.messages || !session.messages.length) {
    renderWelcome("World 0", "This session is empty. Submit an observation or request a projection.");
    return;
  }
  const container = $("#messages");
  container.innerHTML = "";
  let i = 0;
  while (i < session.messages.length) {
    const msg = session.messages[i];
    if (msg.role === "tool_call") {
      const grouped = collectStoredToolCalls(session.messages, i);
      if (grouped.calls.length) {
        appendToolCallMessage(grouped.calls, {
          timestamp: grouped.timestamp,
          subtitle: `${grouped.calls.filter(item => item.phase === "call").length} tool${grouped.calls.filter(item => item.phase === "call").length > 1 ? "s" : ""} used`,
        });
        i = grouped.nextIndex;
        continue;
      }
    }
    if (msg.role !== "tool_result") {
      renderStoredMessage(msg);
    }
    i += 1;
  }
}

async function saveSession() {
  try {
    const data = await apiCall("/api/sessions/save", { method: "POST" });
    addMessage("system", t("sessionSaved", { id: data.session_id }));
    await loadEnvironmentStatus();
  } catch (err) {
    addMessage("error", t("saveSessionFailed"));
  }
}

async function newSession() {
  try {
    const data = await apiCall("/api/sessions/new", { method: "POST" });
    renderWelcome(
      currentLanguage === "zh" ? "新会话" : "New Session",
      t("newSessionReady", { id: data.session_id }),
    );
    await loadEnvironmentStatus();
  } catch (err) {
    addMessage("error", t("createSessionFailed"));
  }
}

async function showSessionsModal() {
  try {
    const data = await apiCall("/api/sessions");
    const sessions = data.sessions || [];
    const rows = sessions.length
      ? sessions.map(session => `
          <div class="session-row">
            <div>
              <strong>${escapeHtml(session.title)}</strong>
              <span>${escapeHtml(session.summary)}</span>
              <span>${escapeHtml(session.updated_at)}</span>
            </div>
            <div class="session-actions">
              <button onclick="resumeSession('${escapeHtml(session.id)}')">${t("resume")}</button>
            </div>
          </div>
        `).join("")
      : `<div class="mode-note">${t("noSavedSessions")}</div>`;

    showModal("sessions-modal", t("sessionsTitle"), rows);
  } catch (err) {
    addMessage("error", t("loadSessionsFailed", { message: err.message }));
  }
}

async function resumeSession(sessionId) {
  try {
    const data = await apiCall("/api/sessions/resume", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    });
    if (!data.success) {
      throw new Error(`Session '${sessionId}' could not be resumed.`);
    }
    closeModal("sessions-modal");
    renderSession(data.session);
    addMessage(
      "system",
      t("sessionResumed", {
        title: data.session.title,
        id: data.session.id,
      }),
    );
    await loadEnvironmentStatus();
  } catch (err) {
    addMessage("error", t("resumeSessionFailed", { message: err.message }));
  }
}

function showModal(id, title, bodyHtml) {
  closeModal(id);
  const modal = document.createElement("div");
  modal.id = id;
  modal.className = "modal-shell";
  modal.innerHTML = `
    <div class="modal-card">
      <div class="modal-header">
        <h3>${escapeHtml(title)}</h3>
        <button class="modal-close" onclick="closeModal('${id}')">${t("close")}</button>
      </div>
      <div class="modal-body">${bodyHtml}</div>
    </div>
  `;
  document.body.appendChild(modal);
}

function modelOptionsForProvider(provider, selectedModel = "") {
  const models = environmentStatus?.suggested_models?.[provider] || [];
  const options = [
    ...models.map(model => `
      <option value="${escapeHtml(model)}" ${selectedModel === model ? "selected" : ""}>
        ${escapeHtml(model)}
      </option>
    `),
  ];
  return options.join("");
}

function buildPromptSettingsPanel(promptsData) {
  const prompts = promptsData?.prompts || [];
  if (!prompts.length) {
    return `<div class="mode-note">${t("noPromptsAvailable")}</div>`;
  }
  const options = prompts.map(prompt => `
    <option value="${escapeHtml(prompt.id)}">
      ${escapeHtml(prompt.id)}${prompt.is_overridden ? " *" : ""}
    </option>
  `).join("");
  return `
    <div class="prompt-settings">
      <div class="prompt-list field-group">
        <label for="settings-prompt-select">${t("promptSelect")}</label>
        <select id="settings-prompt-select" size="10" onchange="refreshPromptEditor()">
          ${options}
        </select>
        <div class="field-help">${escapeHtml(promptsData?.config_path || "")}</div>
        <button type="button" class="modal-close" onclick="exportPrompts()">${t("exportPrompts")}</button>
      </div>
      <div class="prompt-editor">
        <div class="prompt-meta-row" id="settings-prompt-meta"></div>
        <div class="field-group">
          <label for="settings-prompt-template">${t("promptTemplate")}</label>
          <textarea id="settings-prompt-template" spellcheck="false"></textarea>
        </div>
        <details>
          <summary style="color:var(--text-secondary);cursor:pointer">${t("promptDefault")}</summary>
          <pre id="settings-prompt-default" style="white-space:pre-wrap;margin-top:8px;color:var(--text-secondary);font-size:11px;"></pre>
        </details>
        <div class="prompt-actions">
          <button type="button" class="modal-close" onclick="savePromptOverride()">${t("savePrompt")}</button>
          <button type="button" class="modal-close" onclick="resetPromptOverride()">${t("resetPrompt")}</button>
        </div>
      </div>
    </div>
  `;
}

function selectedPromptRecord() {
  const selectedId = fieldValue("settings-prompt-select");
  const prompts = window._promptSettings?.prompts || [];
  return prompts.find(prompt => prompt.id === selectedId) || prompts[0] || null;
}

function refreshPromptEditor() {
  const prompt = selectedPromptRecord();
  const select = document.getElementById("settings-prompt-select");
  const meta = document.getElementById("settings-prompt-meta");
  const editor = document.getElementById("settings-prompt-template");
  const defaultBox = document.getElementById("settings-prompt-default");
  if (!prompt || !select || !meta || !editor || !defaultBox) return;
  if (!select.value) select.value = prompt.id;
  editor.value = prompt.template || "";
  defaultBox.textContent = prompt.default_template || "";
  const variables = (prompt.variables || []).join(", ") || "—";
  const badges = [
    `<span class="prompt-badge">${escapeHtml(prompt.output || "text")}</span>`,
    `<span class="prompt-badge">${escapeHtml(variables)}</span>`,
  ];
  if (prompt.is_overridden) {
    badges.push(`<span class="prompt-badge active">${t("promptOverridden")}</span>`);
  }
  if (prompt.has_active_override) {
    badges.push(`<span class="prompt-badge active">${t("promptActiveOverride")}</span>`);
  }
  meta.innerHTML = `
    <strong style="color:var(--text-primary)">${escapeHtml(prompt.id)}</strong>
    ${badges.join("")}
    <span style="color:var(--text-secondary);font-size:12px">${escapeHtml(prompt.description || "")}</span>
  `;
}

function updatePromptRecord(prompt) {
  if (!window._promptSettings || !prompt) return;
  const prompts = window._promptSettings.prompts || [];
  const index = prompts.findIndex(item => item.id === prompt.id);
  if (index >= 0) {
    prompts[index] = prompt;
  } else {
    prompts.push(prompt);
  }
  const select = document.getElementById("settings-prompt-select");
  if (select) {
    [...select.options].forEach(option => {
      if (option.value === prompt.id) {
        option.textContent = `${prompt.id}${prompt.is_overridden ? " *" : ""}`;
      }
    });
  }
}

async function savePromptOverride() {
  const prompt = selectedPromptRecord();
  if (!prompt) return;
  try {
    const data = await apiCall(`/api/prompts/${encodeURIComponent(prompt.id)}`, {
      method: "POST",
      body: JSON.stringify({ template: rawFieldValue("settings-prompt-template") }),
    });
    updatePromptRecord(data.prompt);
    refreshPromptEditor();
    addMessage("system", t("promptSaved", { id: prompt.id }));
  } catch (err) {
    addMessage("error", t("promptActionFailed", { message: err.message }));
  }
}

async function resetPromptOverride() {
  const prompt = selectedPromptRecord();
  if (!prompt) return;
  try {
    const data = await apiCall(`/api/prompts/${encodeURIComponent(prompt.id)}/reset`, {
      method: "POST",
    });
    updatePromptRecord(data.prompt);
    refreshPromptEditor();
    addMessage("system", t("promptReset", { id: prompt.id }));
  } catch (err) {
    addMessage("error", t("promptActionFailed", { message: err.message }));
  }
}

async function exportPrompts() {
  try {
    const data = await apiCall("/api/prompts/export");
    const blob = new Blob([JSON.stringify(data, null, 2) + "\n"], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "world0-prompts.json";
    link.click();
    URL.revokeObjectURL(url);
    addMessage("system", t("promptExported"));
  } catch (err) {
    addMessage("error", t("promptActionFailed", { message: err.message }));
  }
}

async function showSettingsModal() {
  try {
    const [meta, spacesData, promptsData] = await Promise.all([
      apiCall("/api/agent/status"),
      apiCall("/api/spaces").catch(() => ({ spaces: [], active_space_id: null })),
      apiCall("/api/prompts").catch(() => ({ prompts: [] })),
    ]);
    const settings = meta.settings || {};
    const providerEnv = meta.provider_env || {};
    const spaces = spacesData.spaces || [];
    const activeSpaceId = spacesData.active_space_id;
    const activeSpace = spaces.find(s => s.id === activeSpaceId);
    const spaceOptions = spaces.map(s => `
      <option value="${escapeHtml(s.id)}" ${s.id === activeSpaceId ? "selected" : ""}>
        ${escapeHtml(s.name)}${s.is_active ? (currentLanguage === "zh" ? " (当前)" : " (current)") : ""}
      </option>
    `).join("");
    const zh = currentLanguage === "zh";
    const spacesPanel = `
      <div class="field-group">
        <label>${zh ? "概念空间 (Space)" : "Concept Space"}</label>
        <div class="field-help" style="margin-bottom:8px;">
          ${zh
            ? `当前空间：<strong>${escapeHtml(activeSpace ? activeSpace.name : "—")}</strong>${activeSpace ? `  <span style="opacity:.7">(${escapeHtml(activeSpace.id)})</span>` : ""}`
            : `Active space: <strong>${escapeHtml(activeSpace ? activeSpace.name : "—")}</strong>${activeSpace ? `  <span style="opacity:.7">(${escapeHtml(activeSpace.id)})</span>` : ""}`}
        </div>
        <div style="display:flex;gap:8px;align-items:center;margin-bottom:10px;">
          <select id="settings-space-select" style="flex:1;">${spaceOptions || `<option value="">${zh ? "（暂无空间）" : "(no spaces yet)"}</option>`}</select>
          <button type="button" class="modal-close" onclick="switchSpace()" ${spaces.length ? "" : "disabled"}>${zh ? "切换" : "Switch"}</button>
        </div>
        <div style="display:flex;gap:8px;align-items:center;">
          <input id="settings-space-new-name" placeholder="${zh ? "新空间名称" : "New space name"}" style="flex:1;" />
          <button type="button" class="modal-close" onclick="createSpaceInline()">${zh ? "新建" : "Create"}</button>
        </div>
        <div class="field-help" style="margin-top:6px;">
          ${zh
            ? "每个空间是一个独立的概念世界，切换后对话将使用该空间的概念与关系。"
            : "Each space is an isolated concept world; conversations use that space's concepts and relations."}
        </div>
      </div>
    `;
    const providerOptions = (meta.providers || []).map(p => `
      <option value="${escapeHtml(p.id)}" ${settings.provider === p.id ? "selected" : ""}>
        ${escapeHtml(p.label)}
      </option>
    `).join("");
    const languageOptions = (meta.languages || []).map(lang => `
      <option value="${escapeHtml(lang.id)}" ${settings.language === lang.id ? "selected" : ""}>
        ${escapeHtml(lang.label)}
      </option>
    `).join("");
    const suggested = (meta.suggested_models?.[settings.provider] || []).join(", ");
    const providerMeta = providerEnv[settings.provider] || {};
    const envHint = providerMeta.api_key_env
      ? `${t("apiKeyHint")} ${providerMeta.available ? t("envDetected") : t("envNotDetected")}: ${providerMeta.api_key_env}${
          providerMeta.endpoint_env ? `, ${providerMeta.endpoint_env}` : ""
        }`
      : t("apiKeyHint");
    const presetModels = modelOptionsForProvider(
      settings.provider,
      settings.model || "",
    );
    const autoSedimentChecked = settings.auto_sediment_dialogue !== false ? "checked" : "";
    const sedimentInterval = Number(settings.dialogue_sediment_interval || 1);
    const promptPanel = buildPromptSettingsPanel(promptsData);
    const body = `
      <div class="mode-fields">
        ${spacesPanel}
        <div class="field-group">
          <label for="settings-language">${t("language")}</label>
          <select id="settings-language">${languageOptions}</select>
        </div>
        <div class="field-group">
          <label for="settings-provider">${t("provider")}</label>
          <select id="settings-provider" onchange="refreshSettingsModelHint(); refreshSettingsModelSelect()">${providerOptions}</select>
        </div>
        <div class="field-group">
          <label for="settings-model-preset">${t("modelPreset")}</label>
          <select id="settings-model-preset">${presetModels}</select>
        </div>
        <div class="field-group">
          <label for="settings-model">${t("customModel")}</label>
          <input id="settings-model" value="" placeholder="codex / gpt-5.4 / claude / claude-sonnet-4-6 / your deployment name" />
          <div class="field-help">${currentLanguage === "zh" ? `当前生效模型：${escapeHtml(settings.model || "") || "—"}` : `Current effective model: ${escapeHtml(settings.model || "") || "—"}`}</div>
        </div>
        <div class="field-group">
          <label for="settings-api-key">${t("apiKey")}</label>
          <input id="settings-api-key" type="password" value="${escapeHtml(settings.api_key || "")}" />
          <div class="field-help" id="settings-api-key-hint">${escapeHtml(envHint)}</div>
        </div>
        <div class="field-group">
          <label for="settings-base-url">${t("baseUrl")}</label>
          <input id="settings-base-url" value="${escapeHtml(settings.base_url || "")}" />
        </div>
        <div class="field-group">
          <label for="settings-azure-endpoint">${t("azureEndpoint")}</label>
          <input id="settings-azure-endpoint" value="${escapeHtml(settings.azure_endpoint || "")}" />
        </div>
        <div class="field-group">
          <label for="settings-api-version">${t("apiVersion")}</label>
          <input id="settings-api-version" value="${escapeHtml(settings.api_version || "2024-10-21")}" />
        </div>
        <div class="field-group">
          <label for="settings-auto-sediment-dialogue">${t("autoSedimentDialogue")}</label>
          <label style="display:flex;align-items:center;gap:10px;font-weight:500;">
            <input
              id="settings-auto-sediment-dialogue"
              type="checkbox"
              ${autoSedimentChecked}
              onchange="refreshDialogueSedimentFields()"
            />
            <span>${t("autoSedimentDialogueHelp")}</span>
          </label>
        </div>
        <div class="field-group">
          <label for="settings-dialogue-sediment-interval">${t("dialogueSedimentInterval")}</label>
          <input
            id="settings-dialogue-sediment-interval"
            type="number"
            min="1"
            max="20"
            step="1"
            value="${Number.isFinite(sedimentInterval) ? sedimentInterval : 1}"
          />
          <div class="field-help">${t("dialogueSedimentIntervalHelp")}</div>
        </div>
      </div>
      <div style="margin-top:16px">
        <div class="field-group">
          <label>${t("promptRegistry")}</label>
        </div>
        ${promptPanel}
      </div>
      <div class="field-help" id="settings-model-hint">
        ${suggested ? `Suggested: ${escapeHtml(suggested)}` : ""}
      </div>
      <div style="margin-top:14px">
        <button class="modal-close" onclick="saveSettings()">${t("saveSettings")}</button>
      </div>
    `;
    showModal("settings-modal", t("settingsTitle"), body);
    window._promptSettings = promptsData;
    refreshSettingsModelHint();
    refreshDialogueSedimentFields();
    refreshPromptEditor();
  } catch (err) {
    addMessage("error", t("updateSettingsFailed", { message: err.message }));
  }
}

function refreshSettingsModelSelect() {
  const provider = fieldValue("settings-provider");
  const select = document.getElementById("settings-model-preset");
  if (!select) return;
  const models = environmentStatus?.suggested_models?.[provider] || [];
  const current = models.includes(select.value) ? select.value : (models[0] || "");
  select.innerHTML = modelOptionsForProvider(provider, current);
}

function refreshSettingsModelHint() {
  const provider = fieldValue("settings-provider");
  const hint = document.getElementById("settings-model-hint");
  const keyHint = document.getElementById("settings-api-key-hint");
  const suggested = environmentStatus?.suggested_models?.[provider] || [];
  const providerMeta = environmentStatus?.provider_env?.[provider] || {};
  if (!hint) return;
  hint.textContent = suggested.length
    ? (currentLanguage === "zh"
        ? `建议模型：${suggested.join(", ")}`
        : `Suggested: ${suggested.join(", ")}`)
    : "";
  if (keyHint) {
    if (providerMeta.api_key_env) {
      keyHint.textContent = `${t("apiKeyHint")} ${providerMeta.available ? t("envDetected") : t("envNotDetected")}: ${providerMeta.api_key_env}${
        providerMeta.endpoint_env ? `, ${providerMeta.endpoint_env}` : ""
      }`;
    } else {
      keyHint.textContent = t("apiKeyHint");
    }
  }
}

function refreshDialogueSedimentFields() {
  const enabled = !!document.getElementById("settings-auto-sediment-dialogue")?.checked;
  const interval = document.getElementById("settings-dialogue-sediment-interval");
  if (!interval) return;
  interval.disabled = !enabled;
  interval.style.opacity = enabled ? "1" : "0.55";
}

async function saveSettings() {
  try {
    const customModel = fieldValue("settings-model");
    const presetModel = fieldValue("settings-model-preset");
    const autoSedimentDialogue = !!document.getElementById("settings-auto-sediment-dialogue")?.checked;
    const rawInterval = Number(fieldValue("settings-dialogue-sediment-interval") || "1");
    const dialogueSedimentInterval = Math.min(20, Math.max(1, Math.round(rawInterval || 1)));
    const payload = {
      language: fieldValue("settings-language") || "en",
      provider: fieldValue("settings-provider") || "none",
      model: customModel || presetModel,
      api_key: fieldValue("settings-api-key"),
      base_url: fieldValue("settings-base-url"),
      azure_endpoint: fieldValue("settings-azure-endpoint"),
      api_version: fieldValue("settings-api-version") || "2024-10-21",
      auto_sediment_dialogue: autoSedimentDialogue,
      dialogue_sediment_interval: dialogueSedimentInterval,
    };
    await apiCall("/api/settings", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    currentLanguage = payload.language;
    applyTranslations();
    closeModal("settings-modal");
    addMessage("system", t("settingsUpdated"));
    await loadEnvironmentStatus();
  } catch (err) {
    addMessage("error", t("updateSettingsFailed", { message: err.message }));
  }
}

async function switchSpace() {
  const target = fieldValue("settings-space-select");
  if (!target) return;
  try {
    const resp = await apiCall("/api/spaces/use", {
      method: "POST",
      body: JSON.stringify({ target }),
    });
    if (resp && resp.success === false) {
      addMessage("error", resp.error || "switch failed");
      return;
    }
    const active = (resp.spaces || []).find(s => s.is_active);
    const name = active ? active.name : target;
    const zh = currentLanguage === "zh";
    addMessage("system", zh ? `已切换到空间：${name}` : `Switched to space: ${name}`);
    closeModal("settings-modal");
    await loadEnvironmentStatus();
    await showSettingsModal();
  } catch (err) {
    addMessage("error", err.message);
  }
}

async function createSpaceInline() {
  const name = fieldValue("settings-space-new-name");
  const zh = currentLanguage === "zh";
  if (!name || !name.trim()) {
    addMessage("error", zh ? "请填写空间名称" : "Please enter a space name");
    return;
  }
  try {
    const resp = await apiCall("/api/spaces/create", {
      method: "POST",
      body: JSON.stringify({ name: name.trim(), description: "", activate: true }),
    });
    if (resp && resp.success === false) {
      addMessage("error", resp.error || "create failed");
      return;
    }
    addMessage("system", zh ? `已创建并切换到空间：${name.trim()}` : `Created and switched to space: ${name.trim()}`);
    closeModal("settings-modal");
    await loadEnvironmentStatus();
    await showSettingsModal();
  } catch (err) {
    addMessage("error", err.message);
  }
}

function closeModal(id) {
  const modal = document.getElementById(id);
  if (modal) modal.remove();
}

async function loadSkills(force = false) {
  if (!force && availableSkills.length) return availableSkills;
  const data = await apiCall("/api/skills");
  availableSkills = data.skills || [];
  return availableSkills;
}

async function loadRelationTypes(force = false) {
  if (!force && relationTypes.length) return relationTypes;
  const data = await apiCall("/api/relation_types");
  relationTypes = data.types || [];
  return relationTypes;
}

async function loadEnvironmentStatus() {
  try {
    const [agent, mcp, sessions] = await Promise.all([
      apiCall("/api/agent/status"),
      apiCall("/api/mcp/status"),
      apiCall("/api/sessions"),
    ]);
    environmentStatus = agent;
    currentLanguage = agent.language || currentLanguage;
    applyTranslations();
    const runtimeState = agent.state || {};
    if (agent.agentic_ready) {
      $("#env-agentic").textContent = `${runtimeState.status || "ready"} · ${agent.provider || "ready"} · ${agent.model || "default"}`;
    } else if (runtimeState.reason) {
      $("#env-agentic").textContent = `${runtimeState.status || "blocked"} · ${runtimeState.reason}`;
    } else {
      $("#env-agentic").textContent = agent.llm_enabled ? t("llmConfigured") : t("llmOff");
    }
    $("#env-skills").textContent = `${availableSkills.length} ${currentLanguage === "zh" ? "已加载" : "loaded"}`;
    $("#env-mcp").textContent = `${mcp.connected || 0} ${t("connected")}`;
    $("#env-sessions").textContent = `${(sessions.sessions || []).length} ${t("recent")}`;
    $("#env-store").textContent = `${t("environmentStore")}: ${agent.store_path || "—"}`;
    const hint = $("#env-hint");
    if (agent.unavailable_reason) {
      hint.style.display = "block";
      hint.textContent = agent.unavailable_reason;
    } else {
      hint.style.display = "none";
      hint.textContent = "";
    }
  } catch (err) {
    console.error("Failed to load environment status:", err);
  }
}

async function refreshWorkspace() {
  await Promise.all([
    refreshSidebar(),
    loadEnvironmentStatus(),
  ]);
}

// ── Sidebar ──
async function refreshSidebar() {
  try {
    const [statusData, conceptsData] = await Promise.all([
      apiCall("/api/status"),
      apiCall("/api/concepts"),
    ]);

    // Stats
    $("#stat-concepts").textContent = statusData.total_concepts;
    $("#stat-relations").textContent = statusData.total_relations;
    $("#stat-confidence").textContent = statusData.avg_confidence.toFixed(2);
    $("#concept-count").textContent = statusData.total_concepts;

    // Concept list
    const list = $("#concept-list");
    const concepts = conceptsData.concepts || [];

    if (concepts.length === 0) {
      list.innerHTML = `
        <div class="sidebar-empty" id="sidebar-empty">
          <div style="font-size: 24px; opacity: 0.3; margin-bottom: 8px;">&#9673;</div>
          <div>${t("noConceptsWorld")}</div>
          <div style="font-size: 11px; margin-top: 4px;">${t("startObservation")}</div>
        </div>
      `;
      return;
    }

    // Preserve search filter
    const searchQ = $("#sidebar-search").value.trim().toLowerCase();
    const filtered = searchQ
      ? concepts.filter(c =>
          c.name.toLowerCase().includes(searchQ) ||
          c.description.toLowerCase().includes(searchQ) ||
          c.aliases.some(a => a.toLowerCase().includes(searchQ))
        )
      : concepts;

    list.innerHTML = filtered.map(c => `
      <div class="concept-item" onclick='showConceptCard(${JSON.stringify(c.name)})'>
        <span class="concept-dot mat-${c.maturity}"></span>
        <span class="concept-name">${c.name}</span>
        <span class="concept-meta">${c.confidence.toFixed(2)}</span>
      </div>
    `).join("");

  } catch (err) {
    console.error("Failed to refresh sidebar:", err);
  }
}

function exploreConcept(name) {
  setMode("explore");
  const field = document.getElementById("explore-concept");
  if (field) {
    field.value = name;
  } else {
    $("#user-input").value = name;
  }
  sendMessage();
}

async function showConceptCard(name) {
  try {
    const data = await apiCall(`/api/concepts/${encodeURIComponent(name)}/card`);
    const c = data.card;
    const relations = (c.relations || []).map(rel => `
      <div class="concept-relation-row">
        <span>${rel.direction === "outgoing" ? "→" : "←"}</span>
        <strong>${escapeHtml(rel.semantic_relation || rel.relation_type)}</strong>
        <span style="color:var(--text-muted)">[${escapeHtml(rel.relation_type)}]</span>
        <span>${escapeHtml(rel.other_name)}</span>
        <span class="weight">p ${Number(rel.propagation_strength || rel.weight || 0).toFixed(2)}</span>
      </div>
    `).join("") || `<div class="empty">${t("noRelations")}</div>`;

    const activity = (c.recent_activity || []).slice().reverse().map(item => `
      <div class="concept-relation-row">
        <span>${formatDateTime(item.timestamp)}</span>
        <span>${escapeHtml(item.task || item.source || "—")}</span>
      </div>
    `).join("") || `<div class="empty">${t("noActivity")}</div>`;

    const chips = values => values && values.length
      ? `<div class="concept-chip-row">${values.map(v => `<span class="concept-chip">${escapeHtml(v)}</span>`).join("")}</div>`
      : `<div class="empty">—</div>`;

    const body = `
      <div class="concept-card-grid">
        <div class="concept-card-hero">
          <div class="concept-card-title">
            <h4>${escapeHtml(c.name)}</h4>
            <div>${escapeHtml(c.description || t("noDescription"))}</div>
            <div class="concept-chip-row">
              <span class="concept-chip">${t("maturityLabel")}: ${escapeHtml(c.maturity)}</span>
              <span class="concept-chip">${t("confidenceLabel")}: ${Number(c.confidence || 0).toFixed(2)}</span>
              <span class="concept-chip">${t("relationCountLabel")}: ${c.relation_count || 0}</span>
            </div>
          </div>
          <div class="concept-card-metrics">
            <div class="concept-metric"><strong>${c.activation_count || 0}</strong><span>${t("activationCountLabel")}</span></div>
            <div class="concept-metric"><strong>${c.relation_count || 0}</strong><span>${t("relationCountLabel")}</span></div>
            <div class="concept-metric"><strong>${formatDateTime(c.last_activated)}</strong><span>${t("lastActivatedLabel")}</span></div>
            <div class="concept-metric"><strong>${formatDateTime(c.created_at)}</strong><span>${t("createdAtLabel")}</span></div>
          </div>
        </div>
        <div class="concept-card-actions">
          <button onclick='closeModal("concept-card-modal"); exploreConcept(${JSON.stringify(c.name)})'>${t("inspectConcept")}</button>
          <button onclick='closeModal("concept-card-modal"); setMode("ask"); document.getElementById("user-input").value = ${JSON.stringify(c.name)}; focusInput();'>${t("projectFromConcept")}</button>
        </div>
        <div class="concept-card-section">
          <h4>${t("aliasesLabel")}</h4>
          ${chips(c.aliases || [])}
        </div>
        <div class="concept-card-section">
          <h4>${t("tagsLabel")}</h4>
          ${chips(c.tags || [])}
        </div>
        <div class="concept-card-section">
          <h4>${t("sourcesLabel")}</h4>
          ${chips(c.sources || [])}
        </div>
        <div class="concept-card-section">
          <h4>${t("tasksLabel")}</h4>
          ${chips(c.tasks || [])}
        </div>
        <div class="concept-card-section">
          <h4>${t("relationsLabel")}</h4>
          <div class="concept-card-list">${relations}</div>
        </div>
        <div class="concept-card-section">
          <h4>${t("recentActivityLabel")}</h4>
          <div class="concept-card-list">${activity}</div>
        </div>
      </div>
    `;
    showModal("concept-card-modal", `${t("conceptCard")} · ${c.name}`, body);
  } catch (err) {
    addMessage("error", t("conceptCardLoadFailed", { message: err.message }));
  }
}

async function doReflect() {
  addTypingIndicator();
  try {
    const data = await apiCall("/api/reflect", { method: "POST" });
    removeTypingIndicator();
    addMessage("agent", data.message, "markdown");
    await refreshWorkspace();
  } catch (err) {
    removeTypingIndicator();
    addMessage("error", t("reflectFailed", { message: err.message }));
  }
}

// ── Skill Picker ──
async function showSkillPicker() {
  try {
    const skills = await loadSkills();
    if (skills.length === 0) {
      addMessage("system", t("noSkillsAvailable"));
      return;
    }

    const body = skills.map(s => `
          <div class="session-row">
            <div>
              <strong>${escapeHtml(s.name)}</strong>
              <span>${escapeHtml(s.description)}</span>
          <span>${s.parameters.length ? `${currentLanguage === "zh" ? "参数" : "Params"}: ${escapeHtml(s.parameters.map(p => p.name).join(", "))}` : t("noParams")}</span>
            </div>
            <div class="session-actions">
          <button onclick="selectSkill('${escapeHtml(s.name)}')">${t("use")}</button>
            </div>
          </div>
    `).join("");
    showModal("skill-picker", t("selectSkill"), body);
  } catch (err) {
    addMessage("error", t("loadSkillsFailed", { message: err.message }));
  }
}

function selectSkill(name) {
  window._selectedSkill = name;
  closeModal("skill-picker");
  setMode("skill");
  updateSkillBadge();
  addMessage("system", t("skillSelected", { name }));
  focusInput();
}

function updateSkillBadge() {
  let badge = $("#skill-badge");
  if (window._selectedSkill) {
    if (!badge) {
      badge = document.createElement("div");
      badge.id = "skill-badge";
      badge.style.cssText = "padding:4px 10px;background:var(--purple);color:#fff;border-radius:4px;font-size:11px;font-weight:600;display:inline-block;margin-bottom:4px;cursor:pointer;";
      badge.title = "Click to change skill";
      badge.onclick = showSkillPicker;
      const inputArea = $("#input-area");
      inputArea.insertBefore(badge, inputArea.firstChild);
    }
    badge.textContent = "Skill: " + window._selectedSkill + " ×";
  } else if (badge) {
    badge.remove();
  }
  renderModeContext();
}

// Sidebar search
$("#sidebar-search").addEventListener("input", () => refreshSidebar());

// ── Tabs ──
function switchTab(tab) {
  $$(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === tab));
  $("#chat-panel").style.display = tab === "chat" ? "flex" : "none";
  const gp = $("#graph-panel");
  gp.style.display = tab === "graph" ? "block" : "none";
  gp.classList.toggle("active", tab === "graph");

  if (tab === "graph") {
    loadGraph();
  }
}

// ── Graph ──
let graphSim = null;

async function loadGraph() {
  try {
    const data = await apiCall("/api/graph");
    renderGraph(data);
  } catch (err) {
    console.error("Graph load failed:", err);
  }
}

function renderGraph(data) {
  const svgEl = document.getElementById("graph-svg");
  const panel = document.getElementById("graph-panel");
  const tooltip = document.getElementById("graph-tooltip");

  // Clear previous
  d3.select(svgEl).selectAll("*").remove();
  if (graphSim) graphSim.stop();

  if (!data.nodes.length) {
    d3.select(svgEl).append("text")
      .attr("x", "50%").attr("y", "50%")
      .attr("text-anchor", "middle")
      .attr("fill", "#484f58")
      .attr("font-size", "14px")
      .text("No concept-world yet — submit an observation first");
    return;
  }

  const width = panel.clientWidth;
  const height = panel.clientHeight;
  const svg = d3.select(svgEl);
  const g = svg.append("g");

  // Zoom
  const zoom = d3.zoom().scaleExtent([0.2, 5])
    .on("zoom", e => g.attr("transform", e.transform));
  svg.call(zoom);

  // Arrow markers
  const defs = svg.append("defs");
  Object.entries(RELATION_COLORS).forEach(([type, color]) => {
    defs.append("marker")
      .attr("id", `ga-${type}`).attr("viewBox", "0 -5 10 10")
      .attr("refX", 20).attr("refY", 0)
      .attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto")
      .append("path").attr("d", "M0,-4L10,0L0,4").attr("fill", color).attr("opacity", 0.6);
  });

  // Simulation
  graphSim = d3.forceSimulation(data.nodes)
    .force("link", d3.forceLink(data.edges).id(d => d.id).distance(100))
    .force("charge", d3.forceManyBody().strength(-250))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(d => nr(d) + 6));

  function nr(d) { return 6 + d.confidence * 14 + Math.min(d.connections, 8); }

  const link = g.append("g").selectAll("line")
    .data(data.edges).join("line")
    .attr("class", "graph-link")
    .attr("stroke", d => RELATION_COLORS[d.relation_type] || "#484f58")
    .attr("stroke-width", d => Math.max(1, d.weight * 3))
    .attr("marker-end", d => `url(#ga-${d.relation_type})`);

  const linkLabel = g.append("g").selectAll("text")
    .data(data.edges).join("text")
    .attr("class", "graph-link-label")
    .text(d => d.semantic_relation || d.relation_type);

  const node = g.append("g").selectAll("g")
    .data(data.nodes).join("g")
    .call(d3.drag()
      .on("start", (e, d) => { if (!e.active) graphSim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on("end", (e, d) => { if (!e.active) graphSim.alphaTarget(0); d.fx = null; d.fy = null; })
    );

  node.append("circle")
    .attr("class", "graph-node")
    .attr("r", nr)
    .attr("fill", d => MATURITY_COLORS[d.maturity] || "#484f58")
    .attr("stroke", d => MATURITY_COLORS[d.maturity] || "#484f58")
    .attr("stroke-width", 1.5)
    .attr("stroke-opacity", 0.5)
    .on("click", (e, d) => { showConceptCard(d.name); })
    .on("mouseover", (e, d) => {
      tooltip.innerHTML = `<strong>${d.name}</strong><br>${d.maturity} · conf: ${d.confidence}`;
      tooltip.style.left = (e.offsetX + 12) + "px";
      tooltip.style.top = (e.offsetY - 10) + "px";
      tooltip.style.opacity = 1;
    })
    .on("mouseout", () => { tooltip.style.opacity = 0; });

  node.append("text")
    .attr("class", "graph-label")
    .attr("dy", d => nr(d) + 12)
    .text(d => d.name);

  graphSim.on("tick", () => {
    link
      .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    linkLabel
      .attr("x", d => (d.source.x + d.target.x) / 2)
      .attr("y", d => (d.source.y + d.target.y) / 2);
    node.attr("transform", d => `translate(${d.x},${d.y})`);
  });

  // Fit
  setTimeout(() => {
    const bounds = g.node().getBBox();
    if (bounds.width > 0 && bounds.height > 0) {
      const scale = Math.min(width / (bounds.width + 80), height / (bounds.height + 80), 1.5);
      const tx = width / 2 - (bounds.x + bounds.width / 2) * scale;
      const ty = height / 2 - (bounds.y + bounds.height / 2) * scale;
      svg.transition().duration(600).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
    }
  }, 1500);
}

// ── Init ──
Promise.all([
  loadSkills(),
  loadRelationTypes(),
]).finally(() => {
  handleModeChange();
  refreshWorkspace();
  focusInput();
});
</script>
</body>
</html>"""
