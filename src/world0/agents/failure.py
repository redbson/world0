"""Failure taxonomy and recovery hints for World 0 agent operations."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from world0.llm.base import LLMError


class FailureClass(str, Enum):
    NONE = "none"
    LLM_ERROR = "llm_error"
    PROVIDER_AUTH = "provider_auth"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    TOOL_RUNTIME = "tool_runtime"
    MCP_UNAVAILABLE = "mcp_unavailable"
    SEARCH_FETCH_FAILED = "search_fetch_failed"
    SESSION_CORRUPT = "session_corrupt"
    TOOL_ROUND_LIMIT = "tool_round_limit"
    UNKNOWN = "unknown"


class RecoveryAction(str, Enum):
    RETRY_SAME_OPERATION = "retry_same_operation"
    RETRY_WITHOUT_DOMAINS = "retry_without_domains"
    RETRY_WITHOUT_FOCUS = "retry_without_focus"
    SKIP_SOURCE = "skip_source"
    RECONFIGURE_PROVIDER = "reconfigure_provider"
    CHECK_MCP_SERVERS = "check_mcp_servers"
    START_NEW_SESSION = "start_new_session"
    MANUAL_COMPACT = "manual_compact"


class FailureReport(BaseModel):
    failure_class: FailureClass
    message: str
    retryable: bool = False
    context: str = "operation"
    recovery_actions: list[RecoveryAction] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def normalize_failure_class(value: str | None) -> FailureClass:
    if not value:
        return FailureClass.NONE
    try:
        return FailureClass(value)
    except ValueError:
        return FailureClass.UNKNOWN


def recovery_actions_for_failure_class(failure_class: FailureClass) -> list[RecoveryAction]:
    mapping = {
        FailureClass.NONE: [],
        FailureClass.LLM_ERROR: [RecoveryAction.RETRY_SAME_OPERATION],
        FailureClass.PROVIDER_AUTH: [RecoveryAction.RECONFIGURE_PROVIDER],
        FailureClass.PROVIDER_RATE_LIMIT: [RecoveryAction.RETRY_SAME_OPERATION],
        FailureClass.PROVIDER_UNAVAILABLE: [RecoveryAction.RETRY_SAME_OPERATION],
        FailureClass.TOOL_RUNTIME: [RecoveryAction.RETRY_SAME_OPERATION],
        FailureClass.MCP_UNAVAILABLE: [RecoveryAction.CHECK_MCP_SERVERS],
        FailureClass.SEARCH_FETCH_FAILED: [RecoveryAction.RETRY_WITHOUT_FOCUS, RecoveryAction.RETRY_WITHOUT_DOMAINS],
        FailureClass.SESSION_CORRUPT: [RecoveryAction.START_NEW_SESSION],
        FailureClass.TOOL_ROUND_LIMIT: [RecoveryAction.MANUAL_COMPACT],
        FailureClass.UNKNOWN: [],
    }
    return mapping.get(failure_class, [])


def classify_exception(exc: Exception, *, context: str) -> FailureReport:
    message = str(exc).strip() or exc.__class__.__name__
    lower = message.lower()

    if context == "session":
        failure_class = FailureClass.SESSION_CORRUPT
    elif context == "mcp" or "mcp" in lower:
        failure_class = FailureClass.MCP_UNAVAILABLE
    elif any(token in lower for token in ("rate limit", "429", "quota")):
        failure_class = FailureClass.PROVIDER_RATE_LIMIT
    elif any(token in lower for token in ("api key", "unauthorized", "401", "forbidden", "authentication")):
        failure_class = FailureClass.PROVIDER_AUTH
    elif any(token in lower for token in ("timeout", "timed out", "connection", "503", "bad gateway", "temporarily unavailable")):
        failure_class = FailureClass.PROVIDER_UNAVAILABLE if context in {"llm", "provider"} else FailureClass.SEARCH_FETCH_FAILED
    elif context in {"search", "fetch"}:
        failure_class = FailureClass.SEARCH_FETCH_FAILED
    elif isinstance(exc, LLMError) or "chat failed" in lower or "api call failed" in lower:
        failure_class = FailureClass.LLM_ERROR
    elif context == "tool":
        failure_class = FailureClass.TOOL_RUNTIME
    else:
        failure_class = FailureClass.UNKNOWN

    retryable = failure_class in {
        FailureClass.LLM_ERROR,
        FailureClass.PROVIDER_RATE_LIMIT,
        FailureClass.PROVIDER_UNAVAILABLE,
        FailureClass.TOOL_RUNTIME,
        FailureClass.MCP_UNAVAILABLE,
        FailureClass.SEARCH_FETCH_FAILED,
    }
    return FailureReport(
        failure_class=failure_class,
        message=message,
        retryable=retryable,
        context=context,
        recovery_actions=recovery_actions_for_failure_class(failure_class),
    )


def turn_summary_failure_report(summary) -> FailureReport | None:
    """Convert a turn summary-like object into a structured failure report."""
    failure_class = normalize_failure_class(getattr(summary, "failure_class", "none"))
    if failure_class == FailureClass.NONE:
        return None
    message = f"Latest turn ended with {failure_class.value}."
    failed_tools = list(getattr(summary, "failed_tools", []) or [])
    metadata = {
        "stop_reason": getattr(summary, "stop_reason", "end_turn"),
        "rounds": getattr(summary, "rounds", 1),
        "tool_count": getattr(summary, "tool_count", 0),
    }
    if failed_tools:
        metadata["failed_tools"] = failed_tools
    return FailureReport(
        failure_class=failure_class,
        message=message,
        retryable=failure_class in {
            FailureClass.LLM_ERROR,
            FailureClass.PROVIDER_RATE_LIMIT,
            FailureClass.PROVIDER_UNAVAILABLE,
            FailureClass.TOOL_RUNTIME,
            FailureClass.MCP_UNAVAILABLE,
            FailureClass.SEARCH_FETCH_FAILED,
        },
        context="session_turn",
        recovery_actions=recovery_actions_for_failure_class(failure_class),
        metadata=metadata,
    )
