"""Explicit runtime state models for the World 0 agent."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from world0.agents.session import Session


class AgentLifecycleStatus(str, Enum):
    """Stable lifecycle states for the agent runtime."""

    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


class SessionLifecycleStatus(str, Enum):
    """High-level session state for resume/inspection surfaces."""

    ACTIVE = "active"
    COMPACTED = "compacted"
    ATTENTION_NEEDED = "attention_needed"


class SessionStateSnapshot(BaseModel):
    """Machine-readable state of a persisted session."""

    status: SessionLifecycleStatus
    reason: str | None = None
    latest_failure_class: str = "none"
    has_compaction: bool = False
    open_loops: list[str] = Field(default_factory=list)
    latest_turn: dict[str, object] | None = None


class AgentStateSnapshot(BaseModel):
    """Machine-readable state of the current World 0 agent runtime."""

    status: AgentLifecycleStatus
    reason: str | None = None
    agentic_ready: bool = False
    llm_enabled: bool = False
    provider: str | None = None
    model: str | None = None
    session_id: str = ""
    session_message_count: int = 0
    turn_count: int = 0
    latest_failure_class: str = "none"
    failed_tools: list[str] = Field(default_factory=list)
    has_compaction: bool = False
    open_loops: list[str] = Field(default_factory=list)
    mcp_total_servers: int = 0
    mcp_connected_servers: int = 0
    mcp_failed_servers: list[str] = Field(default_factory=list)
    degraded_sources: list[str] = Field(default_factory=list)
    current_task: str | None = None


def session_state_snapshot(session: Session) -> SessionStateSnapshot:
    """Derive a stable session state from the persisted transcript."""
    latest_turn = session.latest_turn_summary()
    latest_turn_payload = None
    if latest_turn:
        latest_turn_payload = {
            "timestamp": latest_turn.timestamp.isoformat(),
            "stop_reason": latest_turn.stop_reason,
            "failure_class": latest_turn.failure_class,
            "rounds": latest_turn.rounds,
            "tool_count": latest_turn.tool_count,
            "failed_tools": latest_turn.failed_tools,
        }

    open_loops = session.compaction.open_loops if session.compaction else []
    if latest_turn and latest_turn.failure_class != "none":
        return SessionStateSnapshot(
            status=SessionLifecycleStatus.ATTENTION_NEEDED,
            reason=f"Latest turn ended with {latest_turn.failure_class}.",
            latest_failure_class=latest_turn.failure_class,
            has_compaction=session.compaction is not None,
            open_loops=open_loops,
            latest_turn=latest_turn_payload,
        )
    if session.compaction:
        return SessionStateSnapshot(
            status=SessionLifecycleStatus.COMPACTED,
            reason="Older context has been compacted into a reusable summary.",
            latest_failure_class="none",
            has_compaction=True,
            open_loops=open_loops,
            latest_turn=latest_turn_payload,
        )
    return SessionStateSnapshot(
        status=SessionLifecycleStatus.ACTIVE,
        reason=None,
        latest_failure_class="none",
        has_compaction=False,
        open_loops=[],
        latest_turn=latest_turn_payload,
    )
