"""Session persistence — save/resume PKM Agent conversations.

Inspired by claw-code's session management. Sessions are stored as
JSON files under the store directory, enabling conversation continuity.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation session."""
    role: str  # user, assistant, system, tool_call, tool_result
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)


class TurnSummary(BaseModel):
    """Machine-readable result of one agent turn."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stop_reason: str = "end_turn"
    failure_class: str = "none"
    rounds: int = 1
    tool_count: int = 0
    failed_tools: list[str] = Field(default_factory=list)
    user_input_preview: str = ""
    assistant_output_preview: str = ""

    def as_prompt_line(self) -> str:
        failed = ", ".join(self.failed_tools) if self.failed_tools else "none"
        return (
            f"- {self.timestamp.isoformat()} | stop={self.stop_reason} | "
            f"failure={self.failure_class} | rounds={self.rounds} | "
            f"tools={self.tool_count} | failed_tools={failed} | "
            f"user={self.user_input_preview} | assistant={self.assistant_output_preview}"
        )


class SessionCompaction(BaseModel):
    """Compressed summary of older session context."""
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    summary: str = ""
    open_loops: list[str] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)
    covered_messages: int = 0

    def as_prompt(self) -> str:
        lines = [
            "[Session Summary]",
            f"Summary: {self.summary or 'No summary available.'}",
        ]
        if self.key_concepts:
            lines.append(f"Key concepts: {', '.join(self.key_concepts)}")
        if self.open_loops:
            lines.append(f"Open loops: {'; '.join(self.open_loops)}")
        lines.append(f"Covered messages: {self.covered_messages}")
        return "\n".join(lines)


class ProjectionFeedbackEntry(BaseModel):
    """Feedback on a projected concept view."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    query: str = ""
    useful: bool | None = None
    missing_concepts: list[str] = Field(default_factory=list)
    noisy_concepts: list[str] = Field(default_factory=list)
    weak_relations: list[str] = Field(default_factory=list)
    notes: str = ""
    concept_names: list[str] = Field(default_factory=list)
    relation_labels: list[str] = Field(default_factory=list)


class Session(BaseModel):
    """A persistent conversation session."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    title: str = ""
    messages: list[Message] = Field(default_factory=list)
    turn_summaries: list[TurnSummary] = Field(default_factory=list)
    compaction: SessionCompaction | None = None
    projection_feedback: list[ProjectionFeedbackEntry] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def add_message(self, role: str, content: str, **meta) -> Message:
        msg = Message(role=role, content=content, metadata=meta)
        self.messages.append(msg)
        self.updated_at = datetime.now(timezone.utc)
        return msg

    def message_count(self) -> int:
        return len(self.messages)

    def last_message(self) -> Message | None:
        return self.messages[-1] if self.messages else None

    def add_turn_summary(self, summary: TurnSummary) -> TurnSummary:
        self.turn_summaries.append(summary)
        self.updated_at = datetime.now(timezone.utc)
        return summary

    def latest_turn_summary(self) -> TurnSummary | None:
        return self.turn_summaries[-1] if self.turn_summaries else None

    def set_compaction(self, compaction: SessionCompaction) -> SessionCompaction:
        self.compaction = compaction
        self.updated_at = datetime.now(timezone.utc)
        return compaction

    def add_projection_feedback(
        self,
        feedback: ProjectionFeedbackEntry,
    ) -> ProjectionFeedbackEntry:
        self.projection_feedback.append(feedback)
        self.updated_at = datetime.now(timezone.utc)
        return feedback

    def latest_projection_feedback(self) -> ProjectionFeedbackEntry | None:
        return self.projection_feedback[-1] if self.projection_feedback else None

    def needs_compaction(
        self,
        *,
        threshold: int = 48,
        preserve_recent: int = 16,
        min_new_messages: int = 12,
    ) -> bool:
        if len(self.messages) <= threshold:
            return False
        covered = self.compaction.covered_messages if self.compaction else 0
        older_messages = max(0, len(self.messages) - preserve_recent)
        return older_messages - covered >= min_new_messages

    def to_llm_messages(
        self,
        max_messages: int = 50,
        *,
        preserve_recent: int = 16,
        include_turn_summaries: bool = True,
    ) -> list[dict]:
        """Convert to LLM-compatible message format (last N messages)."""
        budget = max_messages
        compaction = self.compaction

        use_compaction = compaction is not None and len(self.messages) > preserve_recent
        if use_compaction and budget > 0:
            budget -= 1

        if include_turn_summaries and self.turn_summaries and budget > 0:
            budget -= 1

        budget = max(1, budget)

        recent_window = min(len(self.messages), preserve_recent, budget)
        if use_compaction:
            recent = self.messages[-recent_window:]
        else:
            recent = self.messages[-budget:]

        result = []
        if use_compaction:
            result.append({
                "role": "system",
                "content": compaction.as_prompt(),
            })
        if include_turn_summaries and self.turn_summaries:
            recent_turns = self.turn_summaries[-3:]
            result.append({
                "role": "system",
                "content": "[Recent Turn State]\n" + "\n".join(
                    item.as_prompt_line() for item in recent_turns
                ),
            })
        for msg in recent:
            if msg.role in ("user", "assistant", "system"):
                result.append({"role": msg.role, "content": msg.content})
            elif msg.role == "tool_result":
                # Pack tool results into assistant context
                result.append({"role": "assistant", "content": f"[Tool Result] {msg.content}"})
        return result

    def summary(self) -> str:
        """One-line summary for session listing."""
        title = self.title or "Untitled"
        age = datetime.now(timezone.utc) - self.updated_at
        hours = age.total_seconds() / 3600
        if hours < 1:
            ago = f"{int(age.total_seconds() / 60)}m ago"
        elif hours < 24:
            ago = f"{int(hours)}h ago"
        else:
            ago = f"{int(hours / 24)}d ago"
        return f"[{self.id}] {title} ({self.message_count()} msgs, {ago})"


class SessionStore:
    """Persistent storage for conversation sessions.

    Sessions are saved as individual JSON files under `{store_path}/sessions/`.
    """

    def __init__(self, store_path: str | Path) -> None:
        self._path = Path(store_path).expanduser() / "sessions"
        self._path.mkdir(parents=True, exist_ok=True)

    def save(self, session: Session) -> Path:
        fp = self._path / f"{session.id}.json"
        fp.write_text(session.model_dump_json(indent=2), encoding="utf-8")
        return fp

    def load(self, session_id: str) -> Session | None:
        fp = self._path / f"{session_id}.json"
        if not fp.exists():
            return None
        try:
            return Session.model_validate_json(fp.read_text(encoding="utf-8"))
        except Exception:
            return None

    def load_latest(self) -> Session | None:
        files = sorted(self._path.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not files:
            return None
        try:
            return Session.model_validate_json(files[0].read_text(encoding="utf-8"))
        except Exception:
            return None

    def list_sessions(self, limit: int = 20) -> list[Session]:
        files = sorted(self._path.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        sessions = []
        for fp in files[:limit]:
            try:
                s = Session.model_validate_json(fp.read_text(encoding="utf-8"))
                sessions.append(s)
            except Exception:
                continue
        return sessions

    def delete(self, session_id: str) -> bool:
        fp = self._path / f"{session_id}.json"
        if fp.exists():
            fp.unlink()
            return True
        return False
