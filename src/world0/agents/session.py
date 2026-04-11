"""Session persistence — save/resume PKM Agent conversations.

Inspired by claw-code's session management. Sessions are stored as
JSON files under the store directory, enabling conversation continuity.
"""

from __future__ import annotations

import json
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


class Session(BaseModel):
    """A persistent conversation session."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    title: str = ""
    messages: list[Message] = Field(default_factory=list)
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

    def to_llm_messages(self, max_messages: int = 50) -> list[dict]:
        """Convert to LLM-compatible message format (last N messages)."""
        recent = self.messages[-max_messages:]
        result = []
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
