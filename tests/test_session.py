"""Tests for session persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from world0.agents.session import (
    Message,
    ProjectionFeedbackEntry,
    Session,
    SessionCompaction,
    SessionStore,
    TurnSummary,
)
from world0.agents.state import session_state_snapshot


class TestSession:
    def test_create_session(self):
        s = Session()
        assert s.id
        assert s.message_count() == 0

    def test_add_message(self):
        s = Session()
        msg = s.add_message("user", "Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert s.message_count() == 1

    def test_last_message(self):
        s = Session()
        assert s.last_message() is None
        s.add_message("user", "First")
        s.add_message("assistant", "Second")
        assert s.last_message().content == "Second"

    def test_to_llm_messages(self):
        s = Session()
        s.add_message("user", "Hi")
        s.add_message("assistant", "Hello!")
        s.add_message("tool_call", '{"name": "search"}')
        s.add_message("tool_result", "Found 3 results")
        msgs = s.to_llm_messages()
        assert len(msgs) == 3  # user, assistant, tool_result as assistant
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_to_llm_messages_max(self):
        s = Session()
        for i in range(10):
            s.add_message("user", f"msg {i}")
        msgs = s.to_llm_messages(max_messages=3)
        assert len(msgs) == 3

    def test_summary(self):
        s = Session(title="Test Session")
        s.add_message("user", "Hi")
        summary = s.summary()
        assert "Test Session" in summary
        assert "1 msgs" in summary

    def test_turn_summary_helpers(self):
        s = Session()
        s.add_turn_summary(TurnSummary(
            stop_reason="end_turn",
            failure_class="none",
            rounds=2,
            tool_count=1,
            user_input_preview="Research MCP",
            assistant_output_preview="Summarized MCP flow",
        ))
        latest = s.latest_turn_summary()
        assert latest is not None
        assert latest.tool_count == 1
        assert "Research MCP" in latest.as_prompt_line()

    def test_to_llm_messages_includes_compaction_and_turn_state(self):
        s = Session()
        for i in range(30):
            s.add_message("user" if i % 2 == 0 else "assistant", f"message {i}")
        s.set_compaction(SessionCompaction(
            summary="Earlier discussion covered MCP integration and session resume bugs.",
            open_loops=["Confirm the MCP server contract."],
            key_concepts=["mcp", "resume session"],
            covered_messages=12,
        ))
        s.add_turn_summary(TurnSummary(
            stop_reason="end_turn",
            failure_class="tool_runtime",
            rounds=3,
            tool_count=4,
            failed_tools=["web_fetch"],
            user_input_preview="Research Claude Code MCP",
            assistant_output_preview="Produced a short research brief.",
        ))

        msgs = s.to_llm_messages(max_messages=8, preserve_recent=6)
        assert msgs[0]["role"] == "system"
        assert "Session Summary" in msgs[0]["content"]
        assert msgs[1]["role"] == "system"
        assert "Recent Turn State" in msgs[1]["content"]
        assert len(msgs) >= 3

    def test_session_state_marks_attention_needed_after_failure(self):
        s = Session()
        s.add_turn_summary(TurnSummary(
            stop_reason="end_turn",
            failure_class="tool_runtime",
            rounds=2,
            tool_count=1,
            failed_tools=["web_fetch"],
            user_input_preview="Fetch a source",
            assistant_output_preview="The fetch failed.",
        ))
        state = session_state_snapshot(s)
        assert state.status.value == "attention_needed"
        assert state.latest_failure_class == "tool_runtime"

    def test_session_state_marks_compacted_without_failure(self):
        s = Session()
        s.set_compaction(SessionCompaction(
            summary="Older context summarized.",
            open_loops=["Validate the source list."],
            key_concepts=["research"],
            covered_messages=10,
        ))
        state = session_state_snapshot(s)
        assert state.status.value == "compacted"
        assert state.has_compaction is True

    def test_projection_feedback_helpers(self):
        s = Session()
        entry = s.add_projection_feedback(ProjectionFeedbackEntry(
            query="fastapi backend",
            useful=True,
            missing_concepts=["postgresql"],
        ))
        assert entry.query == "fastapi backend"
        assert s.latest_projection_feedback() is not None
        assert s.latest_projection_feedback().useful is True


class TestSessionStore:
    @pytest.fixture
    def store(self, tmp_path: Path) -> SessionStore:
        return SessionStore(tmp_path / "sessions_test")

    def test_save_and_load(self, store: SessionStore):
        s = Session(title="Test")
        s.add_message("user", "Hello")
        store.save(s)

        loaded = store.load(s.id)
        assert loaded is not None
        assert loaded.id == s.id
        assert loaded.title == "Test"
        assert loaded.message_count() == 1

    def test_load_nonexistent(self, store: SessionStore):
        assert store.load("nonexistent") is None

    def test_load_latest(self, store: SessionStore):
        s1 = Session(title="First")
        store.save(s1)
        s2 = Session(title="Second")
        store.save(s2)

        latest = store.load_latest()
        assert latest is not None
        assert latest.id == s2.id

    def test_list_sessions(self, store: SessionStore):
        for i in range(5):
            s = Session(title=f"Session {i}")
            store.save(s)

        sessions = store.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_delete(self, store: SessionStore):
        s = Session(title="Delete me")
        store.save(s)
        assert store.load(s.id) is not None

        assert store.delete(s.id)
        assert store.load(s.id) is None

    def test_delete_nonexistent(self, store: SessionStore):
        assert not store.delete("nonexistent")
