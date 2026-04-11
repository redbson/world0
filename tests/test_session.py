"""Tests for session persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from world0.agents.session import Message, Session, SessionStore


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
