"""Deep, deterministic tests for external-agent consultations.

Hermetic: no real subprocess, network, or binaries. ``shutil.which`` and
``subprocess.run`` are monkeypatched; all filesystem state lives under
``tmp_path`` / the PKM store directory.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from world0.agents import external as ext
from world0.agents.external import (
    ExternalAgentError,
    ProblemWorkspace,
    _slugify,
    append_problem_interaction,
    available_external_agents,
    prepare_problem_workspace,
    run_external_agent,
    write_problem_materials,
)
from world0.agents.pkm import PKMAgent
from world0.schemas.types import Observation


# ── Fixtures (mirroring tests/test_pkm_agent.py) ──────────────────────────


class FakeLLM:
    """Fake LLM returning predictable JSON for seed/extraction prompts."""

    def complete_json(self, system: str, user: str) -> str:
        if '"seeds"' in system:
            return json.dumps({"seeds": ["python", "fastapi"]})
        return json.dumps({
            "concepts": [
                {"name": "python", "description": "programming language"},
                {"name": "fastapi", "description": "web framework"},
            ],
            "relations": [
                {"source": "fastapi", "target": "python", "type": "depends_on"},
            ],
        })


@pytest.fixture
def tmp_store(tmp_path: Path) -> Path:
    return tmp_path / "test_pkm"


@pytest.fixture
def agent(tmp_store: Path) -> PKMAgent:
    return PKMAgent(store_path=tmp_store, llm=None)


@pytest.fixture
def agent_with_llm(tmp_store: Path) -> PKMAgent:
    return PKMAgent(store_path=tmp_store, llm=FakeLLM())


def _which(present: set[str]):
    """Return a fake shutil.which that resolves only the given binaries."""

    def fake_which(binary: str) -> str | None:
        return f"/usr/bin/{binary}" if binary in present else None

    return fake_which


def _make_run(captured: list[list[str]], *, stdout: str = "", stderr: str = "",
              returncode: int = 0, raise_exc: Exception | None = None):
    """Build a fake subprocess.run capturing argv into ``captured``."""

    def fake_run(command, **kwargs):  # noqa: ANN001
        captured.append(list(command))
        if raise_exc is not None:
            raise raise_exc
        return subprocess.CompletedProcess(
            args=command, returncode=returncode, stdout=stdout, stderr=stderr
        )

    return fake_run


# ── available_external_agents() ───────────────────────────────────────────


class TestAvailableExternalAgents:
    def test_both_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ext.shutil, "which", _which({"claude", "codex"}))
        found = available_external_agents()
        assert found == {"claude": "/usr/bin/claude", "codex": "/usr/bin/codex"}

    def test_one_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ext.shutil, "which", _which({"codex"}))
        found = available_external_agents()
        assert found == {"codex": "/usr/bin/codex"}
        assert "claude" not in found

    def test_none_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ext.shutil, "which", _which(set()))
        assert available_external_agents() == {}


# ── _slugify() ─────────────────────────────────────────────────────────────


class TestSlugify:
    def test_whitespace_to_dash_and_lowercase(self) -> None:
        assert _slugify("Hello World") == "hello-world"

    def test_punctuation_stripped(self) -> None:
        # Punctuation collapses to dashes which then get trimmed at edges.
        assert _slugify("foo!!!bar???") == "foo-bar"

    def test_cjk_preserved(self) -> None:
        # CJK ideographs are inside the allowed unicode range.
        assert _slugify("中文 概念") == "中文-概念"

    def test_underscore_and_hyphen_kept(self) -> None:
        assert _slugify("a_b-c") == "a_b-c"

    def test_empty_returns_problem(self) -> None:
        assert _slugify("") == "problem"
        assert _slugify("   ") == "problem"

    def test_only_punctuation_returns_problem(self) -> None:
        assert _slugify("!!!") == "problem"

    def test_length_limit_and_trailing_dash_trim(self) -> None:
        # 50 'a's with a dash at position 48 region; limited to 48 then
        # trailing separators trimmed.
        text = "a" * 47 + " " + "b" * 10
        slug = _slugify(text)
        assert len(slug) <= 48
        assert not slug.endswith("-")
        assert not slug.endswith("_")

    def test_trailing_dash_trimmed_after_limit(self) -> None:
        # Construct so that char 48 is a separator -> rstrip removes it.
        text = "x" * 47 + "  more"
        slug = _slugify(text)
        assert slug == "x" * 47


# ── prepare_problem_workspace() ───────────────────────────────────────────


class TestPrepareProblemWorkspace:
    def test_problem_id_is_slug_plus_digest(self, tmp_path: Path) -> None:
        ws = prepare_problem_workspace(tmp_path, agent="claude", problem="Hello World")
        slug, _, digest = ws.problem_id.rpartition("-")
        assert slug == "hello-world"
        assert len(digest) == 10
        assert all(c in "0123456789abcdef" for c in digest)

    def test_reuse_same_problem(self, tmp_path: Path) -> None:
        a = prepare_problem_workspace(tmp_path, agent="claude", problem="shared problem")
        b = prepare_problem_workspace(tmp_path, agent="claude", problem="shared problem")
        assert a.problem_id == b.problem_id
        assert a.workspace == b.workspace

    def test_digest_normalizes_case_and_whitespace(self, tmp_path: Path) -> None:
        a = prepare_problem_workspace(tmp_path, agent="claude", problem="Shared Problem")
        b = prepare_problem_workspace(tmp_path, agent="claude", problem="  shared problem  ")
        # digest is computed on problem.strip().lower(); slug differs only by
        # case-folding too, so the whole id matches.
        assert a.problem_id == b.problem_id

    def test_distinct_problems(self, tmp_path: Path) -> None:
        a = prepare_problem_workspace(tmp_path, agent="claude", problem="problem one")
        b = prepare_problem_workspace(tmp_path, agent="claude", problem="problem two")
        assert a.problem_id != b.problem_id
        assert a.workspace != b.workspace

    def test_per_agent_subdir(self, tmp_path: Path) -> None:
        c = prepare_problem_workspace(tmp_path, agent="claude", problem="same")
        x = prepare_problem_workspace(tmp_path, agent="codex", problem="same")
        assert c.workspace.parent.name == "claude"
        assert x.workspace.parent.name == "codex"
        assert c.problem_id == x.problem_id  # id is agent-independent
        assert c.workspace != x.workspace

    def test_file_path_fields(self, tmp_path: Path) -> None:
        ws = prepare_problem_workspace(tmp_path, agent="claude", problem="p")
        assert isinstance(ws, ProblemWorkspace)
        assert ws.workspace.is_dir()
        assert ws.problem_file == ws.workspace / "problem.md"
        assert ws.context_file == ws.workspace / "world0_context.md"
        assert ws.transcript_file == ws.workspace / "interaction_log.md"
        assert ws.metadata_file == ws.workspace / "metadata.json"


# ── write_problem_materials() ─────────────────────────────────────────────


class TestWriteProblemMaterials:
    def test_writes_all_files(self, tmp_path: Path) -> None:
        ws = prepare_problem_workspace(tmp_path / "root", agent="codex", problem="design api")
        src = tmp_path / "src"
        src.mkdir()
        write_problem_materials(
            ws,
            agent="codex",
            problem="design api",
            prompt="Inspect the layout",
            rendered_context="Some cognitive context.",
            source_workspace=src,
            session_id="sess-1",
        )
        problem_text = ws.problem_file.read_text(encoding="utf-8")
        assert f"# Problem {ws.problem_id}" in problem_text
        assert "Agent: codex" in problem_text
        assert "Session: sess-1" in problem_text
        assert "## Problem" in problem_text
        assert "design api" in problem_text
        assert "## Prompt" in problem_text
        assert "Inspect the layout" in problem_text

        assert ws.context_file.read_text(encoding="utf-8") == "Some cognitive context."

        meta = json.loads(ws.metadata_file.read_text(encoding="utf-8"))
        assert meta["problem_id"] == ws.problem_id
        assert meta["agent"] == "codex"
        assert meta["problem"] == "design api"
        assert meta["session_id"] == "sess-1"
        assert meta["source_workspace"] == str(src.resolve())
        assert "updated_at" in meta

    def test_empty_context_fallback(self, tmp_path: Path) -> None:
        ws = prepare_problem_workspace(tmp_path / "root", agent="claude", problem="p")
        write_problem_materials(
            ws,
            agent="claude",
            problem="p",
            prompt="q",
            rendered_context="   ",
            source_workspace=tmp_path,
        )
        assert ws.context_file.read_text(encoding="utf-8") == (
            "No World 0 context was available for this problem."
        )

    def test_session_fallback_in_problem_file(self, tmp_path: Path) -> None:
        ws = prepare_problem_workspace(tmp_path / "root", agent="claude", problem="p")
        write_problem_materials(
            ws,
            agent="claude",
            problem="p",
            prompt="q",
            rendered_context="",
            source_workspace=tmp_path,
            session_id="",
        )
        assert "Session: n/a" in ws.problem_file.read_text(encoding="utf-8")

    def test_metadata_merge_preserves_preexisting_keys(self, tmp_path: Path) -> None:
        ws = prepare_problem_workspace(tmp_path / "root", agent="claude", problem="p")
        ws.metadata_file.write_text(
            json.dumps({"custom": "keep-me", "agent": "stale"}),
            encoding="utf-8",
        )
        write_problem_materials(
            ws,
            agent="claude",
            problem="p",
            prompt="q",
            rendered_context="ctx",
            source_workspace=tmp_path,
            session_id="s",
        )
        meta = json.loads(ws.metadata_file.read_text(encoding="utf-8"))
        # Pre-existing custom key survives, but new values override stale ones.
        assert meta["custom"] == "keep-me"
        assert meta["agent"] == "claude"

    def test_metadata_merge_ignores_corrupt_file(self, tmp_path: Path) -> None:
        ws = prepare_problem_workspace(tmp_path / "root", agent="claude", problem="p")
        ws.metadata_file.write_text("{not valid json", encoding="utf-8")
        write_problem_materials(
            ws,
            agent="claude",
            problem="p",
            prompt="q",
            rendered_context="ctx",
            source_workspace=tmp_path,
            session_id="s",
        )
        meta = json.loads(ws.metadata_file.read_text(encoding="utf-8"))
        assert meta["agent"] == "claude"
        assert "custom" not in meta


# ── append_problem_interaction() ──────────────────────────────────────────


class TestAppendProblemInteraction:
    def test_appends_in_order_with_role_headers(self, tmp_path: Path) -> None:
        ws = prepare_problem_workspace(tmp_path / "root", agent="claude", problem="p")
        append_problem_interaction(ws, role="pkm", content="first turn")
        append_problem_interaction(ws, role="claude", content="second turn")
        text = ws.transcript_file.read_text(encoding="utf-8")

        assert "[pkm]" in text
        assert "[claude]" in text
        assert "first turn" in text
        assert "second turn" in text
        # Ordering preserved.
        assert text.index("first turn") < text.index("second turn")
        assert text.index("[pkm]") < text.index("[claude]")

    def test_content_is_stripped(self, tmp_path: Path) -> None:
        ws = prepare_problem_workspace(tmp_path / "root", agent="claude", problem="p")
        append_problem_interaction(ws, role="pkm", content="  padded  ")
        text = ws.transcript_file.read_text(encoding="utf-8")
        assert "padded" in text
        # Each entry block has a header then stripped content.
        assert text.count("[pkm]") == 1


# ── run_external_agent() dispatch ─────────────────────────────────────────


class TestRunExternalAgentDispatch:
    def test_missing_binary_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ext.shutil, "which", _which(set()))
        with pytest.raises(ExternalAgentError, match="not installed"):
            run_external_agent("claude", "hi", workspace=tmp_path)

    def test_missing_workspace_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ext.shutil, "which", _which({"claude"}))
        with pytest.raises(ExternalAgentError, match="Workspace does not exist"):
            run_external_agent("claude", "hi", workspace=tmp_path / "missing")

    def test_unsupported_agent_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # available reports it, but dispatch has no branch for it.
        monkeypatch.setattr(
            ext, "available_external_agents", lambda: {"weird": "/usr/bin/weird"}
        )
        with pytest.raises(ExternalAgentError, match="Unsupported"):
            run_external_agent("weird", "hi", workspace=tmp_path)

    def test_claude_builds_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ext.shutil, "which", _which({"claude"}))
        captured: list[list[str]] = []
        monkeypatch.setattr(
            ext.subprocess, "run", _make_run(captured, stdout="claude says hi")
        )
        out = run_external_agent("claude", "PROMPT", workspace=tmp_path)
        assert out == "claude says hi"
        cmd = captured[0]
        assert cmd[0] == "/usr/bin/claude"
        assert "--print" in cmd
        assert "--permission-mode" in cmd
        assert "plan" in cmd
        assert "--add-dir" in cmd
        assert str(tmp_path.resolve()) in cmd
        assert cmd[-1] == "PROMPT"
        assert "--model" not in cmd  # no model passed

    def test_claude_with_model_inserts_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ext.shutil, "which", _which({"claude"}))
        captured: list[list[str]] = []
        monkeypatch.setattr(ext.subprocess, "run", _make_run(captured, stdout="ok"))
        run_external_agent("claude", "P", workspace=tmp_path, model="sonnet")
        cmd = captured[0]
        assert cmd[1] == "--model"
        assert cmd[2] == "sonnet"

    def test_claude_empty_output_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ext.shutil, "which", _which({"claude"}))
        captured: list[list[str]] = []
        monkeypatch.setattr(ext.subprocess, "run", _make_run(captured, stdout="   "))
        with pytest.raises(ExternalAgentError, match="Claude returned no output"):
            run_external_agent("claude", "P", workspace=tmp_path)

    def test_codex_builds_command_and_uses_output_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ext.shutil, "which", _which({"codex"}))
        captured: list[list[str]] = []

        def fake_run(command, **kwargs):  # noqa: ANN001
            captured.append(list(command))
            # codex writes its last message to the --output-last-message path.
            idx = command.index("--output-last-message")
            Path(command[idx + 1]).write_text("codex file answer", encoding="utf-8")
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="ignored stdout", stderr=""
            )

        monkeypatch.setattr(ext.subprocess, "run", fake_run)
        out = run_external_agent("codex", "PROMPT", workspace=tmp_path)
        assert out == "codex file answer"
        cmd = captured[0]
        assert cmd[0] == "/usr/bin/codex"
        assert cmd[1] == "exec"
        assert "--sandbox" in cmd
        assert "read-only" in cmd
        assert "--cd" in cmd
        assert cmd[-1] == "PROMPT"

    def test_codex_with_model_inserts_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ext.shutil, "which", _which({"codex"}))
        captured: list[list[str]] = []

        def fake_run(command, **kwargs):  # noqa: ANN001
            captured.append(list(command))
            idx = command.index("--output-last-message")
            Path(command[idx + 1]).write_text("ans", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "x", "")

        monkeypatch.setattr(ext.subprocess, "run", fake_run)
        run_external_agent("codex", "P", workspace=tmp_path, model="o4")
        cmd = captured[0]
        assert cmd[2] == "--model"
        assert cmd[3] == "o4"

    def test_codex_falls_back_to_stdout_when_file_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ext.shutil, "which", _which({"codex"}))

        def fake_run(command, **kwargs):  # noqa: ANN001
            idx = command.index("--output-last-message")
            Path(command[idx + 1]).write_text("   ", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "stdout fallback", "")

        monkeypatch.setattr(ext.subprocess, "run", fake_run)
        out = run_external_agent("codex", "P", workspace=tmp_path)
        assert out == "stdout fallback"

    def test_codex_empty_everywhere_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ext.shutil, "which", _which({"codex"}))

        def fake_run(command, **kwargs):  # noqa: ANN001
            idx = command.index("--output-last-message")
            Path(command[idx + 1]).write_text("", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        monkeypatch.setattr(ext.subprocess, "run", fake_run)
        with pytest.raises(ExternalAgentError, match="Codex returned no output"):
            run_external_agent("codex", "P", workspace=tmp_path)


# ── run_external_agent() error propagation via _run_process ───────────────


class TestRunProcessErrors:
    def test_nonzero_return_uses_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ext.shutil, "which", _which({"claude"}))
        captured: list[list[str]] = []
        monkeypatch.setattr(
            ext.subprocess,
            "run",
            _make_run(captured, returncode=2, stderr="boom from stderr"),
        )
        with pytest.raises(ExternalAgentError, match="boom from stderr"):
            run_external_agent("claude", "P", workspace=tmp_path)

    def test_nonzero_return_falls_back_to_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ext.shutil, "which", _which({"claude"}))
        captured: list[list[str]] = []
        monkeypatch.setattr(
            ext.subprocess,
            "run",
            _make_run(captured, returncode=1, stderr="", stdout="stdout detail"),
        )
        with pytest.raises(ExternalAgentError, match="stdout detail"):
            run_external_agent("claude", "P", workspace=tmp_path)

    def test_nonzero_no_diagnostic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ext.shutil, "which", _which({"claude"}))
        captured: list[list[str]] = []
        monkeypatch.setattr(
            ext.subprocess,
            "run",
            _make_run(captured, returncode=1, stderr="", stdout=""),
        )
        with pytest.raises(ExternalAgentError, match="No diagnostic output"):
            run_external_agent("claude", "P", workspace=tmp_path)

    def test_timeout_raises_external_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ext.shutil, "which", _which({"claude"}))
        captured: list[list[str]] = []
        monkeypatch.setattr(
            ext.subprocess,
            "run",
            _make_run(
                captured,
                raise_exc=subprocess.TimeoutExpired(cmd="claude", timeout=180),
            ),
        )
        with pytest.raises(ExternalAgentError, match="timed out"):
            run_external_agent("claude", "P", workspace=tmp_path)

    def test_oserror_raises_external_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ext.shutil, "which", _which({"claude"}))
        captured: list[list[str]] = []
        monkeypatch.setattr(
            ext.subprocess,
            "run",
            _make_run(captured, raise_exc=OSError("exec format error")),
        )
        with pytest.raises(ExternalAgentError, match="Failed to execute"):
            run_external_agent("claude", "P", workspace=tmp_path)


# ── PKMAgent.consult_external_agent ───────────────────────────────────────


class TestConsultExternalAgent:
    def test_empty_prompt(self, agent: PKMAgent) -> None:
        result = agent.consult_external_agent("claude", "   ")
        assert "provide a prompt" in result.lower()

    def test_unavailable_agent(
        self, agent: PKMAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "world0.agents.pkm.available_external_agents", lambda: {}
        )
        result = agent.consult_external_agent("claude", "inspect this")
        assert "not available" in result.lower()
        assert "none" in result.lower()

    def test_unavailable_lists_known_agents(
        self, agent: PKMAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "world0.agents.pkm.available_external_agents",
            lambda: {"codex": "/usr/bin/codex"},
        )
        result = agent.consult_external_agent("claude", "inspect this")
        assert "not available" in result.lower()
        assert "codex" in result.lower()

    def test_includes_world0_context_when_seeds_resolve(
        self, agent_with_llm: PKMAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obs = Observation(
            concepts=["python", "fastapi"],
            relations=[("fastapi", "python", "depends_on")],
            task="service design",
        )
        agent_with_llm.learn_structured(obs)

        captured: dict[str, object] = {}
        monkeypatch.setattr(
            "world0.agents.pkm.available_external_agents",
            lambda: {"codex": "/usr/bin/codex"},
        )

        def fake_run(agent, prompt, *, workspace=".", model="", timeout_seconds=180):
            captured["prompt"] = prompt
            return "review summary"

        monkeypatch.setattr("world0.agents.pkm.run_external_agent", fake_run)

        result = agent_with_llm.consult_external_agent(
            "codex", "Inspect the fastapi service in python"
        )
        assert "review summary" in result.lower()
        assert "World 0 Cognitive Context" in str(captured["prompt"])

    def test_no_context_when_use_world0_context_false(
        self, agent_with_llm: PKMAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obs = Observation(
            concepts=["python", "fastapi"],
            relations=[("fastapi", "python", "depends_on")],
        )
        agent_with_llm.learn_structured(obs)

        captured: dict[str, object] = {}
        monkeypatch.setattr(
            "world0.agents.pkm.available_external_agents",
            lambda: {"codex": "/usr/bin/codex"},
        )

        def fake_run(agent, prompt, *, workspace=".", model="", timeout_seconds=180):
            captured["prompt"] = prompt
            return "summary"

        monkeypatch.setattr("world0.agents.pkm.run_external_agent", fake_run)

        agent_with_llm.consult_external_agent(
            "codex", "Inspect fastapi", use_world0_context=False
        )
        assert "World 0 Cognitive Context" not in str(captured["prompt"])

    def test_records_last_consultation_metadata(
        self, agent: PKMAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "world0.agents.pkm.available_external_agents",
            lambda: {"claude": "/usr/bin/claude"},
        )
        monkeypatch.setattr(
            "world0.agents.pkm.run_external_agent",
            lambda agent, prompt, **kwargs: "ok reply",
        )
        agent.consult_external_agent(
            "claude", "design prompt", problem="metadata problem", model="m1"
        )
        meta = agent.session.metadata["last_external_consultation"]
        assert meta["agent"] == "claude"
        assert meta["problem"] == "metadata problem"
        assert meta["model"] == "m1"
        assert meta["used_world0_context"] is True
        workspace = Path(meta["workspace"])
        assert workspace.exists()
        assert (workspace / "problem.md").exists()
        assert (workspace / "world0_context.md").exists()
        log = (workspace / "interaction_log.md").read_text(encoding="utf-8")
        assert "ok reply" in log
        assert "[claude]" in log

    def test_error_path_appends_error_turn_and_returns_failure(
        self, agent: PKMAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "world0.agents.pkm.available_external_agents",
            lambda: {"claude": "/usr/bin/claude"},
        )

        def boom(agent, prompt, **kwargs):
            raise ExternalAgentError("kaboom")

        monkeypatch.setattr("world0.agents.pkm.run_external_agent", boom)

        result = agent.consult_external_agent(
            "claude", "trigger failure", problem="err problem"
        )
        assert result.lower().startswith("external agent 'claude' failed")
        assert "kaboom" in result

        root = agent.external_problem_root()
        logs = list(root.rglob("interaction_log.md"))
        assert logs
        text = logs[0].read_text(encoding="utf-8")
        assert "[error]" in text
        assert "kaboom" in text
        # No success metadata recorded on the error path.
        assert "last_external_consultation" not in agent.session.metadata

    def test_workspace_reuse_across_two_calls_same_problem(
        self, agent: PKMAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "world0.agents.pkm.available_external_agents",
            lambda: {"claude": "/usr/bin/claude"},
        )
        monkeypatch.setattr(
            "world0.agents.pkm.run_external_agent",
            lambda agent, prompt, **kwargs: "ok",
        )
        agent.consult_external_agent("claude", "first", problem="shared problem")
        first_ws = Path(agent.session.metadata["last_external_consultation"]["workspace"])
        agent.consult_external_agent("claude", "second", problem="shared problem")
        second_ws = Path(agent.session.metadata["last_external_consultation"]["workspace"])
        assert first_ws == second_ws
        # Transcript accumulates both pkm requests + both replies.
        log = (first_ws / "interaction_log.md").read_text(encoding="utf-8")
        assert log.count("[claude]") == 2

    def test_workspace_under_external_problem_root(
        self, agent: PKMAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "world0.agents.pkm.available_external_agents",
            lambda: {"claude": "/usr/bin/claude"},
        )
        monkeypatch.setattr(
            "world0.agents.pkm.run_external_agent",
            lambda agent, prompt, **kwargs: "ok",
        )
        agent.consult_external_agent("claude", "x", problem="rooted problem")
        ws = Path(agent.session.metadata["last_external_consultation"]["workspace"])
        root = agent.external_problem_root().resolve()
        assert str(ws).startswith(str(root))
        assert ws.parent.name == "claude"
