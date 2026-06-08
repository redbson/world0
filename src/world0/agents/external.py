"""External agent integrations for system-installed Claude Code and Codex."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


class ExternalAgentError(RuntimeError):
    """Raised when an external agent command fails."""


@dataclass
class ProblemWorkspace:
    """A per-problem isolated workspace for external agent interaction."""

    problem_id: str
    workspace: Path
    problem_file: Path
    context_file: Path
    transcript_file: Path
    metadata_file: Path


def available_external_agents() -> dict[str, str]:
    """Return available external agent binaries keyed by logical name."""
    found: dict[str, str] = {}
    for name, binary in (("claude", "claude"), ("codex", "codex")):
        resolved = shutil.which(binary)
        if resolved:
            found[name] = resolved
    return found


def prepare_problem_workspace(
    root: str | Path,
    *,
    agent: str,
    problem: str,
) -> ProblemWorkspace:
    """Create or reuse a stable per-problem workspace."""
    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    slug = _slugify(problem)
    digest = hashlib.sha1(problem.strip().lower().encode("utf-8")).hexdigest()[:10]
    problem_id = f"{slug}-{digest}"
    workspace = root_path / agent.strip().lower() / problem_id
    workspace.mkdir(parents=True, exist_ok=True)

    return ProblemWorkspace(
        problem_id=problem_id,
        workspace=workspace,
        problem_file=workspace / "problem.md",
        context_file=workspace / "world0_context.md",
        transcript_file=workspace / "interaction_log.md",
        metadata_file=workspace / "metadata.json",
    )


def write_problem_materials(
    problem_workspace: ProblemWorkspace,
    *,
    agent: str,
    problem: str,
    prompt: str,
    rendered_context: str,
    source_workspace: str | Path,
    session_id: str = "",
) -> None:
    """Persist the request and context into the isolated workspace."""
    now = datetime.now(timezone.utc).isoformat()
    source_path = str(Path(source_workspace).expanduser().resolve())
    metadata = {
        "problem_id": problem_workspace.problem_id,
        "agent": agent,
        "problem": problem,
        "source_workspace": source_path,
        "session_id": session_id,
        "updated_at": now,
    }
    if problem_workspace.metadata_file.exists():
        try:
            existing = json.loads(problem_workspace.metadata_file.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                metadata = {**existing, **metadata}
        except Exception:
            pass

    problem_workspace.problem_file.write_text(
        "\n".join([
            f"# Problem {problem_workspace.problem_id}",
            "",
            f"Agent: {agent}",
            f"Source workspace: {source_path}",
            f"Session: {session_id or 'n/a'}",
            "",
            "## Problem",
            problem.strip(),
            "",
            "## Prompt",
            prompt.strip(),
            "",
        ]),
        encoding="utf-8",
    )
    problem_workspace.context_file.write_text(
        rendered_context.strip() or "No World 0 context was available for this problem.",
        encoding="utf-8",
    )
    problem_workspace.metadata_file.write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def append_problem_interaction(
    problem_workspace: ProblemWorkspace,
    *,
    role: str,
    content: str,
) -> None:
    """Append one interaction turn to the problem transcript."""
    timestamp = datetime.now(timezone.utc).isoformat()
    with problem_workspace.transcript_file.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n".join([
                f"## {timestamp} [{role}]",
                content.strip(),
                "",
            ])
        )


def _slugify(text: str, *, limit: int = 48) -> str:
    compact = re.sub(r"\s+", "-", text.strip().lower())
    compact = re.sub(r"[^a-z0-9\u4e00-\u9fff_-]+", "-", compact)
    compact = compact.strip("-_")
    if not compact:
        return "problem"
    return compact[:limit].rstrip("-_") or "problem"


def run_external_agent(
    agent: str,
    prompt: str,
    *,
    workspace: str | Path = ".",
    model: str = "",
    timeout_seconds: int = 180,
) -> str:
    """Run an external assistant in read-only consultation mode."""
    clean_agent = agent.strip().lower()
    workspace_path = Path(workspace).expanduser().resolve()
    if not workspace_path.exists():
        raise ExternalAgentError(f"Workspace does not exist: {workspace_path}")

    binary = available_external_agents().get(clean_agent)
    if not binary:
        raise ExternalAgentError(
            f"External agent '{clean_agent}' is not installed or not in PATH."
        )

    if clean_agent == "claude":
        return _run_claude(
            binary,
            prompt,
            workspace=workspace_path,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    if clean_agent == "codex":
        return _run_codex(
            binary,
            prompt,
            workspace=workspace_path,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    raise ExternalAgentError(f"Unsupported external agent: {clean_agent}")


def _run_claude(
    binary: str,
    prompt: str,
    *,
    workspace: Path,
    model: str,
    timeout_seconds: int,
) -> str:
    command = [
        binary,
        "--print",
        "--output-format",
        "text",
        "--permission-mode",
        "plan",
        "--add-dir",
        str(workspace),
        "--append-system-prompt",
        (
            "You are being consulted by World 0 PKM in read-only mode. "
            "Do not edit files, apply patches, or claim to have changed code."
        ),
        prompt,
    ]
    if model:
        command[1:1] = ["--model", model]
    result = _run_process(command, cwd=workspace, timeout_seconds=timeout_seconds)
    output = (result.stdout or "").strip()
    if not output:
        raise ExternalAgentError("Claude returned no output.")
    return output


def _run_codex(
    binary: str,
    prompt: str,
    *,
    workspace: Path,
    model: str,
    timeout_seconds: int,
) -> str:
    output_path = ""
    with tempfile.NamedTemporaryFile(prefix="world0-codex-", suffix=".txt", delete=False) as handle:
        output_path = handle.name

    command = [
        binary,
        "exec",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "--cd",
        str(workspace),
        "--output-last-message",
        output_path,
        prompt,
    ]
    if model:
        command[2:2] = ["--model", model]

    try:
        result = _run_process(command, cwd=workspace, timeout_seconds=timeout_seconds)
        rendered = Path(output_path).read_text(encoding="utf-8").strip()
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass

    if rendered:
        return rendered

    fallback = (result.stdout or "").strip()
    if fallback:
        return fallback
    raise ExternalAgentError("Codex returned no output.")


def _run_process(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExternalAgentError(
            f"External agent timed out after {timeout_seconds}s."
        ) from exc
    except OSError as exc:
        raise ExternalAgentError(f"Failed to execute external agent: {exc}") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        detail = detail[:600] if detail else "No diagnostic output."
        raise ExternalAgentError(detail)
    return result
