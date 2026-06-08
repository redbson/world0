"""CLI entry point for the PKM Agent.

Usage::

    # With Anthropic (default)
    python -m world0.agents.cli --provider anthropic

    # With OpenAI
    python -m world0.agents.cli --provider openai --model gpt-4o

    # Custom store path
    python -m world0.agents.cli --store ~/.my_knowledge

    # Quick learn without entering chat
    python -m world0.agents.cli learn "Transformers use self-attention..."

    # Quick ask
    python -m world0.agents.cli ask "How does attention work?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from world0.agents.pkm import PKMAgent
from world0.agents.provider import normalize_provider_name
from world0.llm.base import LLMProvider


def _prompt_config_target(args: argparse.Namespace) -> Path:
    """Return the prompt config file targeted by CLI arguments."""
    from world0.prompts import prompt_config_path
    from world0.spaces import SpaceRegistry

    root = Path(args.store).expanduser()
    if args.space:
        registry = SpaceRegistry(root)
        space = registry.resolve(args.space)
        if space is None:
            print(f"error: space {args.space!r} not found", file=sys.stderr)
            sys.exit(1)
        return prompt_config_path(registry.path_for(space.id))
    return prompt_config_path(root)


def _model_config_target(args: argparse.Namespace) -> Path:
    """Return the model config file targeted by CLI arguments."""
    from world0.models import model_config_path
    from world0.spaces import SpaceRegistry

    root = Path(args.store).expanduser()
    if args.space:
        registry = SpaceRegistry(root)
        space = registry.resolve(args.space)
        if space is None:
            print(f"error: space {args.space!r} not found", file=sys.stderr)
            sys.exit(1)
        return model_config_path(registry.path_for(space.id))
    return model_config_path(root)


def _handle_model_command(args: argparse.Namespace) -> None:
    """Manage per-operation model overrides without constructing an LLM."""
    from world0.models import (
        OperationModelSpec,
        load_operation_model_config,
        save_operation_model_config,
    )

    config_path = _model_config_target(args)
    config = load_operation_model_config(config_path)
    cmd = args.model_cmd or "list"

    if cmd == "list":
        print(f"{'OPERATION':<22} {'OVERRIDE':<8} {'PROVIDER':<13} MODEL")
        for operation in config.operations():
            spec = config.raw(operation)
            if spec:
                print(
                    f"{operation:<22} {'yes':<8} {spec.provider:<13} {spec.model}"
                )
            else:
                print(f"{operation:<22} {'no':<8} {'':<13} inherited")
        return

    if cmd == "show":
        if args.operation not in config.operations():
            print(f"error: unknown operation {args.operation!r}", file=sys.stderr)
            sys.exit(1)
        spec = config.raw(args.operation)
        data = (
            spec.to_dict(include_operation=True)
            if spec
            else {
                "operation": args.operation,
                "enabled": False,
                "provider": "",
                "model": "",
                "notes": "inherits runtime provider/model",
            }
        )
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    if cmd == "set":
        try:
            config.set(OperationModelSpec(
                operation=args.operation,
                provider=args.provider,
                model=args.model_name,
                enabled=not args.disabled,
                notes=args.notes,
            ))
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        issues = config.validate()
        if issues:
            for issue in issues:
                print(f"error: {issue}", file=sys.stderr)
            sys.exit(1)
        save_operation_model_config(config_path, config)
        print(f"Updated {args.operation} model config in {config_path}")
        return

    if cmd == "reset":
        try:
            config.clear(args.operation)
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        save_operation_model_config(config_path, config)
        print(f"Reset {args.operation} model config in {config_path}")
        return

    if cmd == "validate":
        issues = config.validate()
        if not issues:
            print(f"Model config OK: {config_path}")
            return
        for issue in issues:
            print(f"error: {issue}", file=sys.stderr)
        sys.exit(1)

    print(f"Unknown model subcommand: {cmd}", file=sys.stderr)
    sys.exit(1)


def _handle_prompt_command(args: argparse.Namespace) -> None:
    """Manage runtime prompt overrides without constructing an LLM provider."""
    from world0.prompts import (
        PromptSpec,
        export_prompt_config,
        load_prompt_registry,
        save_prompt_overrides,
    )

    config_path = _prompt_config_target(args)
    registry = load_prompt_registry(config_path)
    cmd = args.prompt_cmd or "list"

    if cmd == "list":
        print(f"{'ID':<42} {'OVERRIDE':<8} {'OUTPUT':<6} DESCRIPTION")
        for spec in registry.all():
            overridden = "yes" if registry.is_overridden(spec.id) else "no"
            print(f"{spec.id:<42} {overridden:<8} {spec.output:<6} {spec.description}")
        return

    if cmd == "show":
        spec = registry.get(args.prompt_id)
        if args.json:
            print(json.dumps(spec.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(spec.template)
        return

    if cmd == "export":
        data = export_prompt_config(registry)
        rendered = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            Path(args.output).expanduser().write_text(rendered, encoding="utf-8")
            print(f"Exported prompts to {args.output}")
        else:
            print(rendered, end="")
        return

    if cmd == "set":
        base = registry.default(args.prompt_id)
        if args.file:
            template = Path(args.file).expanduser().read_text(encoding="utf-8")
        elif args.template is not None:
            template = args.template
        else:
            print("error: provide --file or --template", file=sys.stderr)
            sys.exit(1)

        data = base.to_dict(include_id=False)
        data["template"] = template
        data["variables"] = []
        registry.set_override(PromptSpec.from_dict(args.prompt_id, data))
        save_prompt_overrides(config_path, registry)
        print(f"Updated {args.prompt_id} in {config_path}")
        return

    if cmd == "reset":
        registry.default(args.prompt_id)
        registry.clear_override(args.prompt_id)
        save_prompt_overrides(config_path, registry)
        print(f"Reset {args.prompt_id} in {config_path}")
        return

    if cmd == "validate":
        issues = registry.validate()
        if not issues:
            print(f"Prompt config OK: {config_path}")
            return
        for issue in issues:
            print(issue.render())
        if any(issue.severity == "error" for issue in issues):
            sys.exit(1)
        return

    if cmd == "diff":
        overrides = registry.overrides()
        if not overrides:
            print("No prompt overrides.")
            return
        for prompt_id in sorted(overrides):
            current = registry.get(prompt_id)
            base = registry.default(prompt_id)
            print(f"## {prompt_id}")
            print(f"- default length: {len(base.template)} chars")
            print(f"- override length: {len(current.template)} chars")
        return

    print(f"Unknown prompt subcommand: {cmd}", file=sys.stderr)
    sys.exit(1)


def _handle_space_command(args: argparse.Namespace) -> None:
    """Space subcommands don't need an LLM or a full PKMAgent."""
    from world0.spaces import SpaceRegistry

    root = Path(args.store).expanduser()
    registry = SpaceRegistry(root)
    cmd = args.space_cmd

    if cmd is None or cmd == "list":
        spaces = registry.list()
        if not spaces:
            print("No spaces yet. Create one with:")
            print("  pkm space create <name>")
            return
        active_id = registry.active().id if registry.active() else None
        print(f"{'':2} {'ID':<20} {'NAME':<24} DESCRIPTION")
        for s in spaces:
            marker = "*" if s.id == active_id else " "
            desc = s.description or ""
            print(f"{marker:2} {s.id:<20} {s.name:<24} {desc}")
        return

    if cmd == "show":
        space = registry.active()
        if space is None:
            print("No active space. Use `pkm space create <name>`.")
            return
        print(f"Active space:")
        print(f"  id:          {space.id}")
        print(f"  name:        {space.name}")
        print(f"  description: {space.description or '(none)'}")
        print(f"  path:        {registry.path_for(space.id)}")
        print(f"  created:     {space.created_at.isoformat()}")
        print(f"  last active: {space.last_active_at.isoformat()}")
        return

    if cmd == "create":
        try:
            space = registry.create(args.name, description=args.description)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Created space {space.name!r} (id: {space.id}).")
        if registry.active() and registry.active().id == space.id:
            print("Set as active space.")
        return

    if cmd == "use":
        space = registry.resolve(args.target)
        if space is None:
            print(f"error: space {args.target!r} not found", file=sys.stderr)
            sys.exit(1)
        registry.set_active(space.id)
        print(f"Active space is now {space.name!r} ({space.id}).")
        return

    if cmd == "rename":
        space = registry.resolve(args.target)
        if space is None:
            print(f"error: space {args.target!r} not found", file=sys.stderr)
            sys.exit(1)
        try:
            registry.rename(space.id, args.new_name)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Renamed {space.id} → {args.new_name}.")
        return

    if cmd == "delete":
        space = registry.resolve(args.target)
        if space is None:
            print(f"error: space {args.target!r} not found", file=sys.stderr)
            sys.exit(1)
        registry.delete(space.id, purge_data=args.purge)
        detail = " (data purged)" if args.purge else ""
        print(f"Deleted space {space.name!r}{detail}.")
        return

    print(f"Unknown space subcommand: {cmd}", file=sys.stderr)
    sys.exit(1)


def _create_provider(provider: str, model: str | None) -> LLMProvider | None:
    """Create an LLM provider from CLI arguments."""
    provider = normalize_provider_name(provider)
    if provider == "none":
        return None

    if provider == "anthropic":
        from world0.llm.anthropic import AnthropicProvider

        kwargs = {}
        if model:
            kwargs["model"] = model
        return AnthropicProvider(**kwargs)

    if provider == "openai":
        from world0.llm.openai import OpenAIProvider

        kwargs = {}
        if model:
            kwargs["model"] = model
        return OpenAIProvider(**kwargs)

    if provider == "azure-openai":
        from world0.llm.azure_openai import AzureOpenAIProvider

        kwargs = {}
        if model:
            kwargs["model"] = model
        return AzureOpenAIProvider(**kwargs)

    print(f"Unknown provider: {provider}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="World 0 PKM Agent — Personal Knowledge Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s                          Start interactive chat (Anthropic)
  %(prog)s --provider openai        Start with OpenAI
  %(prog)s --provider none          Start without LLM (structured input only)
  %(prog)s learn "some text..."     Quick learn
  %(prog)s ask "some question?"     Quick ask
  %(prog)s web-search "latest MCP patterns"   Quick web search
  %(prog)s explore "concept"        Quick explore
  %(prog)s claude "review this architecture"
  %(prog)s codex "inspect this repository for test gaps"
  %(prog)s prompt list              List configurable prompts
  %(prog)s model list               List operation model overrides
  %(prog)s status                   Show world status
  %(prog)s reflect                  Run consolidation
""",
    )

    parser.add_argument(
        "--store",
        default="~/.pkm_world",
        help="Path to knowledge store (default: ~/.pkm_world)",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "claude", "openai", "codex", "azure-openai", "none"],
        default="anthropic",
        help="LLM provider (default: anthropic; claude/codex are aliases)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override (e.g., gpt-5.4, claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--space",
        default=None,
        help="Space to use for this command (name or id). "
        "Overrides the active space without changing it.",
    )

    subparsers = parser.add_subparsers(dest="command")

    # learn
    learn_parser = subparsers.add_parser("learn", help="Learn from text")
    learn_parser.add_argument("text", help="Text to learn from")
    learn_parser.add_argument("--task", default="knowledge intake")
    learn_parser.add_argument("--source", default="")

    # ask
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("query", help="Question to ask")

    # explore
    explore_parser = subparsers.add_parser("explore", help="Explore a concept")
    explore_parser.add_argument("concept", help="Concept name to explore")

    # connect
    connect_parser = subparsers.add_parser(
        "connect", help="Connect two concepts"
    )
    connect_parser.add_argument("source", help="Source concept")
    connect_parser.add_argument("target", help="Target concept")
    connect_parser.add_argument(
        "--type",
        default="generic_relation",
        help="Semantic relation label, for example membership, inclusion, conflict, overlap",
    )

    # search
    search_parser = subparsers.add_parser("search", help="Search concepts")
    search_parser.add_argument("query", help="Search query")

    # web-search
    web_search_parser = subparsers.add_parser(
        "web-search", help="Search the public web"
    )
    web_search_parser.add_argument("query", help="Web search query")
    web_search_parser.add_argument("--focus", default="", help="Optional search focus")
    web_search_parser.add_argument(
        "--domains",
        default="",
        help="Optional comma-separated domain filters",
    )
    web_search_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results"
    )
    web_search_parser.add_argument(
        "--fetch-pages",
        action="store_true",
        help="Fetch the top pages and include short excerpts",
    )

    # status
    subparsers.add_parser("status", help="Show world status")

    # reflect
    subparsers.add_parser("reflect", help="Run consolidation")

    # claude
    claude_parser = subparsers.add_parser(
        "claude", help="Consult system Claude Code in read-only mode"
    )
    claude_parser.add_argument("prompt", help="Prompt for Claude Code")
    claude_parser.add_argument("--workspace", default=".")
    claude_parser.add_argument("--problem", default="")
    claude_parser.add_argument("--external-model", default="")
    claude_parser.add_argument(
        "--no-world0-context",
        action="store_true",
        help="Do not prepend World 0 projection context",
    )

    # codex
    codex_parser = subparsers.add_parser(
        "codex", help="Consult system Codex in read-only mode"
    )
    codex_parser.add_argument("prompt", help="Prompt for Codex")
    codex_parser.add_argument("--workspace", default=".")
    codex_parser.add_argument("--problem", default="")
    codex_parser.add_argument("--external-model", default="")
    codex_parser.add_argument(
        "--no-world0-context",
        action="store_true",
        help="Do not prepend World 0 projection context",
    )

    # viz
    viz_parser = subparsers.add_parser("viz", help="Generate visualization")
    viz_parser.add_argument("--output", default=None, help="Output file path")

    # space
    space_parser = subparsers.add_parser(
        "space", help="Manage isolated concept worlds (spaces)"
    )
    space_sub = space_parser.add_subparsers(dest="space_cmd")
    space_sub.add_parser("list", help="List all spaces")
    space_sub.add_parser("show", help="Show the currently active space")
    sp_create = space_sub.add_parser("create", help="Create a new space")
    sp_create.add_argument("name", help="Human-readable space name")
    sp_create.add_argument("--description", default="")
    sp_use = space_sub.add_parser("use", help="Switch the active space")
    sp_use.add_argument("target", help="Space name or id")
    sp_rename = space_sub.add_parser("rename", help="Rename a space")
    sp_rename.add_argument("target", help="Space name or id")
    sp_rename.add_argument("new_name")
    sp_delete = space_sub.add_parser("delete", help="Delete a space")
    sp_delete.add_argument("target", help="Space name or id")
    sp_delete.add_argument(
        "--purge",
        action="store_true",
        help="Also delete the on-disk concept data",
    )

    # prompt
    prompt_parser = subparsers.add_parser(
        "prompt", help="Manage runtime prompt configuration"
    )
    prompt_sub = prompt_parser.add_subparsers(dest="prompt_cmd")
    prompt_sub.add_parser("list", help="List all configurable prompts")
    prompt_show = prompt_sub.add_parser("show", help="Show an effective prompt")
    prompt_show.add_argument("prompt_id")
    prompt_show.add_argument(
        "--json",
        action="store_true",
        help="Show prompt metadata as JSON",
    )
    prompt_export = prompt_sub.add_parser(
        "export", help="Export all effective prompts as JSON"
    )
    prompt_export.add_argument("--output", default="")
    prompt_set = prompt_sub.add_parser("set", help="Override a prompt")
    prompt_set.add_argument("prompt_id")
    prompt_set.add_argument("--file", default="")
    prompt_set.add_argument("--template", default=None)
    prompt_reset = prompt_sub.add_parser("reset", help="Remove a prompt override")
    prompt_reset.add_argument("prompt_id")
    prompt_sub.add_parser("validate", help="Validate prompt configuration")
    prompt_sub.add_parser("diff", help="Summarize prompt overrides")

    # model
    model_parser = subparsers.add_parser(
        "model", help="Manage per-operation model configuration"
    )
    model_sub = model_parser.add_subparsers(dest="model_cmd")
    model_sub.add_parser("list", help="List operation model overrides")
    model_show = model_sub.add_parser("show", help="Show one operation override")
    model_show.add_argument("operation")
    model_set = model_sub.add_parser("set", help="Set one operation model override")
    model_set.add_argument("operation")
    model_set.add_argument("--provider", default="")
    model_set.add_argument("--model", dest="model_name", required=True)
    model_set.add_argument("--notes", default="")
    model_set.add_argument(
        "--disabled",
        action="store_true",
        help="Store the override but leave it disabled",
    )
    model_reset = model_sub.add_parser("reset", help="Remove one model override")
    model_reset.add_argument("operation")
    model_sub.add_parser("validate", help="Validate model configuration")

    args = parser.parse_args()

    # ── space subcommands: handled before constructing PKMAgent ───────
    if args.command == "space":
        _handle_space_command(args)
        return
    if args.command == "prompt":
        _handle_prompt_command(args)
        return
    if args.command == "model":
        _handle_model_command(args)
        return

    llm = _create_provider(args.provider, args.model)
    agent = PKMAgent(store_path=args.store, llm=llm, space_id=args.space)

    if args.command is None:
        # Interactive mode
        if agent.space is not None:
            print(f"[space: {agent.space.name} ({agent.space.id})]")
        agent.chat()
        return

    if args.command == "learn":
        print(agent.learn(args.text, task=args.task, source=args.source))
    elif args.command == "ask":
        print(agent.ask(args.query))
    elif args.command == "explore":
        print(agent.explore(args.concept))
    elif args.command == "connect":
        print(agent.connect(args.source, args.target, args.type))
    elif args.command == "search":
        print(agent.search(args.query))
    elif args.command == "web-search":
        print(agent.search_web(
            args.query,
            focus=args.focus,
            max_results=args.limit,
            domains=args.domains,
            fetch_pages=args.fetch_pages,
        ))
    elif args.command == "status":
        print(agent.status())
    elif args.command == "reflect":
        print(agent.reflect())
    elif args.command == "claude":
        print(agent.consult_external_agent(
            "claude",
            args.prompt,
            workspace=args.workspace,
            problem=args.problem,
            model=args.external_model,
            use_world0_context=not args.no_world0_context,
        ))
    elif args.command == "codex":
        print(agent.consult_external_agent(
            "codex",
            args.prompt,
            workspace=args.workspace,
            problem=args.problem,
            model=args.external_model,
            use_world0_context=not args.no_world0_context,
        ))
    elif args.command == "viz":
        print(agent.visualize(output=args.output))


if __name__ == "__main__":
    main()
