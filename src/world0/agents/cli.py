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
import sys
from pathlib import Path

from world0.agents.pkm import PKMAgent
from world0.llm.base import LLMProvider


def _create_provider(provider: str, model: str | None) -> LLMProvider | None:
    """Create an LLM provider from CLI arguments."""
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
        choices=["anthropic", "openai", "azure-openai", "none"],
        default="anthropic",
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override (e.g., gpt-5.4, claude-sonnet-4-6)",
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
        "--type", default="related_to", help="Relation type"
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

    # viz
    viz_parser = subparsers.add_parser("viz", help="Generate visualization")
    viz_parser.add_argument("--output", default=None, help="Output file path")

    args = parser.parse_args()

    llm = _create_provider(args.provider, args.model)
    agent = PKMAgent(store_path=args.store, llm=llm)

    if args.command is None:
        # Interactive mode
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
    elif args.command == "viz":
        print(agent.visualize(output=args.output))


if __name__ == "__main__":
    main()
