"""Browser-based launcher for the World 0 web UI."""

from __future__ import annotations

import argparse

from world0.agents.gui import launch_browser


def main() -> None:
    parser = argparse.ArgumentParser(
        description="World 0 Concept World — browser launcher",
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
        help="Model name override",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the web server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8420,
        help="Port number (default: 8420)",
    )

    args = parser.parse_args()
    launch_browser(
        store_path=args.store,
        provider=args.provider,
        model=args.model,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
