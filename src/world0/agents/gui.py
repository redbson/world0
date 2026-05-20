"""Native macOS GUI launcher for the PKM Agent.

Uses pywebview to create a native macOS window with a WebKit webview,
and uvicorn to serve the FastAPI backend.

Usage::

    # Launch the GUI
    python -m world0.agents.gui

    # With options
    python -m world0.agents.gui --provider openai --model gpt-4o
    python -m world0.agents.gui --store ~/.my_knowledge --port 9000

    # Or via the installed command
    pkm-gui
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path


def _find_free_port() -> int:
    """Find a free port on localhost."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _create_provider(provider: str, model: str | None):
    """Create an LLM provider."""
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


def _start_server(app, host: str, port: int) -> None:
    """Start uvicorn in a background thread."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")


def _wait_for_server(host: str, port: int, timeout: float = 10.0) -> bool:
    """Wait until the server is ready to accept connections."""
    import socket

    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def launch_native(
    store_path: str = "~/.pkm_world",
    provider: str = "anthropic",
    model: str | None = None,
    space_id: str | None = None,
    host: str = "127.0.0.1",
    port: int | None = None,
    width: int = 1200,
    height: int = 800,
) -> None:
    """Launch the PKM Agent as a native macOS window.

    Starts a FastAPI server in a background thread, then opens
    a pywebview native window pointing to it.
    """
    try:
        import webview
    except ImportError:
        print(
            "pywebview is required for the native GUI.\n"
            "Install with: pip install pywebview\n\n"
            "Alternatively, run the web version:\n"
            "  pkm-web  (opens in browser)",
            file=sys.stderr,
        )
        sys.exit(1)

    if port is None:
        port = _find_free_port()

    llm = _create_provider(provider, model)

    # Resolve model name for agentic mode
    from world0.agents.provider import MODEL_ALIASES
    agentic_model = model or "sonnet"
    agentic_model = MODEL_ALIASES.get(agentic_model, agentic_model)

    from world0.agents.web import create_app

    app = create_app(
        store_path=store_path,
        llm=llm,
        model=agentic_model,
        space_id=space_id,
    )

    # Start server in background
    server_thread = threading.Thread(
        target=_start_server, args=(app, host, port), daemon=True
    )
    server_thread.start()

    if not _wait_for_server(host, port):
        print("Failed to start server", file=sys.stderr)
        sys.exit(1)

    url = f"http://{host}:{port}"

    # Create native window
    window = webview.create_window(
        "World 0 — PKM Agent",
        url,
        width=width,
        height=height,
        min_size=(800, 600),
        text_select=True,
    )

    # Start the webview event loop (blocks until window closes)
    webview.start(
        gui="cocoa",  # macOS native
        debug=False,
    )


def launch_browser(
    store_path: str = "~/.pkm_world",
    provider: str = "anthropic",
    model: str | None = None,
    space_id: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8420,
    open_browser: bool = True,
) -> None:
    """Launch the PKM Agent as a web app (opens in default browser)."""
    import webbrowser

    llm = _create_provider(provider, model)

    from world0.agents.provider import MODEL_ALIASES
    agentic_model = model or "sonnet"
    agentic_model = MODEL_ALIASES.get(agentic_model, agentic_model)

    from world0.agents.web import create_app

    app = create_app(
        store_path=store_path,
        llm=llm,
        model=agentic_model,
        space_id=space_id,
    )

    url = f"http://{host}:{port}"
    print(f"Starting World 0 PKM Agent at {url}")
    print("Press Ctrl+C to stop.\n")

    # Open browser after a short delay when this is an interactive launch.
    def _open_browser():
        time.sleep(1.0)
        webbrowser.open(url)

    if open_browser:
        threading.Thread(target=_open_browser, daemon=True).start()

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="World 0 PKM Agent — GUI Application",
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
        "--port",
        type=int,
        default=None,
        help="Port number (default: auto for native, 8420 for web)",
    )
    parser.add_argument(
        "--space",
        default=None,
        help="Space to use for this run (name or id)",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Open in browser instead of native window",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically open the browser for --web",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1200,
        help="Window width (native only, default: 1200)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Window height (native only, default: 800)",
    )

    args = parser.parse_args()

    if args.web:
        launch_browser(
            store_path=args.store,
            provider=args.provider,
            model=args.model,
            space_id=args.space,
            host="127.0.0.1",
            port=args.port or 8420,
            open_browser=not args.no_open,
        )
    else:
        launch_native(
            store_path=args.store,
            provider=args.provider,
            model=args.model,
            space_id=args.space,
            port=args.port,
            width=args.width,
            height=args.height,
        )


if __name__ == "__main__":
    main()
