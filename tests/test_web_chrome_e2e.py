"""Browser smoke tests for the embedded World 0 web UI.

These tests drive a real Chrome browser through Playwright against a
real uvicorn server, instead of using TestClient alone.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from world0.agents.web import create_app

playwright = pytest.importorskip("playwright.sync_api")


def _find_chrome_executable() -> str | None:
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return str(path)
    return None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(host: str, port: int, timeout: float = 10.0) -> None:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"Server at http://{host}:{port} did not become ready.")


@pytest.fixture
def chrome_path() -> str:
    executable = _find_chrome_executable()
    if not executable:
        pytest.skip("Chrome/Chromium executable not found.")
    return executable


@pytest.fixture
def web_server(tmp_path: Path):
    host = "127.0.0.1"
    port = _find_free_port()
    app = create_app(store_path=tmp_path / "chrome_web", llm=None)

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_for_server(host, port)

    try:
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_chrome_smoke_settings_and_connect_flow(web_server: str, chrome_path: str):
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=chrome_path,
            headless=True,
            args=["--no-proxy-server"],
        )
        page = browser.new_page()
        page.set_default_timeout(15000)
        page.goto(web_server, wait_until="domcontentloaded")

        page.wait_for_selector("#app-title")
        assert "World 0" in (page.locator("#app-title").text_content() or "")

        page.click("#btn-settings")
        page.wait_for_selector("#settings-auto-sediment-dialogue")
        assert page.locator("#settings-auto-sediment-dialogue").is_checked() is True
        page.locator("#settings-dialogue-sediment-interval").fill("3")
        page.locator("#settings-auto-sediment-dialogue").uncheck()
        assert page.locator("#settings-dialogue-sediment-interval").is_disabled() is True
        page.locator("#settings-auto-sediment-dialogue").check()
        assert page.locator("#settings-dialogue-sediment-interval").is_disabled() is False
        page.locator("#settings-dialogue-sediment-interval").fill("3")
        page.locator("#settings-modal .modal-header .modal-close").click()

        page.click("#btn-settings")
        page.wait_for_selector("#settings-dialogue-sediment-interval")
        page.locator("#settings-dialogue-sediment-interval").fill("3")
        page.locator("#settings-modal button[onclick='saveSettings()']").click()
        page.wait_for_timeout(500)

        messages_text = page.locator("#messages").inner_text().lower()
        assert "settings updated" in messages_text

        page.select_option("#mode-select", "connect")
        page.fill("#user-input", "python fastapi supports")
        page.click("#send-btn")
        page.wait_for_timeout(1200)

        messages_text = page.locator("#messages").inner_text().lower()
        concept_list_text = page.locator("#concept-list").inner_text().lower()

        assert "python" in messages_text
        assert "connected" in messages_text
        assert "python" in concept_list_text
        assert "fastapi" in concept_list_text

        page.click("#btn-settings")
        page.wait_for_selector("#settings-dialogue-sediment-interval")
        assert page.locator("#settings-dialogue-sediment-interval").input_value() == "3"
        assert page.locator("#settings-auto-sediment-dialogue").is_checked() is True

        browser.close()


def test_chrome_input_send_waits_for_response_and_respects_ime(
    web_server: str,
    chrome_path: str,
):
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=chrome_path,
            headless=True,
            args=["--no-proxy-server"],
        )
        page = browser.new_page()
        page.set_default_timeout(15000)
        page.goto(web_server, wait_until="domcontentloaded")
        page.wait_for_selector("#user-input")

        page.evaluate("""
          window.__originalFetchForInputTest = window.fetch;
          window.__releaseConnectForInputTest = null;
          window.fetch = async (...args) => {
            const url = String(args[0]);
            if (url.includes("/api/connect")) {
              await new Promise(resolve => {
                window.__releaseConnectForInputTest = resolve;
              });
            }
            return window.__originalFetchForInputTest(...args);
          };
        """)

        page.select_option("#mode-select", "connect")
        page.fill("#user-input", "python fastapi supports")
        page.click("#send-btn")
        page.wait_for_function("window.__releaseConnectForInputTest !== null")

        assert page.locator("#user-input").input_value() == "python fastapi supports"

        page.evaluate("window.__releaseConnectForInputTest()")
        page.wait_for_function("document.querySelector('#user-input').value === ''")

        page.fill("#user-input", "zhongwen")
        state = page.evaluate("""
          const input = document.querySelector("#user-input");
          const before = document.querySelectorAll(".msg-user").length;
          input.dispatchEvent(new CompositionEvent("compositionstart", { bubbles: true }));
          const event = new KeyboardEvent("keydown", {
            key: "Enter",
            bubbles: true,
            cancelable: true,
            isComposing: true,
          });
          input.dispatchEvent(event);
          input.dispatchEvent(new CompositionEvent("compositionend", { bubbles: true }));
          ({
            inputValue: input.value,
            userMessagesDelta: document.querySelectorAll(".msg-user").length - before,
            defaultPrevented: event.defaultPrevented,
          });
        """)

        assert state["inputValue"] == "zhongwen"
        assert state["userMessagesDelta"] == 0
        assert state["defaultPrevented"] is False

        browser.close()
