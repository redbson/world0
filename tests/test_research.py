"""Tests for web research helpers."""

from __future__ import annotations

from world0.agents.research import parse_duckduckgo_results


def test_parse_duckduckgo_results():
    html = """
    <div class="result">
      <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Falpha">Alpha Result</a>
      <a class="result__snippet">First snippet about alpha.</a>
    </div>
    <div class="result">
      <a class="result__a" href="https://example.org/beta">Beta Result</a>
      <div class="result__snippet">Second snippet about beta.</div>
    </div>
    """

    results = parse_duckduckgo_results(html, limit=5)

    assert len(results) == 2
    assert results[0].title == "Alpha Result"
    assert results[0].url == "https://example.com/alpha"
    assert "alpha" in results[0].snippet.lower()
    assert results[1].url == "https://example.org/beta"
