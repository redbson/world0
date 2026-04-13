"""Web research helpers for World 0 agent workflows.

Provides lightweight, dependency-free web search and fetch utilities that
can feed source material into the concept-world pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import re
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
import urllib.request

USER_AGENT = "World0-Research/1.0"


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    domain: str = ""


@dataclass
class FetchedDocument:
    title: str
    url: str
    text: str


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>[\s\S]*?</script>", " ", html, flags=re.I)
    text = re.sub(r"<style[^>]*>[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<!--[\s\S]*?-->", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return _collapse_ws(unescape(text))


def _decode_result_url(href: str) -> str:
    if not href:
        return ""
    if href.startswith("//"):
        return f"https:{href}"
    if href.startswith("http://") or href.startswith("https://"):
        return href

    parsed = urlparse(href)
    if parsed.path == "/l/":
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target) if target else ""
    return ""


def _extract_domain(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _normalize_domains(domains: list[str] | tuple[str, ...] | None) -> list[str]:
    if not domains:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for domain in domains:
        clean = domain.strip().lower()
        if not clean:
            continue
        clean = clean.removeprefix("https://").removeprefix("http://")
        clean = clean.split("/", 1)[0].removeprefix("www.")
        if clean and clean not in seen:
            normalized.append(clean)
            seen.add(clean)
    return normalized


def parse_duckduckgo_results(html: str, limit: int = 5) -> list[SearchResult]:
    """Extract search results from DuckDuckGo's HTML endpoint."""
    anchors = list(re.finditer(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        flags=re.I | re.S,
    ))
    results: list[SearchResult] = []
    seen_urls: set[str] = set()

    for idx, match in enumerate(anchors):
        href = match.group(1)
        title = _strip_html(match.group(2))
        url = _decode_result_url(href)
        block_end = anchors[idx + 1].start() if idx + 1 < len(anchors) else len(html)
        block = html[match.end():block_end]
        snippet_match = re.search(
            r'<(?:a|div)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
            block,
            flags=re.I | re.S,
        )
        snippet = _strip_html(snippet_match.group(1)) if snippet_match else ""

        domain = _extract_domain(url)
        if title and url and url not in seen_urls:
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                domain=domain,
            ))
            seen_urls.add(url)
        if len(results) >= limit:
            break

    return results


def search_web(
    query: str,
    limit: int = 5,
    timeout: int = 15,
    domains: list[str] | tuple[str, ...] | None = None,
) -> list[SearchResult]:
    """Run a lightweight web search using DuckDuckGo's HTML endpoint."""
    query = query.strip()
    if not query:
        return []

    domain_filters = _normalize_domains(domains)
    if domain_filters:
        query = f"{query} " + " OR ".join(f"site:{domain}" for domain in domain_filters)

    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}&kl=wt-wt"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
        html = raw.decode(charset, errors="replace")
    results = parse_duckduckgo_results(html, limit=max(limit * 2, limit))
    if domain_filters:
        results = [item for item in results if item.domain in domain_filters]
    return results[:limit]


def fetch_web_document(
    url: str,
    *,
    max_chars: int = 12000,
    timeout: int = 20,
) -> FetchedDocument:
    """Fetch a web page and extract a compact plain-text representation."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
        html = raw.decode(charset, errors="replace")

    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
    title = _strip_html(title_match.group(1)) if title_match else url
    text = _strip_html(html)
    if len(text) > max_chars:
        text = f"{text[:max_chars]}\n...[truncated]"
    return FetchedDocument(title=title or url, url=url, text=text)
