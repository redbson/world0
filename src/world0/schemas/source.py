"""Source records for raw material ingested into World 0."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from pydantic import BaseModel, Field


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def source_hash(raw_text: str) -> str:
    """Stable content hash for raw source text."""
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def source_id_for(raw_text: str, *, task: str = "", source: str = "") -> str:
    """Stable source id from raw content plus caller context."""
    key = "\n".join([task.strip(), source.strip(), raw_text])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def source_tokens(text: str) -> list[str]:
    """Small token list used for JSON source indexing/search."""
    seen: set[str] = set()
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


class SourceRecord(BaseModel):
    """Raw material captured before LLM extraction."""

    id: str
    raw_text: str
    content_hash: str
    source: str = ""
    task: str = ""
    domain: str = ""
    concepts: list[str] = Field(default_factory=list)
    relation_count: int = 0
    tokens: list[str] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @classmethod
    def from_raw(
        cls,
        raw_text: str,
        *,
        task: str = "",
        source: str = "",
    ) -> "SourceRecord":
        return cls(
            id=source_id_for(raw_text, task=task, source=source),
            raw_text=raw_text,
            content_hash=source_hash(raw_text),
            source=source,
            task=task,
            tokens=source_tokens(" ".join([source, task, raw_text])),
        )

    def attach_extraction(
        self,
        *,
        concepts: list[str],
        relation_count: int,
        domain: str = "",
    ) -> None:
        self.concepts = list(dict.fromkeys([c for c in concepts if c]))
        self.relation_count = relation_count
        if domain:
            self.domain = domain
        self.updated_at = datetime.now(timezone.utc)
