"""Shared parsing helpers for LLM responses.

Every LLM-facing subsystem (extraction, similarity judging, relation
typing) needs the same tolerance for model output quirks — markdown
fences, prose around the JSON object.  One implementation here keeps
their behavior identical.
"""

from __future__ import annotations

import re

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> str:
    """Extract a JSON object from LLM output.

    Handles markdown code fences and leading/trailing prose; returns
    the input unchanged when no object can be located (the caller's
    ``json.loads`` will then raise with the original content visible).
    """
    match = _FENCED_JSON.search(text)
    if match:
        return match.group(1)
    match = _BARE_JSON.search(text)
    if match:
        return match.group(0)
    return text
