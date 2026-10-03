"""Forgiving JSON parsing for LLM output.

Models love to wrap JSON in prose or code fences, add trailing commas, or use smart
quotes. Everything here is about recovering the payload instead of raising.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_OPENERS = {"[": "]", "{": "}"}
_SMART_QUOTES = {
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u00ab": '"',
    "\u00bb": '"',
}
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def strip_code_fences(text: str) -> str:
    """Return the first fenced block if present, else the original text."""
    if not text:
        return ""
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text


def find_json_span(text: str) -> tuple[int, int] | None:
    """Locate the outermost balanced ``{...}``/``[...]`` span, string-aware.

    Returns ``(start, end_exclusive)`` or ``None``.
    """
    if not text:
        return None
    for start, ch in enumerate(text):
        if ch not in _OPENERS:
            continue
        stack = [ch]
        in_string = False
        escaped = False
        for i in range(start + 1, len(text)):
            c = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
            elif c in _OPENERS:
                stack.append(c)
            elif c in ("}", "]"):
                if not stack:
                    break
                expected = _OPENERS[stack[-1]]
                if c != expected:
                    break  # malformed nesting: give up on this opener
                stack.pop()
                if not stack:
                    return start, i + 1
    return None


def _repair(text: str) -> str:
    for src, dst in _SMART_QUOTES.items():
        text = text.replace(src, dst)
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    # Python-style literals that models sometimes emit inside JSON
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    return text


def loads_lenient(text: str) -> Any:
    """Best-effort ``json.loads``. Raises :class:`ValueError` if nothing is recoverable."""
    if text is None:
        raise ValueError("empty model output")
    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)
    inner = strip_code_fences(text).strip()
    if inner and inner != stripped:
        candidates.append(inner)
    span = find_json_span(inner or stripped)
    if span:
        candidates.append((inner or stripped)[span[0] : span[1]])

    for candidate in candidates:
        if not candidate:
            continue
        for variant in (candidate, _repair(candidate)):
            try:
                return json.loads(variant)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"could not parse JSON from model output: {text[:200]!r}")


def loads_or_none(text: str) -> Any | None:
    try:
        return loads_lenient(text)
    except ValueError:
        return None


def find_all_json_spans(text: str) -> list[str]:
    """Every top-level balanced span in the text, in order."""
    spans: list[str] = []
    cursor = 0
    body = text or ""
    while cursor < len(body):
        span = find_json_span(body[cursor:])
        if not span:
            break
        start, end = span
        spans.append(body[cursor + start : cursor + end])
        cursor += end
    return spans


def as_list(payload: Any, *keys: str) -> list[Any]:
    """Coerce a payload into a list, optionally digging into a wrapper key.

    Handles the usual shapes: ``[...]``, ``{"items": [...]}``, ``{"data": {...}}``.
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in keys or ("items", "data", "results", "examples", "instructions", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        # A single object that is itself the item
        return [payload]
    return []
