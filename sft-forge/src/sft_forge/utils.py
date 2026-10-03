"""Small shared helpers: ids, jsonl IO, text normalisation, rate limiting."""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

LOGGER_NAME = "sft_forge"

# --------------------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------------------


def setup_logging(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """Configure root-ish logging once and return this package's logger."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        return logger
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING if quiet else logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


# --------------------------------------------------------------------------------------
# ids / time
# --------------------------------------------------------------------------------------


def utcnow_iso() -> str:
    """UTC timestamp, second resolution, e.g. ``2026-10-03T12:00:00Z``."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utcstamp() -> str:
    """Compact UTC timestamp safe for filenames, e.g. ``20261003-120000``."""
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def stable_hash(*parts: Any) -> str:
    """Order-independent-ish sha256 over arbitrary JSON-serialisable parts."""
    digest = hashlib.sha256()
    for part in parts:
        payload = json.dumps(part, sort_keys=True, ensure_ascii=False, default=str)
        digest.update(payload.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def short_id(*parts: Any, n: int = 12) -> str:
    return stable_hash(*parts)[:n]


def slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (slug or "run")[:max_len].strip("-")


# --------------------------------------------------------------------------------------
# jsonl IO
# --------------------------------------------------------------------------------------


def _open_maybe_gzip(path: Path, mode: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def read_jsonl(path: str | os.PathLike[str]) -> list[dict]:
    """Read a ``.jsonl`` / ``.jsonl.gz`` file into a list of dicts.

    Blank lines are skipped; a malformed line raises with the line number so broken
    hand-edited seed files are easy to debug.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    out: list[dict] = []
    with _open_maybe_gzip(p, "r") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - message quality
                raise ValueError(f"{p}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{p}:{lineno}: expected a JSON object per line")
            out.append(obj)
    return out


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[dict]:
    """Streaming variant of :func:`read_jsonl` for large datasets."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    with _open_maybe_gzip(p, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> int:
    """Atomically write rows as JSONL (UTF-8, one object per line). Returns the count."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp = p.with_name(p.name + ".tmp")
    with _open_maybe_gzip(tmp, "w") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=False))
            fh.write("\n")
            count += 1
    os.replace(tmp, p)
    return count


def append_jsonl(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> int:
    """Append rows to a JSONL file with an fsync, so a crash keeps completed work."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with _open_maybe_gzip(p, "a") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False))
            fh.write("\n")
            count += 1
        fh.flush()
        with contextlib.suppress(OSError, ValueError):  # pragma: no cover - e.g. gzip streams
            os.fsync(fh.fileno())
    return count


def write_json(path: str | os.PathLike[str], obj: Any, indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=indent, ensure_ascii=False, sort_keys=False)
    atomic_write_text(p, text + "\n")
    return p


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def atomic_write_text(path: str | os.PathLike[str], text: str) -> Path:
    """Write text via a temp file + rename so readers never see a partial file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_name, p)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return p


# --------------------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------------------


_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")

# Deliberately tiny: used as a cheap "is this English-ish?" heuristic, not a real LID model.
ENGLISH_STOPWORDS = frozenset(
    ["a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those", "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "doing", "have", "has", "had", "having", "of", "in", "on", "at", "to", "for", "from", "with", "without", "into", "over", "under", "about", "as", "by", "it", "its", "it's", "you", "your", "we", "our", "they", "their", "he", "she", "i", "me", "my", "not", "no", "can", "could", "should", "would", "will", "just", "also", "more", "most", "some", "any", "there", "here", "what", "which", "who", "when", "where", "why", "how"]
)


def normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def normalize_for_dedup(text: str) -> str:
    """Lowercase, strip punctuation/markdown, collapse whitespace — dedup input form."""
    text = (text or "").lower()
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return _WS_RE.sub(" ", text).strip()


def words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def word_count(text: str) -> int:
    return len((text or "").split())


def approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for datasets without a real tokenizer."""
    return max(1, int(len(text or "") / 4))


def english_ratio(text: str) -> float:
    """Fraction of tokens that are common English stopwords. 0.0 for empty text."""
    toks = words(text)
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in ENGLISH_STOPWORDS) / len(toks)


def non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if ord(ch) > 127) / len(text)


def ngram_repeat_ratio(text: str, n: int = 4) -> float:
    """Share of repeated n-grams; high values flag degenerate/looping generations."""
    toks = words(text)
    if len(toks) < n * 2:
        return 0.0
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def longest_line_ratio(text: str) -> float:
    """Length of the longest line over total length — flags runaway single lines."""
    if not text:
        return 0.0
    longest = max((len(line) for line in text.splitlines()), default=0)
    return longest / max(1, len(text))


def truncate(text: str, limit: int, suffix: str = " …") -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - len(suffix)] + suffix


def unique_token_ratio(text: str) -> float:
    toks = words(text)
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


def chunked(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def batched_indices(total: int, batch: int) -> Iterator[range]:
    for start in range(0, total, batch):
        yield range(start, min(start + batch, total))


# --------------------------------------------------------------------------------------
# concurrency helpers
# --------------------------------------------------------------------------------------


class RateLimiter:
    """Simple thread-safe token-bucket limiter. ``rate=None`` disables limiting."""

    def __init__(self, rate_per_minute: float | None):
        self.rate = float(rate_per_minute) if rate_per_minute else None
        self.capacity = max(1.0, self.rate / 10.0) if self.rate else 0.0
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        if not self.rate:
            return
        with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate / 60.0)
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                time.sleep(min(2.0, deficit * 60.0 / self.rate))


class Counter:
    """Thread-safe counter with dict payload merge (used for token/cost accounting)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, int] = {}
        self._sums: dict[str, float] = {}

    def incr(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._items[key] = self._items.get(key, 0) + amount

    def add_float(self, key: str, amount: float) -> None:
        with self._lock:
            self._sums[key] = self._sums.get(key, 0.0) + amount

    def get(self, key: str, default: int = 0) -> int:
        with self._lock:
            return self._items.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {**self._items, **{k: round(v, 6) for k, v in self._sums.items()}}


# --------------------------------------------------------------------------------------
# .env loading (no python-dotenv dependency)
# --------------------------------------------------------------------------------------


def load_env_file(path: str | os.PathLike[str] | None = None) -> int:
    """Load ``KEY=VALUE`` lines into ``os.environ`` (existing vars win). Returns count."""
    candidates = [Path(path)] if path else [Path(".env"), Path(".env.local")]
    loaded = 0
    for candidate in candidates:
        if not candidate.exists():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded += 1
    return loaded
