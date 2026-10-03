"""On-disk request cache.

Iterating on prompts is the normal workflow, and re-running a pipeline should not re-pay
for identical requests. The cache is a tiny SQLite file keyed by a hash of the full
request, so it is safe to delete at any time.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .utils import get_logger, stable_hash, utcnow_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS completions (
    key        TEXT PRIMARY KEY,
    model      TEXT NOT NULL,
    hint       TEXT,
    response   TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_completions_model ON completions(model);
"""


class ResponseCache:
    """Thread-safe, process-safe-enough completion cache.

    Pass ``path=None`` to disable caching entirely (useful in tests).
    """

    def __init__(self, path: str | Path | None, enabled: bool = True):
        self.enabled = bool(enabled and path)
        self.path = Path(path).expanduser() if path else None
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self.hits = 0
        self.misses = 0
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- keys --------------------------------------------------------------------------

    @staticmethod
    def make_key(request: dict[str, Any]) -> str:
        return stable_hash("v1", request)

    # -- api ---------------------------------------------------------------------------

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled or self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM completions WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:  # pragma: no cover - corrupt row
            get_logger().warning("cache entry %s is corrupt; ignoring", key[:12])
            return None

    def put(self, key: str, value: dict[str, Any], model: str = "", hint: str | None = None) -> None:
        if not self.enabled or self._conn is None:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO completions (key, model, hint, response, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (key, model, hint, json.dumps(value, ensure_ascii=False), utcnow_iso()),
            )
            self._conn.commit()

    def stats(self) -> dict[str, int]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }

    def close(self) -> None:
        if self._conn is not None:
            with self._lock:
                self._conn.close()
            self._conn = None

    def __enter__(self) -> ResponseCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
