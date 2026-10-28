"""Reproducible model answers: every structured answer stored by what was asked.

The same question to the same model (provider, model name, temperature, output
schema, full prompt and input) is answered from the cache, so screening the
same resumes for the same job description gives the same requirements, deep
reviews and rationales in every run and session, and costs nothing the second
time. A changed prompt, schema, model or input is a different question.

The cache keeps the model's raw answer; guardrails and scoring still run on
it, so improvements to them apply to cached answers too. Each entry records
the request, the answer, the model, when it was made and how often it was
reused: an audit trail of what the model was asked and said.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Part of every key: bump it if what a stored answer means ever changes.
CACHE_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    kind TEXT NOT NULL,
    request TEXT NOT NULL,
    answer TEXT NOT NULL,
    created TEXT NOT NULL,
    last_used TEXT NOT NULL,
    uses INTEGER NOT NULL DEFAULT 1
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cache_key(model: str, schema: dict, system: str, user: str) -> str:
    """What identifies a question: the model and everything it was given."""
    question = {"version": CACHE_VERSION, "model": model, "schema": schema, "system": system, "user": user}
    return hashlib.sha256(json.dumps(question, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class CacheStats:
    answers: int
    reused: int  # answers served from the cache instead of the model
    size_mb: float
    by_kind: dict[str, int]


class ResponseCache:
    """The answers stored in one SQLite file, safe to share between threads."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)

    def get(self, key: str) -> str | None:
        """The stored answer for ``key`` (and count the reuse), or None."""
        with self._lock:
            row = self._db.execute("SELECT answer FROM answers WHERE key = ?", (key,)).fetchone()
            if row:
                self._db.execute("UPDATE answers SET uses = uses + 1, last_used = ? WHERE key = ?", (_now(), key))
        return row[0] if row else None

    def put(self, key: str, *, model: str, kind: str, request: dict, answer: str, replace: bool = False) -> str:
        """Store an answer and return the stored one: if another thread answered the same
        question first, its answer wins, so everyone uses the same one. ``replace``
        overwrites an existing answer (one that could not be read)."""
        now = _now()
        with self._lock:
            self._db.execute(
                f"INSERT OR {'REPLACE' if replace else 'IGNORE'} INTO answers "
                "(key, model, kind, request, answer, created, last_used) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, model, kind, json.dumps(request), answer, now, now),
            )
            return self._db.execute("SELECT answer FROM answers WHERE key = ?", (key,)).fetchone()[0]

    def stats(self) -> CacheStats:
        with self._lock:
            answers, uses = self._db.execute("SELECT COUNT(*), COALESCE(SUM(uses), 0) FROM answers").fetchone()
            by_kind = dict(self._db.execute("SELECT kind, COUNT(*) FROM answers GROUP BY kind ORDER BY 2 DESC"))
        size = sum(p.stat().st_size for p in self.path.parent.glob(self.path.name + "*")) / 1e6
        return CacheStats(answers, uses - answers, size, by_kind)

    def unused_since(self, days: int) -> int:
        """How many answers were last used more than ``days`` ago."""
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM answers WHERE last_used < ?", (_cutoff(days),)).fetchone()[0]

    def remove_unused(self, days: int) -> int:
        with self._lock:
            return self._db.execute("DELETE FROM answers WHERE last_used < ?", (_cutoff(days),)).rowcount

    def clear(self) -> int:
        with self._lock:
            removed = self._db.execute("DELETE FROM answers").rowcount
            self._db.execute("VACUUM")
        return removed

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __del__(self) -> None:
        db = getattr(self, "_db", None)
        if db is not None:
            db.close()


def _cutoff(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
