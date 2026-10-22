"""Cleaning up local data: old upload batches, orphaned indexes and old sessions.

Nothing is removed until ``remove`` is called on what ``find_stale`` found, so
the CLI can show a dry run first.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from . import sessions
from .config import Settings
from .pools import pool_name


@dataclass(frozen=True)
class Stale:
    kind: str  # "uploads", "index" or "session"
    name: str
    detail: str
    path: Path | None = None  # upload batch folder
    collection: str | None = None  # index to drop
    session_id: str | None = None

    def line(self) -> str:
        return f"{self.kind:<8} {self.name}  ({self.detail})"


def _client(settings: Settings) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=settings.db_dir, settings=ChromaSettings(anonymized_telemetry=False))


def _last_change(folder: Path) -> datetime:
    times = [folder.stat().st_mtime, *(f.stat().st_mtime for f in folder.iterdir())]
    return datetime.fromtimestamp(max(times))


def _size_mb(folder: Path) -> float:
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e6


def find_stale(settings: Settings, older_than_days: int) -> list[Stale]:
    """Upload batches and sessions unused for ``older_than_days``, and indexes whose
    folder no longer exists (or is one of those batches)."""
    cutoff = datetime.now() - timedelta(days=older_than_days)
    stale: list[Stale] = []

    uploads = Path(settings.uploads_dir).expanduser()
    old_batches = [b for b in uploads.iterdir() if b.is_dir() and _last_change(b) < cutoff] if uploads.exists() else []
    for batch in old_batches:
        age = (datetime.now() - _last_change(batch)).days
        files = sum(1 for _ in batch.iterdir())
        stale.append(Stale("uploads", batch.name, f"{files} files, {_size_mb(batch):.1f} MB, {age} days old",
                           path=batch, collection=pool_name(batch.resolve())))
    covered = {item.collection for item in stale}

    if Path(settings.db_dir).expanduser().exists():
        for collection in _client(settings).list_collections():
            folder = (collection.metadata or {}).get("folder")
            if collection.name in covered or not collection.name.startswith("pool-") or not folder:
                continue
            if not Path(folder).exists():
                stale.append(Stale("index", collection.name, f"folder {folder} no longer exists",
                                   collection=collection.name))

    if Path(settings.sessions_db).expanduser().exists():
        saver = sessions.open_saver(settings.sessions_db)
        for session_id, updated in sessions.last_updated(saver).items():
            if updated.replace(tzinfo=None) < cutoff:
                age = (datetime.now() - updated.replace(tzinfo=None)).days
                stale.append(Stale("session", session_id, f"last used {age} days ago", session_id=session_id))
    return stale


def remove(settings: Settings, items: list[Stale]) -> None:
    client = _client(settings) if any(item.collection for item in items) else None
    saver = sessions.open_saver(settings.sessions_db) if any(item.session_id for item in items) else None
    existing = {c.name for c in client.list_collections()} if client else set()
    for item in items:
        if item.path is not None:
            shutil.rmtree(item.path, ignore_errors=True)
        if item.collection in existing:
            client.delete_collection(item.collection)
        if item.session_id:
            sessions.delete_session(saver, item.session_id)
