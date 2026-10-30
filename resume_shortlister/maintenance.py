"""Cleaning up local data: old upload batches, orphaned indexes, old sessions and
stored model answers no screening has used for a while (``find_stale``), or every
piece of candidate data to start from scratch (``find_all``).

Also the ChromaDB files that versions before 1.1 kept in the index folder; the
current index (``vector_store.DB_FILE``) replaced them, and folders re-index
on their next use.

Nothing is removed until ``remove`` is called on what ``find_stale`` found, so
the CLI can show a dry run first.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from . import sessions
from .config import Settings
from .llm_cache import ResponseCache
from .pools import pool_name
from .rag.vector_store import DB_FILE, VectorStore

# ChromaDB's database file; its collections were folders named by UUID next to it.
LEGACY_CHROMA_FILE = "chroma.sqlite3"


@dataclass(frozen=True)
class Stale:
    kind: str  # "uploads", "index", "session" or "cache"
    name: str
    detail: str
    paths: tuple[Path, ...] = ()  # files and folders to delete
    collection: str | None = None  # index to drop
    session_id: str | None = None
    unused_days: int | None = None  # stored model answers unused for this long

    def line(self) -> str:
        return f"{self.kind:<8} {self.name}  ({self.detail})"


def _store(settings: Settings) -> VectorStore | None:
    """The index, if there is one (never created just to look)."""
    return VectorStore(settings.db_dir) if (Path(settings.db_dir).expanduser() / DB_FILE).exists() else None


def _is_uuid(name: str) -> bool:
    try:
        uuid.UUID(name)
    except ValueError:
        return False
    return True


def _legacy_chroma(index_dir: Path) -> tuple[Path, ...]:
    if not (index_dir / LEGACY_CHROMA_FILE).is_file():
        return ()
    folders = [entry for entry in index_dir.iterdir() if entry.is_dir() and _is_uuid(entry.name)]
    return (index_dir / LEGACY_CHROMA_FILE, *folders)


def _last_change(folder: Path) -> datetime:
    times = [folder.stat().st_mtime, *(f.stat().st_mtime for f in folder.iterdir())]
    return datetime.fromtimestamp(max(times))


def _size_mb(*paths: Path) -> float:
    files = [f for path in paths for f in ([path] if path.is_file() else path.rglob("*")) if f.is_file()]
    return sum(f.stat().st_size for f in files) / 1e6


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
                           paths=(batch,), collection=pool_name(batch.resolve())))
    covered = {item.collection for item in stale}

    legacy = _legacy_chroma(Path(settings.db_dir).expanduser())
    if legacy:
        stale.append(Stale("index", "ChromaDB index (before 1.1)",
                           f"{_size_mb(*legacy):.1f} MB, replaced by {DB_FILE}", paths=legacy))
    store = _store(settings)
    if store:
        for collection in store.list_collections():
            folder = collection.metadata.get("folder")
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

    if Path(settings.llm_cache_db).expanduser().exists():
        unused = ResponseCache(settings.llm_cache_db).unused_since(older_than_days)
        if unused:
            stale.append(Stale("cache", f"{unused} stored model answers", f"unused for {older_than_days}+ days",
                               unused_days=older_than_days))
    return stale


def _with_wal(database: Path) -> tuple[Path, ...]:
    """A SQLite database and the write-ahead files beside it."""
    return tuple(p for p in (database, Path(f"{database}-wal"), Path(f"{database}-shm")) if p.exists())


def find_all(settings: Settings) -> list[Stale]:
    """Every piece of candidate data, to start from scratch: the indexes (resume text and
    search vectors), uploaded copies, saved sessions, stored model answers and the web app's
    reports. Only the tool's own files: never the settings (``.env``) or the downloaded
    embedding model, and never a folder a setting points at, only what the tool put in it."""
    found: list[Stale] = []
    index_dir = Path(settings.db_dir).expanduser()
    index = _with_wal(index_dir / DB_FILE) + _legacy_chroma(index_dir) if index_dir.exists() else ()
    if index:
        store = _store(settings)
        folders = len(store.list_collections()) if store else 0
        if store:
            store.close()  # counted: an open connection would write its files again on exit
        found.append(Stale("index", "every index", f"resume text and search vectors of {folders} folder(s), "
                                                   f"{_size_mb(*index):.1f} MB", paths=index))
    uploads = Path(settings.uploads_dir).expanduser()
    batches = tuple(b for b in uploads.iterdir() if b.is_dir()) if uploads.exists() else ()
    if batches:
        files = sum(1 for b in batches for f in b.rglob("*") if f.is_file())
        found.append(Stale("uploads", f"{len(batches)} upload batch(es)", f"{files} files, {_size_mb(*batches):.1f} MB",
                           paths=batches))
    saved = _with_wal(Path(settings.sessions_db).expanduser())
    if saved:
        saver = sessions.open_saver(settings.sessions_db)
        count = len(sessions.last_updated(saver))
        saver.conn.close()
        found.append(Stale("sessions", f"{count} saved session(s)", f"{_size_mb(*saved):.1f} MB", paths=saved))
    answers = _with_wal(Path(settings.llm_cache_db).expanduser())
    if answers:
        found.append(Stale("cache", "every stored model answer", f"{_size_mb(*answers):.1f} MB", paths=answers))
    reports = Path(settings.home).expanduser() / "reports"
    if reports.is_dir() and any(reports.iterdir()):
        found.append(Stale("reports", "the web app's reports", f"{_size_mb(reports):.1f} MB", paths=(reports,)))
    return found


def kept(settings: Settings) -> list[str]:
    """What ``find_all`` never removes, as lines for the recruiter."""
    home = Path(settings.home).expanduser()
    return [f"your settings ({home / '.env'})", f"the embedding model ({Path(settings.models_dir).expanduser()})"]


def remove(settings: Settings, items: list[Stale]) -> None:
    store = _store(settings) if any(item.collection for item in items) else None
    saver = sessions.open_saver(settings.sessions_db) if any(item.session_id for item in items) else None
    for item in items:
        for path in item.paths:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        if item.collection and store:
            store.delete_collection(item.collection)
        if item.session_id:
            sessions.delete_session(saver, item.session_id)
        if item.unused_days is not None:
            ResponseCache(settings.llm_cache_db).remove_unused(item.unused_days)
