"""A small persistent vector store: SQLite for the records, NumPy for search.

Every resume pool is a collection of chunks. A chunk row holds its id, its
resume's path, the chunk text, a JSON metadata record and its embedding as
float32 bytes, all in one SQLite file. Search is exact cosine similarity over
a collection's vectors, held as one normalised NumPy matrix that is rebuilt
only when the collection changes (in this process or, through SQLite's
``data_version``, in another one).

Exact search over a few tens of thousands of chunks takes milliseconds, so an
approximate index, with its tuning and its occasional missed neighbours, is
not needed: results are the true nearest chunks, every time.

One ``VectorStore`` may be used from several threads (tool calls and the
round-2 fan-out run in parallel); a lock serialises access to the connection.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DB_FILE = "index.sqlite"
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    name TEXT PRIMARY KEY,
    metadata TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    collection TEXT NOT NULL,
    id TEXT NOT NULL,
    resume_path TEXT NOT NULL,
    document TEXT NOT NULL,
    metadata TEXT NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (collection, id)
);
CREATE INDEX IF NOT EXISTS chunks_by_resume ON chunks (collection, resume_path);
"""


@dataclass(frozen=True)
class Chunk:
    id: str
    document: str
    metadata: dict


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    similarity: float  # cosine similarity, -1..1


class VectorStore:
    """The collections stored in ``directory``/``DB_FILE``."""

    def __init__(self, directory: str | Path):
        folder = Path(directory).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / DB_FILE
        self._lock = threading.RLock()
        # Commits through this connection, which data_version does not count.
        self.writes = 0
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        with self._lock:
            # WAL: readers in other processes (another chat) never block on a writer.
            self._db.execute("PRAGMA journal_mode=WAL")
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise ValueError(f"{self.path} was written by a newer profile-match; upgrade to use it.")
            self._db.executescript(_SCHEMA)
            self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")
            self.writes += 1

    def _query(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    def version(self) -> tuple[int, int]:
        """Changes whenever anything commits: this connection or another one."""
        return self._query("PRAGMA data_version")[0][0], self.writes

    # --- collections ---

    def get_collection(self, name: str) -> Collection | None:
        rows = self._query("SELECT metadata FROM collections WHERE name = ?", (name,))
        return Collection(self, name, json.loads(rows[0][0])) if rows else None

    def create_collection(self, name: str, metadata: dict) -> Collection:
        with self._transaction() as db:
            db.execute("INSERT INTO collections (name, metadata) VALUES (?, ?)", (name, json.dumps(metadata)))
        return Collection(self, name, dict(metadata))

    def delete_collection(self, name: str) -> None:
        with self._transaction() as db:
            db.execute("DELETE FROM chunks WHERE collection = ?", (name,))
            db.execute("DELETE FROM collections WHERE name = ?", (name,))

    def list_collections(self) -> list[Collection]:
        rows = self._query("SELECT name, metadata FROM collections ORDER BY name")
        return [Collection(self, name, json.loads(metadata)) for name, metadata in rows]

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __del__(self) -> None:
        # A pool replaced by another (/resumes) drops its store; close rather than leak the connection.
        db = getattr(self, "_db", None)
        if db is not None:
            db.close()


class Collection:
    """One collection's chunks. Records come back in insertion order."""

    def __init__(self, store: VectorStore, name: str, metadata: dict):
        self.store = store
        self.name = name
        self.metadata = metadata
        # (store version, chunks, unit vectors), rebuilt when the store version changes.
        self._matrix: tuple[tuple[int, int], list[Chunk], np.ndarray] | None = None

    def count(self) -> int:
        return self.store._query("SELECT COUNT(*) FROM chunks WHERE collection = ?", (self.name,))[0][0]

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Embeddings must all have the same length")
        dimension = self._dimension()
        if dimension is not None and vectors.shape[1] != dimension:
            raise ValueError(f"Embeddings have {vectors.shape[1]} dimensions; this collection has {dimension}")
        rows = [
            (self.name, chunk.id, chunk.metadata["resume_path"], chunk.document, json.dumps(chunk.metadata),
             vector.tobytes())
            for chunk, vector in zip(chunks, vectors)
        ]
        with self.store._transaction() as db:
            db.executemany(
                "INSERT INTO chunks (collection, id, resume_path, document, metadata, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?)", rows,
            )

    def delete(self, resume_path: str) -> None:
        """Remove every chunk of ``resume_path``."""
        with self.store._transaction() as db:
            db.execute("DELETE FROM chunks WHERE collection = ? AND resume_path = ?", (self.name, resume_path))

    def update_metadata(self, resume_path: str, changes: dict) -> None:
        """Merge ``changes`` into the metadata of every chunk of ``resume_path``."""
        with self.store._transaction() as db:
            rows = db.execute(
                "SELECT id, metadata FROM chunks WHERE collection = ? AND resume_path = ?", (self.name, resume_path)
            ).fetchall()
            db.executemany(
                "UPDATE chunks SET metadata = ? WHERE collection = ? AND id = ?",
                [(json.dumps(json.loads(metadata) | changes), self.name, chunk_id) for chunk_id, metadata in rows],
            )

    def chunks(self, resume_path: str | None = None, limit: int | None = None) -> list[Chunk]:
        """Stored chunks, of one resume or of all."""
        sql = "SELECT id, document, metadata FROM chunks WHERE collection = ?"
        params: tuple = (self.name,)
        if resume_path is not None:
            sql, params = sql + " AND resume_path = ?", params + (resume_path,)
        sql += " ORDER BY rowid"
        if limit is not None:
            sql, params = sql + " LIMIT ?", params + (limit,)
        return [Chunk(chunk_id, document, json.loads(metadata)) for chunk_id, document, metadata
                in self.store._query(sql, params)]

    def query(self, embedding: list[float], n_results: int, resume_paths: list[str] | None = None) -> list[Hit]:
        """The ``n_results`` chunks most similar to ``embedding`` (exact cosine), best first,
        optionally only among the chunks of ``resume_paths``."""
        chunks, matrix = self._load()
        if not chunks or n_results <= 0:
            return []
        query = np.asarray(embedding, dtype=np.float32)
        if query.shape != (matrix.shape[1],):
            raise ValueError(f"Query has {query.size} dimensions; this collection has {matrix.shape[1]}")
        norm = np.linalg.norm(query)
        similarities = matrix @ (query / norm if norm else query)
        if resume_paths is not None:
            allowed = set(resume_paths)
            keep = np.fromiter((c.metadata["resume_path"] in allowed for c in chunks), bool, len(chunks))
            similarities = np.where(keep, similarities, -np.inf)
        # Stable sort: equal similarities keep insertion order, so results never flip between runs.
        order = np.argsort(-similarities, kind="stable")[:n_results]
        return [Hit(chunks[i], float(similarities[i])) for i in order if similarities[i] != -np.inf]

    # --- the search matrix ---

    def _dimension(self) -> int | None:
        rows = self.store._query("SELECT length(embedding) FROM chunks WHERE collection = ? LIMIT 1", (self.name,))
        return rows[0][0] // np.dtype(np.float32).itemsize if rows else None

    def _load(self) -> tuple[list[Chunk], np.ndarray]:
        """Every chunk and its unit-length vector, cached until the collection changes."""
        key = self.store.version()
        if self._matrix is None or self._matrix[0] != key:
            rows = self.store._query(
                "SELECT id, document, metadata, embedding FROM chunks WHERE collection = ? ORDER BY rowid",
                (self.name,),
            )
            chunks = [Chunk(chunk_id, document, json.loads(metadata)) for chunk_id, document, metadata, _ in rows]
            matrix = np.array([np.frombuffer(row[3], dtype=np.float32) for row in rows], dtype=np.float32)
            if rows:
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                matrix = matrix / np.where(norms == 0, 1, norms)
            self._matrix = (key, chunks, matrix)
        return self._matrix[1], self._matrix[2]
