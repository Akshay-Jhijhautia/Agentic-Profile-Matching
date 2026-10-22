"""Resume pools: one per folder.

Each folder of resumes gets its own ChromaDB collection, so applicants for
different jobs never mix and screening only ever ranks the active folder.
Syncing a pool is incremental: new and changed files are indexed, unchanged
ones skipped, deleted ones removed, and every file that could not be used is
reported with the reason.

Resumes given as individual files ("uploads") are copied into a batch folder
under the uploads directory, which then works like any other pool; the
originals are never touched.
"""

from __future__ import annotations

import hashlib
import re
import shlex
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path

from .candidates import CandidateDirectory
from . import llm
from .config import Settings
from .rag import job_matcher, resume_rag


def pool_name(folder: Path) -> str:
    """The collection for ``folder``: readable, and unique per absolute path."""
    slug = re.sub(r"[^a-z0-9]+", "-", folder.name.lower()).strip("-")[:40] or "resumes"
    digest = hashlib.sha1(str(folder).encode()).hexdigest()[:10]
    return f"pool-{slug}-{digest}"


def resolve_folder(folder: str | Path) -> Path:
    path = Path(folder).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resume folder not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Not a folder: {path}")
    return path


# How uploaded resumes are referred to, instead of their batch folder's path.
UPLOADS_LABEL = "your uploaded resumes"


def split_paths(text: str) -> list[str]:
    """Paths typed or dragged into a terminal: quoted, or with backslash-escaped spaces."""
    try:
        return shlex.split(text)
    except ValueError:  # an unbalanced quote: fall back to whitespace
        return text.split()


def classify(paths: list[str]) -> tuple[Path | None, list[Path]]:
    """``(folder, None)`` for one folder, ``(None, files)`` for files to upload."""
    resolved = [Path(p).expanduser().resolve() for p in paths]
    missing = [str(p) for p in resolved if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Not found: {', '.join(missing)}")
    folders = [p for p in resolved if p.is_dir()]
    if not folders:
        return None, resolved
    if len(folders) == 1 and len(resolved) == 1:
        return folders[0], []
    raise ValueError("Give one folder, or one or more resume files, but not a mix or several folders.")


def batch_for(files: list[Path], uploads_dir: str) -> Path:
    """A batch folder named after the file set, so re-running with the same files reuses its index."""
    digest = hashlib.sha1("\n".join(sorted(map(str, files))).encode()).hexdigest()[:10]
    return Path(uploads_dir).expanduser() / f"files-{digest}"


def new_batch(uploads_dir: str) -> Path:
    """A fresh batch folder, for a chat session's uploads."""
    return Path(uploads_dir).expanduser() / f"session-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def upload(files: list[Path], batch: Path) -> list[str]:
    """Copy ``files`` into ``batch``; returns the names they were stored under.

    A file already there with the same content is not copied again; a
    different file with a name already taken gets a numbered name.
    """
    batch.mkdir(parents=True, exist_ok=True)
    stored = []
    for source in files:
        target, n = batch / source.name, 2
        while target.exists() and target.read_bytes() != source.read_bytes():
            target, n = batch / f"{source.stem}-{n}{source.suffix}", n + 1
        if not target.exists():
            shutil.copy2(source, target)
        stored.append(target.name)
    return stored


@dataclass(frozen=True)
class IntakeReport:
    """What syncing a folder did, for the user."""

    folder: str
    ready: int  # resumes in the pool after the sync
    indexed: list[str]  # new or changed, (re)indexed now
    unchanged: int
    removed: list[str]
    skipped: list[dict]  # {"file", "reason"}

    def summary(self) -> str:
        changes = f"{len(self.indexed)} new or changed, {self.unchanged} unchanged"
        if self.removed:
            changes += f", {len(self.removed)} removed"
        heading = "Uploaded resumes" if self.folder == UPLOADS_LABEL else f"Resumes in {self.folder}"
        lines = [f"{heading}: {self.ready} ready ({changes})."]
        if self.skipped:
            lines.append(f"Skipped {len(self.skipped)} file(s):")
            lines += [f"- {item['file']}: {item['reason']}" for item in self.skipped]
        if not self.ready:
            lines.append("No readable resumes: add PDF, DOCX, TXT or MD files with a text layer.")
        return "\n".join(lines)


class ResumePool:
    """The indexed resumes of one folder, with their matcher and candidate lookup."""

    def __init__(self, folder: Path, rag: resume_rag.ResumeRAG):
        self.folder = folder
        self.rag = rag

    @classmethod
    def open(
        cls, folder: str | Path, settings: Settings, embedder: resume_rag.Embedder | None = None
    ) -> ResumePool:
        """The pool for ``folder`` (created empty on first use; ``sync`` fills it)."""
        path = resolve_folder(folder)
        if embedder is None and settings.embedder == "hashing":
            embedder = resume_rag.HashingEmbedder()
        # Otherwise None: a new pool uses OpenAI embeddings (created lazily, so a missing key
        # is reported by ``Services.check_ready``), an existing one the embedder that built it.
        rag = resume_rag.ResumeRAG(
            settings.db_dir, embedder=embedder, metadata_llm=llm.metadata_fallback(),
            collection_name=pool_name(path), collection_metadata={"folder": str(path)},
        )
        return cls(path, rag)

    @property
    def id(self) -> str:
        return self.rag.collection.name

    @property
    def size(self) -> int:
        return len(self.candidates.all())

    @cached_property
    def matcher(self) -> job_matcher.JobMatcher:
        return job_matcher.JobMatcher(self.rag)

    @cached_property
    def candidates(self) -> CandidateDirectory:
        return CandidateDirectory(self.rag)

    def sync(self) -> IntakeReport:
        """Index new and changed resumes, drop deleted ones, report what was skipped."""
        summary = self.rag.index_directory(str(self.folder))
        self.candidates.refresh()
        return IntakeReport(
            folder=resume_rag.display_path(str(self.folder)), ready=self.size,
            indexed=summary["indexed"], unchanged=len(summary["unchanged"]), removed=summary["removed"],
            skipped=summary["skipped"],
        )
