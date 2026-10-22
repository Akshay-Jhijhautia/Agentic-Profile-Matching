"""The shared dependencies of tools and graph nodes, created once per session.

Tools are built by factories that close over a ``Services`` instance, so
nothing reaches for module-level globals and tests swap in a hashing index
and a fake chat model by constructing ``Services`` directly.

``Services`` screens one resume pool at a time (see ``pools``): ``use_folder``
syncs a folder and makes it the active pool, ``use_paths`` also accepts
resume files (uploaded into a batch folder); ``rag``, ``matcher`` and
``candidates`` always refer to the active pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from . import pools, providers
from .candidates import CandidateDirectory
from .config import Settings, get_settings
from .pools import IntakeReport, ResumePool
from .rag import job_matcher, resume_rag


@dataclass
class Services:
    pool: ResumePool | None = None
    settings: Settings = field(default_factory=get_settings)
    # None means ``llm.get_chat_model()``, the configured provider's model.
    model: BaseChatModel | None = None

    def _active(self) -> ResumePool:
        if self.pool is None:
            raise ValueError("No resume folder is loaded. Point the agent at a folder of resumes first.")
        return self.pool

    @property
    def rag(self) -> resume_rag.ResumeRAG:
        return self._active().rag

    @property
    def matcher(self) -> job_matcher.JobMatcher:
        return self._active().matcher

    @property
    def candidates(self) -> CandidateDirectory:
        return self._active().candidates

    def use_folder(self, folder: str | Path, embedder: resume_rag.Embedder | None = None) -> IntakeReport:
        """Sync ``folder``'s pool (indexing what changed) and make it the active pool."""
        pool = ResumePool.open(folder, self.settings, embedder=embedder)
        report = pool.sync()
        self.pool = pool
        return report

    def use_paths(self, paths: list[str], batch: str | Path | None = None) -> tuple[IntakeReport, Path | None]:
        """Activate one folder, or upload resume files and activate their batch.

        Files go into ``batch`` (a chat session's uploads, added to over time)
        or, without one, a batch named after the file set. Returns the report
        and the batch folder used for uploads (None for a folder).
        """
        folder, files = pools.classify(paths)
        if folder is not None:
            return self.use_folder(folder), None
        target = Path(batch) if batch else pools.batch_for(files, self.settings.uploads_dir)
        pools.upload(files, target)
        report = self.use_folder(target)
        return replace(report, folder=pools.UPLOADS_LABEL), target

    def check_ready(self, require_resumes: bool = True) -> None:
        """Raise ``ValueError`` with a fix when an API key is missing (the chat model's, or
        OpenAI's for an index built with OpenAI embeddings) or, if required, when no
        readable resumes are loaded."""
        if self.model is None:
            key = providers.missing_key(providers.required_key_vars(self.settings))
            if key:
                raise ValueError(providers.missing_key_message(key, f"needed for {self.settings.model_label}"))
        if self.pool is None and not require_resumes:
            return
        pool = self._active()
        if pool.rag.collection.count() == 0:
            raise ValueError(f"No readable resumes in {resume_rag.display_path(str(pool.folder))}.")
        if resume_rag.missing_api_key(pool.rag.embedder_name):
            raise ValueError(providers.missing_key_message(
                "OPENAI_API_KEY", "this folder's index uses OpenAI embeddings; EMBEDDER=hashing avoids them for new folders"
            ))

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Services:
        """Services with no pool loaded yet (call ``use_folder``)."""
        return cls(settings=settings or get_settings())
