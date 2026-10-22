"""The shared dependencies of tools and graph nodes, created once per session.

Tools are built by factories that close over a ``Services`` instance, so
nothing reaches for module-level globals and tests swap in a hashing index
and a fake chat model by constructing ``Services`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property

from langchain_core.language_models import BaseChatModel

from .candidates import CandidateDirectory
from .config import Settings, get_settings
from .rag import job_matcher, resume_rag


@dataclass
class Services:
    rag: resume_rag.ResumeRAG
    settings: Settings = field(default_factory=get_settings)
    # None means ``llm.get_chat_model()``, the configured OpenAI model.
    model: BaseChatModel | None = None

    @cached_property
    def matcher(self) -> job_matcher.JobMatcher:
        return job_matcher.JobMatcher(self.rag)

    @cached_property
    def candidates(self) -> CandidateDirectory:
        return CandidateDirectory(self.rag)

    def check_ready(self) -> None:
        """Raise ``ValueError`` with a fix when the index is empty or the API key is missing."""
        if self.rag.collection.count() == 0:
            raise ValueError(
                f"The index at '{self.rag.persist_dir}' is empty. Run: python main.py index resumes"
            )
        chat_needs_key = self.model is None and not os.getenv("OPENAI_API_KEY")
        if chat_needs_key or resume_rag.missing_api_key(self.rag.embedder_name):
            raise ValueError("OPENAI_API_KEY is not set. Add it to .env (see .env.example).")

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Services:
        settings = settings or get_settings()
        return cls(rag=resume_rag.ResumeRAG(settings.db_dir), settings=settings)
