"""The shared dependencies of tools and graph nodes, created once per session.

Tools are built by factories that close over a ``Services`` instance, so
nothing reaches for module-level globals and tests swap in a hashing index
and a fake chat model by constructing ``Services`` directly.
"""

from __future__ import annotations

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

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Services:
        settings = settings or get_settings()
        return cls(rag=resume_rag.ResumeRAG(settings.db_dir), settings=settings)
