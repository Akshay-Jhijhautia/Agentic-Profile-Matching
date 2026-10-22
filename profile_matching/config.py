"""Runtime settings, read once from the environment (and ``.env``)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv

from .rag import resume_rag


def _optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else None


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value in (None, "") else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_model: str = resume_rag.DEFAULT_LLM_MODEL
    # None keeps the provider default; newer OpenAI models accept no other value.
    llm_temperature: float | None = None
    llm_timeout: float = 60.0
    llm_max_retries: int = 2
    # OpenAI's Responses API: needed for tool calling with reasoning models (gpt-5.6, gpt-6).
    llm_responses_api: bool = True
    db_dir: str = resume_rag.DEFAULT_DB_DIR
    # Where resumes uploaded in chat (or given as files) are copied, one subfolder per batch.
    uploads_dir: str = str(Path.home() / ".profile-match" / "uploads")
    # "openai" (default) or "hashing" (offline, keyword overlap only) for new resume pools.
    embedder: str = "openai"
    # The only directory the agent may write to (reports, summaries).
    reports_dir: str = "reports"
    # Candidates kept after the initial screen.
    shortlist_size: int = 10
    # 1: retrieval only; 2: + deep analysis of the shortlist; 3: + hire/no-hire decisions.
    screening_rounds: int = 3
    # Save the shortlist and per-candidate reports under reports_dir after every screening.
    save_reports: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        defaults = cls()
        return cls(
            llm_model=os.getenv("OPENAI_MODEL") or defaults.llm_model,
            llm_temperature=_optional_float("OPENAI_TEMPERATURE"),
            llm_timeout=_optional_float("OPENAI_TIMEOUT") or defaults.llm_timeout,
            llm_max_retries=int(os.getenv("OPENAI_MAX_RETRIES") or defaults.llm_max_retries),
            llm_responses_api=_flag("OPENAI_RESPONSES_API", defaults.llm_responses_api),
            db_dir=os.getenv("CHROMA_DB_DIR") or defaults.db_dir,
            uploads_dir=os.getenv("UPLOADS_DIR") or defaults.uploads_dir,
            embedder=os.getenv("EMBEDDER") or defaults.embedder,
            reports_dir=os.getenv("REPORTS_DIR") or defaults.reports_dir,
            shortlist_size=int(os.getenv("SHORTLIST_SIZE") or defaults.shortlist_size),
            screening_rounds=int(os.getenv("SCREENING_ROUNDS") or defaults.screening_rounds),
            save_reports=_flag("SAVE_REPORTS", defaults.save_reports),
        )

    def __post_init__(self) -> None:
        if not 1 <= self.screening_rounds <= 3:
            raise ValueError(f"screening_rounds must be 1, 2 or 3, got {self.screening_rounds}")
        if self.shortlist_size < 1:
            raise ValueError(f"shortlist_size must be at least 1, got {self.shortlist_size}")
        if self.embedder not in {"openai", "hashing"}:
            raise ValueError(f"embedder must be openai or hashing, got {self.embedder}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings.from_env()
