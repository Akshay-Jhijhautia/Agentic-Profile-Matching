"""Runtime settings, read once from the environment and ``.env`` files.

Local data (folder indexes, uploaded resumes, saved sessions) lives under one
data home, ``~/.profile-match`` unless ``PROFILE_MATCH_HOME`` says otherwise,
so the tool works the same from any directory. ``.env`` is read from the
current directory and from the data home, in that order of precedence.

The chat model comes from ``LLM_PROVIDER`` and ``LLM_MODEL`` (see
``providers``). The older ``OPENAI_MODEL``, ``OPENAI_TEMPERATURE``,
``OPENAI_TIMEOUT`` and ``OPENAI_MAX_RETRIES`` still configure the OpenAI
provider when the ``LLM_`` names are unset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from . import providers

DEFAULT_HOME = Path.home() / ".profile-match"


def _env(*names: str | None) -> str | None:
    """The first of ``names`` set to a non-empty value."""
    return next((os.environ[name] for name in names if name and os.getenv(name)), None)


def _optional_float(*names: str | None) -> float | None:
    value = _env(*names)
    return float(value) if value is not None else None


def _flag(name: str, default: bool | None) -> bool | None:
    value = os.getenv(name)
    return default if value in (None, "") else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # openai, anthropic, google or ollama (see ``providers.PROVIDERS``).
    llm_provider: str = "openai"
    # Empty: the provider's default model.
    llm_model: str = ""
    # None keeps the provider default; newer models accept no other value.
    llm_temperature: float | None = None
    llm_timeout: float = 60.0
    llm_max_retries: int = 2
    # An OpenAI-compatible server for the openai provider, or the Ollama server.
    llm_base_url: str | None = None
    # OpenAI's Responses API, needed for tool calling with reasoning models (gpt-5.6, gpt-6).
    # None: on for OpenAI itself, off for OpenAI-compatible servers, which rarely have it.
    llm_responses_api: bool | None = None
    # Local data: one folder holding the indexes, uploads and sessions below.
    home: str = str(DEFAULT_HOME)
    db_dir: str = str(DEFAULT_HOME / "index")
    # Where resumes uploaded in chat (or given as files) are copied, one subfolder per batch.
    uploads_dir: str = str(DEFAULT_HOME / "uploads")
    # Saved chat sessions (SQLite).
    sessions_db: str = str(DEFAULT_HOME / "sessions.sqlite")
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
        home = Path(os.getenv("PROFILE_MATCH_HOME") or defaults.home).expanduser()
        provider = (os.getenv("LLM_PROVIDER") or defaults.llm_provider).strip().lower()
        # The older OPENAI_ names configure only the OpenAI provider.
        legacy = "OPENAI_{}".format if provider == "openai" else lambda _: None
        return cls(
            llm_provider=provider,
            llm_model=_env("LLM_MODEL", legacy("MODEL")) or "",  # "": the provider default
            llm_temperature=_optional_float("LLM_TEMPERATURE", legacy("TEMPERATURE")),
            llm_timeout=_optional_float("LLM_TIMEOUT", legacy("TIMEOUT")) or defaults.llm_timeout,
            llm_max_retries=int(_env("LLM_MAX_RETRIES", legacy("MAX_RETRIES")) or defaults.llm_max_retries),
            llm_base_url=_env("LLM_BASE_URL"),
            llm_responses_api=_flag("OPENAI_RESPONSES_API", defaults.llm_responses_api),
            home=str(home),
            db_dir=os.getenv("CHROMA_DB_DIR") or str(home / "index"),
            uploads_dir=os.getenv("UPLOADS_DIR") or str(home / "uploads"),
            sessions_db=os.getenv("SESSIONS_DB") or str(home / "sessions.sqlite"),
            embedder=os.getenv("EMBEDDER") or defaults.embedder,
            reports_dir=os.getenv("REPORTS_DIR") or defaults.reports_dir,
            shortlist_size=int(os.getenv("SHORTLIST_SIZE") or defaults.shortlist_size),
            screening_rounds=int(os.getenv("SCREENING_ROUNDS") or defaults.screening_rounds),
            save_reports=_flag("SAVE_REPORTS", defaults.save_reports),
        )

    def __post_init__(self) -> None:
        provider = providers.get(self.llm_provider)
        if not self.llm_model:
            if provider.default_model is None:
                raise ValueError(f"Set LLM_MODEL to the {provider.label} model to use (for example from: ollama list).")
            object.__setattr__(self, "llm_model", provider.default_model)
        if self.llm_temperature is not None and not provider.takes_temperature:
            raise ValueError(f"{provider.label} models accept no temperature; unset LLM_TEMPERATURE.")
        if not 1 <= self.screening_rounds <= 3:
            raise ValueError(f"screening_rounds must be 1, 2 or 3, got {self.screening_rounds}")
        if self.shortlist_size < 1:
            raise ValueError(f"shortlist_size must be at least 1, got {self.shortlist_size}")
        if self.embedder not in {"openai", "hashing"}:
            raise ValueError(f"embedder must be openai or hashing, got {self.embedder}")

    @property
    def responses_api(self) -> bool:
        return self.llm_responses_api if self.llm_responses_api is not None else not self.llm_base_url

    @property
    def model_label(self) -> str:
        """The chat model as people read it, e.g. "Anthropic claude-opus-5-5"."""
        return f"{providers.get(self.llm_provider).label} {self.llm_model}"


def load_env_files() -> None:
    """Load ``.env`` from the current directory, then from the data home (no overrides)."""
    load_dotenv(find_dotenv(usecwd=True))
    home = Path(os.getenv("PROFILE_MATCH_HOME") or DEFAULT_HOME).expanduser()
    load_dotenv(home / ".env")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_env_files()
    return Settings.from_env()
