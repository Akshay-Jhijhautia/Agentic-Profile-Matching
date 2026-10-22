"""Runtime settings, read once from the environment (and ``.env``)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

from .rag import resume_rag


def _optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else None


@dataclass(frozen=True)
class Settings:
    llm_model: str = resume_rag.DEFAULT_LLM_MODEL
    # None leaves the provider default, for models that reject a temperature.
    llm_temperature: float | None = 0.0
    llm_timeout: float = 60.0
    llm_max_retries: int = 2
    db_dir: str = resume_rag.DEFAULT_DB_DIR

    @classmethod
    def from_env(cls) -> Settings:
        defaults = cls()
        temperature = (
            _optional_float("OPENAI_TEMPERATURE")
            if "OPENAI_TEMPERATURE" in os.environ
            else defaults.llm_temperature
        )
        return cls(
            llm_model=os.getenv("OPENAI_MODEL") or defaults.llm_model,
            llm_temperature=temperature,
            llm_timeout=_optional_float("OPENAI_TIMEOUT") or defaults.llm_timeout,
            llm_max_retries=int(os.getenv("OPENAI_MAX_RETRIES") or defaults.llm_max_retries),
            db_dir=os.getenv("CHROMA_DB_DIR") or defaults.db_dir,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings.from_env()
