"""Look up indexed candidates by id or name.

A candidate's id is its resume file stem without the ``resume_`` prefix
(``resumes/resume_maria_garcia.pdf`` -> ``maria_garcia``). Users and the LLM
refer to candidates loosely ("Maria", "maria garcia", "Garcia"), so
``CandidateDirectory.resolve`` accepts an id, a full name or a unique part of
a name, and explains ambiguous or unknown references instead of guessing.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

from .rag import fs_tools, resume_rag

_SEPARATORS_RE = re.compile(r"[\s_\-]+")
# Resume text sent to the LLM, enough for any realistic resume.
MAX_RESUME_CHARS = 6_000


class CandidateNotFoundError(LookupError):
    """No candidate, or more than one, matches a reference."""


def candidate_id(resume_path: str) -> str:
    return Path(resume_path).stem.removeprefix("resume_")


def _normalize(text: str) -> str:
    """Lower-case words: "resumes/resume_Maria-Garcia.pdf" -> "maria garcia"."""
    text = text.strip()
    if Path(text).suffix.lower() in fs_tools.SUPPORTED_READ_EXTENSIONS:
        text = Path(text).stem
    return _SEPARATORS_RE.sub(" ", text.lower()).strip().removeprefix("resume ")


@dataclass(frozen=True)
class CandidateProfile:
    id: str
    name: str
    resume_path: str  # absolute, as stored in the index
    metadata: dict  # ``resume_rag.parse_record`` fields

    def summary(self) -> dict:
        """Compact facts for tool output and prompts."""
        return {
            "id": self.id,
            "name": self.name,
            "resume": resume_rag.display_path(self.resume_path),
            "experience_years": self.metadata["experience_years"],
            "highest_degree": self.metadata["highest_degree"],
            "skills": self.metadata["skills"],
        }


class CandidateDirectory:
    """Every candidate in a ``ResumeRAG`` index, loaded once on first use."""

    def __init__(self, rag: resume_rag.ResumeRAG):
        self._rag = rag
        self._profiles: list[CandidateProfile] | None = None

    def all(self) -> list[CandidateProfile]:
        if self._profiles is None:
            self._profiles = [
                CandidateProfile(candidate_id(meta["resume_path"]), meta["candidate_name"],
                                 meta["resume_path"], meta)
                for meta in self._rag.list_resumes()
            ]
        return self._profiles

    def refresh(self) -> None:
        """Forget the cached profiles (after re-indexing)."""
        self._profiles = None

    def resolve(self, ref: str) -> CandidateProfile:
        """The one candidate ``ref`` names: an id, a full name or a unique part of a name."""
        key = _normalize(ref)
        if not key:
            raise CandidateNotFoundError("Candidate reference must be a non-empty string")
        profiles = self.all()
        exact = [p for p in profiles if key in (_normalize(p.id), _normalize(p.name))]
        if len(exact) == 1:
            return exact[0]
        words = set(key.split())
        partial = exact or [p for p in profiles if words <= set(_normalize(p.name).split())]
        if len(partial) == 1:
            return partial[0]
        if partial:
            options = ", ".join(f"{p.name} ({p.id})" for p in partial)
            raise CandidateNotFoundError(f"'{ref}' matches several candidates: {options}. Use an id.")
        close = difflib.get_close_matches(key, [_normalize(p.name) for p in profiles], n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise CandidateNotFoundError(f"No candidate matches '{ref}'.{hint}")

    def find(self, ref: str) -> CandidateProfile | None:
        """``resolve``, but None instead of an error (for ids written by the LLM)."""
        try:
            return self.resolve(ref)
        except CandidateNotFoundError:
            return None

    def resolve_many(self, refs: list[str]) -> list[CandidateProfile]:
        """Resolve each reference, dropping repeats but keeping the given order."""
        return list({p.id: p for p in map(self.resolve, refs)}.values())

    @staticmethod
    def resume_text(profile: CandidateProfile, max_chars: int | None = MAX_RESUME_CHARS) -> str:
        """The resume's text, cut to ``max_chars`` (None for all of it)."""
        result = fs_tools.read_file(profile.resume_path)
        if not result["success"]:
            raise FileNotFoundError(f"Could not read {profile.name}'s resume: {result['error']}")
        return result["content"][:max_chars]
