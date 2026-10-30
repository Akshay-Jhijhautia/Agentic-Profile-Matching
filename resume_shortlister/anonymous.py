"""Candidates as a model that judges them sees them: "Candidate 1", "Candidate 2"…

Ids are file names ("anika_laurent") and so carry the name; in blind screening the model
gets a label instead, with the resume text stripped of who the candidate is
(``candidates.blind_resume``). Its answers are mapped back: references to ids, and the
labels in its text to names, so the recruiter reads "Anika Laurent", never "Candidate 1".
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .schemas import CandidateMatch

_LABEL_RE = re.compile(r"\bcandidate[ _-]?(\d+)\b", re.IGNORECASE)


class Anonymous:
    def __init__(self, candidates: Iterable[CandidateMatch | tuple[str, str]]):
        """``candidates``: matches, or ``(id, name)`` pairs, numbered in the given order."""
        pairs = [(c.id, c.name) if isinstance(c, CandidateMatch) else c for c in candidates]
        self._ids = [candidate_id for candidate_id, _ in pairs]
        self._names = {candidate_id: name for candidate_id, name in pairs}

    def label(self, candidate_id: str) -> str:
        return f"Candidate {self._ids.index(candidate_id) + 1}"

    def id_of(self, ref: str) -> str | None:
        """The id behind a label as the model wrote it ("Candidate 2", "candidate_2"), or None."""
        match = _LABEL_RE.fullmatch(ref.strip())
        number = int(match[1]) if match else None
        return self._ids[number - 1] if number and number <= len(self._ids) else None

    def named(self, text: str) -> str:
        """``text`` with every label replaced by the candidate's name."""

        def name(match: re.Match) -> str:
            candidate_id = self.id_of(match[0])
            return self._names[candidate_id] if candidate_id else match[0]

        return _LABEL_RE.sub(name, text)
