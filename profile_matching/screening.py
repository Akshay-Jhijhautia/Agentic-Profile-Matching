"""Screening rounds.

Round 1 (initial screen) scores every indexed resume against the requirements
with the Milestone 2 hybrid matcher and keeps the best candidates who pass
every must-have. Unlike ``JobMatcher.match``, it covers the whole index rather
than a retrieval pool, so no qualifying candidate is missed.
"""

from __future__ import annotations

from collections.abc import Iterable

from .candidates import CandidateProfile
from .schemas import CandidateMatch, Requirements
from .services import Services


def rank_matches(matches: Iterable[CandidateMatch]) -> list[CandidateMatch]:
    """Candidates passing every must-have first, then by score; ranks numbered from 1."""
    ordered = sorted(matches, key=lambda m: (not m.passes_must_haves, -m.match_score, m.name))
    return [match.model_copy(update={"rank": rank}) for rank, match in enumerate(ordered, 1)]


def evaluate(
    services: Services, requirements: Requirements, profiles: list[CandidateProfile] | None = None
) -> list[CandidateMatch]:
    """Every indexed candidate (or just ``profiles``) scored, checked and ranked."""
    profiles = services.candidates.all() if profiles is None else profiles
    results = services.matcher.evaluate(
        requirements.to_parsed(), resume_paths=[p.resume_path for p in profiles]
    )
    return rank_matches(CandidateMatch.from_evaluation(result) for result in results)


def shortlist(ranked: list[CandidateMatch], size: int) -> list[CandidateMatch]:
    """The ``size`` best candidates that pass every must-have."""
    return [match for match in ranked if match.passes_must_haves][:size]
