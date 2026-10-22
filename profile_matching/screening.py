"""The three screening rounds.

1. Initial screen: every indexed resume is scored with the Milestone 2 hybrid
   matcher and checked against the must-haves; the best ``shortlist_size``
   candidates who pass them all go on. Unlike ``JobMatcher.match`` this covers
   the whole index, not a retrieval pool, so no qualifying candidate is missed.
2. Deep analysis: an LLM reviews each shortlisted candidate's full resume
   (the graph runs these in parallel) and the shortlist is re-ranked by the
   final score, which blends the deep score with the retrieval score.
3. Final decision: one committee call decides hire or no-hire for the whole
   shortlist, so decisions are calibrated against each other.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from . import llm, prompts
from .candidates import CandidateProfile
from .schemas import Assessment, CandidateMatch, HiringDecisions, Recommendation, Requirements
from .services import Services

# Final score at or above which a candidate the committee skipped defaults to hire.
FALLBACK_HIRE_SCORE = 75


def rank_matches(matches: Iterable[CandidateMatch]) -> list[CandidateMatch]:
    """Candidates passing every must-have first, then by final score; ranks numbered from 1."""
    ordered = sorted(matches, key=lambda m: (not m.passes_must_haves, -m.final_score, m.name))
    return [match.model_copy(update={"rank": rank}) for rank, match in enumerate(ordered, 1)]


# --- round 1 ------------------------------------------------------------------------------------


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


# --- round 2 ------------------------------------------------------------------------------------


def requirements_key(requirements: Requirements) -> str:
    """Identifies what an assessment depends on: the hard requirements, not their wording."""
    basis = requirements.model_dump(include={"must_have", "nice_to_have", "min_total_years", "min_degree"})
    return hashlib.sha1(json.dumps(basis, sort_keys=True).encode()).hexdigest()[:12]


def assessment_key(requirements: Requirements, candidate_id: str) -> str:
    return f"{requirements_key(requirements)}:{candidate_id}"


def assess(services: Services, requirements: Requirements, match: CandidateMatch) -> Assessment:
    """Round 2 for one candidate: a deep review of their full resume."""
    profile = services.candidates.resolve(match.id)
    payload = {
        "requirements": requirements.model_dump(),
        "first_round": match.model_dump(include={
            "match_score", "must_have_checks", "matched_skills", "missing_must_haves",
            "experience_years", "highest_degree",
        }),
        "resume_text": services.candidates.resume_text(profile),
    }
    return llm.structured(
        Assessment, prompts.DEEP_ASSESSMENT, json.dumps(payload, ensure_ascii=False), model=services.model
    )


def apply_assessments(
    shortlisted: list[CandidateMatch], requirements: Requirements, assessments: dict[str, Assessment]
) -> list[CandidateMatch]:
    """Attach each candidate's assessment and re-rank by final score."""
    return rank_matches(
        match.model_copy(update={"assessment": assessments.get(assessment_key(requirements, match.id))})
        for match in shortlisted
    )


# --- round 3 ------------------------------------------------------------------------------------


def _fallback(match: CandidateMatch) -> Recommendation:
    """Used only when the committee's answer leaves a candidate out."""
    decision = "hire" if match.final_score >= FALLBACK_HIRE_SCORE else "no_hire"
    return Recommendation(decision=decision, confidence="low",
                          rationale=f"No committee decision; based on the final score of {match.final_score}.")


def decide(services: Services, requirements: Requirements, shortlisted: list[CandidateMatch]) -> list[CandidateMatch]:
    """Round 3: hire or no-hire for every shortlisted candidate, in one committee call."""
    if not shortlisted:
        return []
    candidates = [
        match.model_dump(include={"id", "name", "final_score", "match_score", "passes_must_haves", "assessment"})
        | {"unverified_must_haves": match.unverified}
        for match in shortlisted
    ]
    payload = json.dumps({"requirements": requirements.model_dump(), "candidates": candidates}, ensure_ascii=False)
    result = llm.structured(HiringDecisions, prompts.HIRING_DECISIONS, payload, model=services.model)

    by_id = {}
    for decision in result.decisions:
        profile = services.candidates.find(decision.candidate_id)  # the LLM may answer with a name
        if profile is not None:
            by_id[profile.id] = Recommendation(**decision.model_dump(exclude={"candidate_id"}))
    return [
        match.model_copy(update={"recommendation": by_id.get(match.id) or _fallback(match)})
        for match in shortlisted
    ]
