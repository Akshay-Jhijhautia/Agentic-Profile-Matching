"""The three screening rounds.

1. Initial screen: every indexed resume is scored with the hybrid matcher
   (``job_matcher``) and checked against the must-haves; the best ``shortlist_size``
   candidates who pass them all go on. Unlike ``JobMatcher.match`` this covers
   the whole index, not a retrieval pool, so no qualifying candidate is missed.
2. Deep analysis: an LLM reviews each shortlisted candidate's full resume
   (the graph runs these in parallel), giving categorical verdicts with
   verbatim quotes; ``guardrails`` checks them and computes the deep score.
   The shortlist is re-ranked by the final score, which blends the deep score
   with the retrieval score.
3. Final decision: hire or no-hire by ``guardrails.decide``, a fixed rule; one
   LLM call writes the rationales. Borderline candidates then get improvement
   suggestions: what would make them a clear hire.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from typing import TypeVar

from . import guardrails, llm, prompts
from .candidates import CandidateProfile
from .schemas import (
    Assessment,
    CandidateMatch,
    DeepReview,
    HiringRationales,
    ImprovementSuggestions,
    Requirements,
)
from .services import Services

# Shortlist fields the rationale writer and the improvement advisor see per candidate.
COHORT_FIELDS = {"id", "name", "final_score", "match_score", "passes_must_haves", "assessment", "recommendation"}

T = TypeVar("T")


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


def assessment_key(pool_id: str, requirements: Requirements, candidate_id: str) -> str:
    """Cache key of an assessment: the pool (ids repeat across folders), requirements and candidate."""
    return f"{pool_id}:{requirements_key(requirements)}:{candidate_id}"


def assess(services: Services, requirements: Requirements, match: CandidateMatch) -> Assessment:
    """Round 2 for one candidate: the model reviews the full resume, the guardrails
    check its verdicts and compute the deep score."""
    profile = services.candidates.resolve(match.id)
    resume_text = services.candidates.resume_text(profile)
    items = guardrails.review_items(requirements)
    payload = {
        "requirements": guardrails.items_for_prompt(items),
        "first_round": match.model_dump(include={
            "must_have_checks", "matched_skills", "missing_must_haves", "experience_years", "highest_degree",
        }),
        "resume_text": resume_text,
    }
    review = llm.structured(
        DeepReview, prompts.DEEP_ASSESSMENT, json.dumps(payload, ensure_ascii=False), model=services.model,
        cache=services.llm_cache,
    )
    return guardrails.build_assessment(review, items, match, resume_text)


def apply_assessments(
    pool_id: str,
    shortlisted: list[CandidateMatch],
    requirements: Requirements,
    assessments: dict[str, Assessment],
) -> list[CandidateMatch]:
    """Attach each candidate's assessment and re-rank by final score."""
    return rank_matches(
        match.model_copy(update={"assessment": assessments.get(assessment_key(pool_id, requirements, match.id))})
        for match in shortlisted
    )


# --- round 3 ------------------------------------------------------------------------------------


def _cohort_payload(requirements: Requirements, matches: list[CandidateMatch]) -> str:
    candidates = [
        match.model_dump(include=COHORT_FIELDS) | {"unverified_must_haves": match.unverified}
        for match in matches
    ]
    return json.dumps({"requirements": requirements.model_dump(), "candidates": candidates}, ensure_ascii=False)


def by_candidate(services: Services, items: Iterable[T], ref: Callable[[T], str]) -> dict[str, T]:
    """LLM answers keyed by candidate id; the LLM sometimes names a candidate instead,
    and answers about unknown candidates are dropped."""
    found = {}
    for item in items:
        profile = services.candidates.find(ref(item))
        if profile is not None:
            found[profile.id] = item
    return found


def decide(services: Services, requirements: Requirements, shortlisted: list[CandidateMatch]) -> list[CandidateMatch]:
    """Round 3: the decision rule decides hire or no-hire for every assessed candidate;
    one model call then explains each decision (the rule's outcome stays if it skips one)."""
    decided = [
        match.model_copy(update={"recommendation": guardrails.decide(match)}) if match.assessment else match
        for match in shortlisted
    ]
    with_decision = [match for match in decided if match.recommendation is not None]
    if not with_decision:
        return decided
    payload = _cohort_payload(requirements, with_decision)
    result = llm.structured(HiringRationales, prompts.HIRING_RATIONALES, payload, model=services.model, cache=services.llm_cache)
    rationales = by_candidate(services, result.rationales, lambda r: r.candidate_id)
    return [
        match.model_copy(update={"recommendation": match.recommendation.model_copy(
            update={"rationale": rationales[match.id].rationale}
        )})
        if match.id in rationales and match.recommendation is not None else match
        for match in decided
    ]


def suggest_improvements(
    services: Services, requirements: Requirements, shortlisted: list[CandidateMatch]
) -> list[CandidateMatch]:
    """Improvement suggestions for the borderline candidates, in one call; others unchanged."""
    borderline = [match for match in shortlisted if match.borderline]
    if not borderline:
        return shortlisted
    payload = _cohort_payload(requirements, borderline)
    result = llm.structured(ImprovementSuggestions, prompts.IMPROVEMENT_SUGGESTIONS, payload, model=services.model, cache=services.llm_cache)
    suggestions = by_candidate(services, result.candidates, lambda c: c.candidate_id)
    return [
        match.model_copy(update={"improvements": suggestions[match.id].improvements})
        if match.borderline and match.id in suggestions else match
        for match in shortlisted
    ]
