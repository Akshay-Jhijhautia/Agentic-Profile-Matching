"""The three screening rounds.

1. Initial screen: every indexed resume is scored with the hybrid matcher
   (``job_matcher``) and checked against the must-haves; the best ``shortlist_size``
   candidates who pass them all go on. Unlike ``JobMatcher.match`` this covers
   the whole index, not a retrieval pool, so no qualifying candidate is missed.
2. Deep analysis: an LLM reviews each shortlisted candidate's full resume
   (the graph runs these in parallel), blind to who they are (no name, contact
   details or personal fields; see ``candidates.blind_resume``), giving categorical verdicts with
   verbatim quotes; ``guardrails`` checks them and computes the deep score.
   The shortlist is re-ranked by the final score, which blends the deep score
   with the retrieval score.
3. Recommendation: interview or not now, by ``guardrails.decide``, a fixed rule; one
   LLM call writes the rationales. Borderline candidates then get improvement
   suggestions: what would make them a clear interview.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from . import guardrails, llm, prompts
from .anonymous import Anonymous
from .candidates import CandidateProfile
from .schemas import (
    Assessment,
    CandidateMatch,
    DeepReview,
    DecisionRationales,
    ImprovementSuggestions,
    Requirements,
)
from .services import Services

# Shortlist fields the rationale writer and the improvement advisor see per candidate
# (never the id or name: they are anonymous, see ``anonymous``).
COHORT_FIELDS = {"final_score", "match_score", "passes_must_haves", "assessment", "recommendation"}


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
    resume_text = services.candidates.blind_text(profile)
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


def _cohort_payload(requirements: Requirements, matches: list[CandidateMatch], anonymous: Anonymous) -> str:
    candidates = [
        {"candidate_id": anonymous.label(match.id)} | match.model_dump(include=COHORT_FIELDS)
        | {"unverified_must_haves": match.unverified}
        for match in matches
    ]
    return json.dumps({"requirements": requirements.model_dump(), "candidates": candidates}, ensure_ascii=False)


def decide(services: Services, requirements: Requirements, shortlisted: list[CandidateMatch]) -> list[CandidateMatch]:
    """Round 3: the decision rule recommends an interview, or not, for every assessed candidate;
    one model call then explains each decision (the rule's outcome stays if it skips one)."""
    decided = [
        match.model_copy(update={"recommendation": guardrails.decide(match)}) if match.assessment else match
        for match in shortlisted
    ]
    with_decision = [match for match in decided if match.recommendation is not None]
    if not with_decision:
        return decided
    anonymous = Anonymous(with_decision)
    payload = _cohort_payload(requirements, with_decision, anonymous)
    result = llm.structured(DecisionRationales, prompts.DECISION_RATIONALES, payload, model=services.model, cache=services.llm_cache)
    # Answers about labels the model made up are dropped.
    rationales = {anonymous.id_of(r.candidate_id): anonymous.named(r.rationale) for r in result.rationales}
    return [
        match.model_copy(update={"recommendation": match.recommendation.model_copy(
            update={"rationale": rationales[match.id]}
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
    anonymous = Anonymous(borderline)
    payload = _cohort_payload(requirements, borderline, anonymous)
    result = llm.structured(ImprovementSuggestions, prompts.IMPROVEMENT_SUGGESTIONS, payload, model=services.model, cache=services.llm_cache)
    suggestions = {
        anonymous.id_of(c.candidate_id): [
            i.model_copy(update={field: anonymous.named(getattr(i, field)) for field in ("gap", "suggestion", "how_to_verify")})
            for i in c.improvements
        ]
        for c in result.candidates
    }
    return [
        match.model_copy(update={"improvements": suggestions[match.id]})
        if match.borderline and match.id in suggestions else match
        for match in shortlisted
    ]
