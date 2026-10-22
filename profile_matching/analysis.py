"""Candidate analysis behind the agent's tools: head-to-head comparison and
interview questions.

Each analysis has a deterministic part (facts, scores, must-have checks from
the Milestone 2 matcher) and an LLM part that reasons over those facts and the
resume text. The facts are returned alongside the LLM output, so every claim
the agent makes can be traced back to a number or a resume line.
"""

from __future__ import annotations

import json

from . import llm, prompts, screening
from .candidates import CandidateProfile
from .schemas import ComparisonVerdict, InterviewQuestions, Requirements
from .services import Services

COMPARE_LIMITS = (2, 5)
QUESTION_LIMITS = (3, 10)


def as_requirements(value: Requirements | dict | None) -> Requirements | None:
    """Requirements from graph state, which may hold a model or its dict form."""
    if value is None or isinstance(value, Requirements):
        return value
    return Requirements.model_validate(value)


def _check_range(name: str, value: int, limits: tuple[int, int]) -> None:
    low, high = limits
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")


# Match fields shown next to a candidate's profile in comparisons and prompts.
FACT_FIELDS = {
    "rank", "match_score", "passes_must_haves", "must_have_checks", "matched_skills",
    "missing_must_haves", "score_breakdown", "reasoning",
}


def candidate_facts(
    services: Services, profiles: list[CandidateProfile], requirements: Requirements | None
) -> list[dict]:
    """Each profile's summary and, given requirements, how it matches them.

    With requirements, candidates are ranked among themselves (see
    ``screening.rank_matches``). Without, the given order is kept.
    """
    facts = {profile.id: profile.summary() for profile in profiles}
    if requirements is None:
        return list(facts.values())
    matches = screening.evaluate(services, requirements, profiles)
    return [facts[m.id] | m.model_dump(include=FACT_FIELDS) for m in matches]


def _llm_payload(requirements: Requirements | None, **sections) -> str:
    body = {"requirements": requirements.model_dump() if requirements else None} | sections
    return json.dumps(body, ensure_ascii=False)


def _to_id(services: Services, ref: str) -> str | None:
    """The id the LLM meant (it sometimes answers with a name), or None if unknown."""
    profile = services.candidates.find(ref)
    return profile.id if profile else None


def compare_candidates(
    services: Services, refs: list[str], requirements: Requirements | None = None
) -> dict:
    """Head-to-head comparison of 2-5 candidates, for the role when one is set."""
    profiles = services.candidates.resolve_many(refs)
    _check_range("Number of distinct candidates", len(profiles), COMPARE_LIMITS)
    facts = candidate_facts(services, profiles, requirements)
    by_id = {p.id: p for p in profiles}
    payload = _llm_payload(
        requirements,
        candidates=[fact | {"resume_text": services.candidates.resume_text(by_id[fact["id"]])} for fact in facts],
    )
    verdict = llm.structured(ComparisonVerdict, prompts.COMPARE_CANDIDATES, payload, model=services.model)

    # Keep the LLM's order for the ids it named, then any it left out.
    named = [i for i in dict.fromkeys(_to_id(services, ref) for ref in verdict.ranking) if i in by_id]
    ranking = named + [p.id for p in profiles if p.id not in named]
    notes = [
        note.model_dump() | {"candidate_id": _to_id(services, note.candidate_id) or note.candidate_id}
        for note in verdict.notes
    ]
    return {
        "role": requirements.role_title if requirements else None,
        "candidates": facts,
        "verdict": {"ranking": ranking, "notes": notes, "summary": verdict.summary},
    }


def focus_areas(fact: dict, requirements: Requirements | None) -> dict:
    """What an interview should probe: gaps, unverified claims and strengths."""
    strengths = fact.get("matched_skills") or fact["skills"]
    if requirements is None:
        return {"gaps": [], "unverified": [], "strengths": strengths[:5]}
    have = {s.lower() for s in fact["skills"]}
    checks = fact["must_have_checks"]
    return {
        "gaps": [c["detail"] for c in checks if c["status"] == "not_met"]
        + [f"nice to have: {s}" for s in requirements.nice_to_have if s.lower() not in have],
        "unverified": [c["detail"] for c in checks if c["status"] == "unverified"],
        "strengths": strengths[:5],
    }


def generate_interview_questions(
    services: Services, ref: str, requirements: Requirements | None = None, count: int = 6
) -> dict:
    """Screening questions targeted at one candidate's gaps, claims and strengths."""
    _check_range("count", count, QUESTION_LIMITS)
    profile = services.candidates.resolve(ref)
    (fact,) = candidate_facts(services, [profile], requirements)
    focus = focus_areas(fact, requirements)
    payload = _llm_payload(
        requirements,
        candidate=fact, focus_areas=focus, resume_text=services.candidates.resume_text(profile),
        question_count=count,
    )
    result = llm.structured(InterviewQuestions, prompts.INTERVIEW_QUESTIONS, payload, model=services.model)
    return {
        "candidate": {"id": profile.id, "name": profile.name},
        "role": requirements.role_title if requirements else None,
        "focus_areas": focus,
        "questions": [q.model_dump() for q in result.questions[:count]],
    }
