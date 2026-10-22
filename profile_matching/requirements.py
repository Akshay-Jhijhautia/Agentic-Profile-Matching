"""Turn a job description or a natural-language request into ``Requirements``.

The LLM decides what is a must-have and what is only nice to have. Two
deterministic passes then make the result safe to use as knockout filters:

1. Recall floor (``apply_rule_floor``): the Milestone 2 rule parser adds
   vocabulary skills the LLM skipped, as nice-to-have only so a heuristic
   never knocks anyone out, and years it found for a must-have the LLM left
   without them.
2. Knockout policy (``apply_knockout_policy``): a must-have can only reject a
   candidate when the text truly demands it and the resume parser can check
   it. Skills offered as alternatives ("PyTorch or TensorFlow") and broad
   concepts outside the skills vocabulary ("Relational Databases") become
   nice-to-have, and a degree is dropped unless some requirement line asks
   for it without "or equivalent", "preferred" and the like.
"""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel

from . import llm, prompts
from .rag import job_matcher, skills_vocab
from .schemas import Requirements

# Clause boundaries for spotting alternatives: "PyTorch or TensorFlow and
# scikit-learn" offers a choice between the first two only.
_CLAUSE_SPLIT_RE = re.compile(r"[\n,;()]|\band\b", re.IGNORECASE)
_OR_RE = re.compile(r"\bor\b", re.IGNORECASE)
_SOFTENED_RE = re.compile(
    r"\b(?:equivalent|preferred|a plus|nice to have|bonus|desirable|ideally)\b", re.IGNORECASE
)


def apply_rule_floor(requirements: Requirements, rules: dict) -> Requirements:
    """Merge what ``job_matcher.parse_job_description`` found into LLM requirements."""
    rule_years = {
        req["skill"].lower(): req["min_years"]
        for req in rules["must_haves"]
        if req["type"] == "skill_years"
    }
    must_have = [
        req.model_copy(update={"min_years": rule_years[req.skill.lower()]})
        if req.min_years is None and req.skill.lower() in rule_years
        else req
        for req in requirements.must_have
    ]
    known = {s.lower() for s in requirements.must_have_skills + requirements.nice_to_have}
    missed = [
        skill for skill in rules["required_skills"] + rules["preferred_skills"]
        if skill.lower() not in known
    ]
    # model_validate re-runs normalisation and de-duplication on the merged result.
    return Requirements.model_validate(
        requirements.model_dump()
        | {
            "must_have": [req.model_dump() for req in must_have],
            "nice_to_have": requirements.nice_to_have + missed,
        }
    )


def alternative_skills(text: str) -> set[str]:
    """Lower-cased skills offered as one option among several ("Oracle or PostgreSQL")."""
    found = set()
    for clause in _CLAUSE_SPLIT_RE.split(text):
        if _OR_RE.search(clause):
            options = skills_vocab.find_skills(clause)
            if len(options) > 1:
                found.update(skill.lower() for skill in options)
    return found


def degree_required(rules: dict) -> bool:
    """True when a requirement line (per the rule parser) names a degree without softening it."""
    return any(
        req["type"] == "degree" and not _SOFTENED_RE.search(req["source"])
        for req in rules["must_haves"]
    )


def apply_knockout_policy(requirements: Requirements, text: str, rules: dict) -> Requirements:
    """Demote must-haves the text does not strictly demand (see module docstring)."""
    alternatives = alternative_skills(text)
    demoted = [
        req.skill for req in requirements.must_have
        if req.skill.lower() in alternatives or req.skill not in skills_vocab.SKILLS
    ]
    return Requirements.model_validate(
        requirements.model_dump()
        | {
            "must_have": [
                req.model_dump() for req in requirements.must_have if req.skill not in demoted
            ],
            "nice_to_have": demoted + requirements.nice_to_have,
            "min_degree": requirements.min_degree if degree_required(rules) else None,
        }
    )


def extract_requirements(text: str, *, model: BaseChatModel | None = None) -> Requirements:
    """Must-have and nice-to-have requirements for a JD or a short request."""
    if not text or not text.strip():
        raise ValueError("Job description must be a non-empty string")
    extracted = llm.structured(Requirements, prompts.EXTRACT_REQUIREMENTS, text.strip(), model=model)
    rules = job_matcher.parse_job_description(text)
    return apply_knockout_policy(apply_rule_floor(extracted, rules), text, rules)
