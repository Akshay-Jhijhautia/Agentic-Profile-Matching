"""Pydantic models shared by the LLM layer, tools, graph nodes and reports.

Models used as LLM output schemas keep every field required (nullable where
optional) and avoid free-form dicts, as OpenAI strict structured output needs.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .rag import skills_vocab

# Matches ``resume_rag.DEGREE_LEVELS`` so degree must-haves reuse its checks.
DegreeLevel = Literal["Associate", "Bachelor's", "Master's", "PhD"]


# Durations and degrees the model sometimes lists as skills ("6 years", "PhD").
_NOT_A_SKILL_RE = re.compile(
    r"^\d+(?:\.\d+)?\+?\s*(?:years?|yrs?)\b|^(?:ph\.?\s?d|doctorate|master'?s|bachelor'?s)\b",
    re.IGNORECASE,
)


def is_skill(text: str) -> bool:
    return bool(text.strip()) and not _NOT_A_SKILL_RE.match(text.strip())


def canonical_skill(raw: str) -> str:
    """The vocabulary spelling of a skill ("react.js" -> "React"), or the trimmed input."""
    raw = raw.strip()
    return skills_vocab.normalize_skill(raw) or raw


class SkillRequirement(BaseModel):
    skill: str = Field(description="One skill, short canonical name, e.g. 'React', 'PostgreSQL'.")
    min_years: float | None = Field(
        description="Minimum years with this skill when the text states it, else null."
    )

    @field_validator("skill")
    @classmethod
    def _canonical(cls, value: str) -> str:
        return canonical_skill(value)

    @property
    def label(self) -> str:
        return f"{self.skill} ({self.min_years:g}+ yrs)" if self.min_years else self.skill


class Requirements(BaseModel):
    """What a role needs. Must-haves knock candidates out; nice-to-haves only rank."""

    role_title: str = Field(description="Role title, or a short label for an ad-hoc request.")
    must_have: list[SkillRequirement] = Field(
        description="Skills stated as essential. Candidates without them are rejected."
    )
    nice_to_have: list[str] = Field(
        description="Preferred, bonus or alternative skills that improve the ranking only."
    )
    min_total_years: float | None = Field(
        description="Minimum years of total professional experience, else null."
    )
    min_degree: DegreeLevel | None = Field(
        description="Lowest acceptable degree when one is required, else null."
    )
    ideal_profile: str = Field(
        description="Two or three sentences describing the ideal candidate in resume language."
    )

    @field_validator("nice_to_have")
    @classmethod
    def _canonical_list(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(canonical_skill(v) for v in values if is_skill(v)))

    @model_validator(mode="after")
    def _dedupe(self) -> Requirements:
        """One entry per real skill (strictest years win); must-haves leave nice-to-have."""
        strictest: dict[str, SkillRequirement] = {}
        for req in filter(lambda r: is_skill(r.skill), self.must_have):
            current = strictest.get(req.skill.lower())
            if current is None or (req.min_years or 0) > (current.min_years or 0):
                strictest[req.skill.lower()] = req
        self.must_have = list(strictest.values())
        self.nice_to_have = [s for s in self.nice_to_have if s.lower() not in strictest]
        return self

    # --- views ---

    @property
    def must_have_skills(self) -> list[str]:
        return [req.skill for req in self.must_have]

    def query_text(self) -> str:
        """A JD-like rendering used for semantic and keyword search.

        Built from the requirements rather than the user's words, so a short
        request searches like a full job description and every refinement
        changes the search as well as the filters.
        """
        lines = [self.role_title, self.ideal_profile]
        if self.must_have:
            lines.append("Must have: " + ", ".join(req.label for req in self.must_have))
        if self.nice_to_have:
            lines.append("Nice to have: " + ", ".join(self.nice_to_have))
        if self.min_total_years:
            lines.append(f"Experience: {self.min_total_years:g}+ years")
        if self.min_degree:
            lines.append(f"Education: {self.min_degree} degree or higher")
        return "\n".join(lines)

    # --- Milestone 2 matcher format ---

    def to_must_haves(self) -> list[dict]:
        """The must-have checks in ``job_matcher.check_must_haves`` format."""
        checks = []
        for req in self.must_have:
            if req.min_years:
                checks.append({"type": "skill_years", "skill": req.skill, "min_years": req.min_years,
                               "source": f"must have {req.label}"})
            else:
                checks.append({"type": "skill", "skill": req.skill, "source": f"must have {req.skill}"})
        if self.min_total_years:
            checks.append({"type": "total_years", "min_years": self.min_total_years,
                           "source": f"{self.min_total_years:g}+ years of experience"})
        if self.min_degree:
            checks.append({"type": "degree", "degree": self.min_degree,
                           "source": f"{self.min_degree} degree"})
        return checks

    def to_parsed(self) -> dict:
        """The ``job_matcher.parse_job_description`` shape, ready for ``JobMatcher.match_parsed``."""
        return {
            "text": self.query_text(),
            "required_skills": self.must_have_skills,
            "preferred_skills": list(self.nice_to_have),
            "must_haves": self.to_must_haves(),
        }
