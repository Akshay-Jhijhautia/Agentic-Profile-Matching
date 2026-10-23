"""Pydantic models shared by the LLM layer, tools, graph nodes and reports.

Models used as LLM output schemas keep every field required (nullable where
optional) and avoid free-form dicts, as OpenAI strict structured output needs.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from .candidates import candidate_id
from .rag import job_matcher, skills_vocab

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

        Built only from the structured requirements, never from model-written
        prose (not even the role title), so the same requirements always search
        the same way whichever model extracted them, and a refinement changes
        the search exactly as much as it changes the requirements.
        """
        lines = []
        if self.must_have:
            lines.append("Must have: " + ", ".join(req.label for req in self.must_have))
        if self.nice_to_have:
            lines.append("Nice to have: " + ", ".join(self.nice_to_have))
        if self.min_total_years:
            lines.append(f"Experience: {self.min_total_years:g}+ years")
        if self.min_degree:
            lines.append(f"Education: {self.min_degree} degree or higher")
        return "\n".join(lines)

    # --- hybrid matcher (``job_matcher``) format ---

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


# --- candidate analysis (LLM output) ---------------------------------------------------------


class CandidateNote(BaseModel):
    candidate_id: str = Field(description="The candidate's id exactly as given in the input.")
    strengths: list[str] = Field(description="Concrete strengths for this role, citing resume evidence.")
    weaknesses: list[str] = Field(description="Gaps or risks for this role, citing resume evidence.")


class ComparisonVerdict(BaseModel):
    ranking: list[str] = Field(description="Candidate ids, best fit first.")
    notes: list[CandidateNote] = Field(description="One entry per candidate.")
    summary: str = Field(
        description="Two to four sentences naming the best fit and the decisive differences."
    )


class InterviewQuestion(BaseModel):
    question: str = Field(description="The question, specific to this candidate's resume.")
    focus: str = Field(description="The skill or topic the question probes.")
    purpose: Literal["close_gap", "verify_claim", "assess_depth"] = Field(
        description="close_gap: a missing requirement; verify_claim: a skill listed without "
        "clear evidence; assess_depth: a strength worth probing."
    )
    strong_answer: str = Field(description="What a strong answer would cover.")


class InterviewQuestions(BaseModel):
    questions: list[InterviewQuestion]


# --- screening results ---------------------------------------------------------------------------

CheckStatus = Literal["met", "not_met", "unverified"]


class MustHaveCheck(BaseModel):
    requirement: str
    status: CheckStatus
    detail: str


class ScoreBreakdown(BaseModel):
    """The hybrid score's inputs, each 0..1 (skill_coverage is None without must-have skills)."""

    semantic: float
    keyword: float
    skill_coverage: float | None


# --- rounds 2 and 3 ---------------------------------------------------------------------------------
#
# The LLM only judges (``DeepReview``, ``DecisionRationales``); code turns its judgements into
# scores and decisions (``Assessment``, ``Recommendation``) under the rules in ``guardrails``.

Verdict = Literal["strong", "partial", "missing"]
Severity = Literal["low", "medium", "high"]


class RequirementReview(BaseModel):
    requirement_id: str = Field(description="The requirement's id from the input, e.g. 'M1' or 'N2'.")
    verdict: Verdict = Field(
        description="strong: clearly demonstrated; partial: present but thin or indirect; missing: not shown."
    )
    evidence_quote: str = Field(
        description="A short verbatim quote from the resume supporting the verdict; empty when missing."
    )


class RiskFlag(BaseModel):
    flag: str = Field(description="The concern, specific to this resume.")
    severity: Severity = Field(
        description="high: likely disqualifying (e.g. fabricated-looking claims); medium: needs "
        "checking (gaps, contradictions); low: minor."
    )

    @property
    def label(self) -> str:
        return f"{self.flag} ({self.severity})"


class DeepReview(BaseModel):
    """Round 2, as the LLM returns it: categorical judgements only, no scores."""

    requirements: list[RequirementReview] = Field(description="One entry per requirement id in the input.")
    strengths: list[str]
    gaps: list[str]
    risk_flags: list[RiskFlag] = Field(
        description="Short tenures, unexplained gaps, contradictions, claims without supporting experience."
    )
    summary: str = Field(description="Two sentences on overall fit.")


class RequirementEvidence(BaseModel):
    """One requirement after the guardrails: the verdict that counts and why."""

    requirement: str
    kind: Literal["must", "nice"]
    verdict: Verdict
    evidence: str
    note: str = ""  # set when a guardrail changed the model's verdict


class Assessment(BaseModel):
    """Round 2 result: the review with verdicts checked by code and the rubric's deep score."""

    requirement_evidence: list[RequirementEvidence]
    strengths: list[str]
    gaps: list[str]
    risk_flags: list[RiskFlag]
    deep_score: int
    summary: str

    @property
    def adjustments(self) -> list[str]:
        """Every verdict a guardrail changed, with the reason."""
        return [f"{e.requirement}: {e.note}" for e in self.requirement_evidence if e.note]


# The screening recommends whom to interview; people make the hiring decisions.
Decision = Literal["interview", "not_now"]
# Decisions as earlier versions stored them (saved sessions), read as today's.
_LEGACY_DECISIONS = {"hire": "interview", "no_hire": "not_now"}
Confidence = Literal["high", "medium", "low"]


class DecisionRationale(BaseModel):
    candidate_id: str = Field(description="The candidate's id exactly as given in the input.")
    rationale: str = Field(description="One or two sentences explaining the given decision with evidence.")


class DecisionRationales(BaseModel):
    """Round 3, as the LLM returns it: an explanation for each decision code already made."""

    rationales: list[DecisionRationale]


class Recommendation(BaseModel):
    decision: Decision
    confidence: Confidence
    rule: str  # the decision rule's outcome, e.g. "final 82 ≥ 75; no must-have missing; no high risk"
    rationale: str

    @field_validator("decision", mode="before")
    @classmethod
    def _current_name(cls, value: str) -> str:
        return _LEGACY_DECISIONS.get(value, value)

    @property
    def label(self) -> str:
        return "Recommend interview" if self.decision == "interview" else "Not now"


class Improvement(BaseModel):
    gap: str = Field(description="The gap or doubt this addresses.")
    suggestion: str = Field(description="What the candidate could do or show to close it.")
    how_to_verify: str = Field(description="How the recruiter can check it: a question, portfolio, reference.")


class CandidateImprovements(BaseModel):
    candidate_id: str = Field(description="The candidate's id exactly as given in the input.")
    improvements: list[Improvement] = Field(description="Two to four, most important first.")


class ImprovementSuggestions(BaseModel):
    """Suggestions for borderline candidates: what would make them a clear interview."""

    candidates: list[CandidateImprovements]


# Final score = DEEP_WEIGHT × deep score + (1 − DEEP_WEIGHT) × retrieval score. The deep
# review reads the whole resume, so it outweighs retrieval once it exists.
DEEP_WEIGHT = 0.6
# Final scores in [low, high) are borderline: worth a second look before rejecting.
BORDERLINE_BAND = (60, 75)
# The retrieval score's signals and their weights (see ``job_matcher.hybrid_score``).
RETRIEVAL_WEIGHTS = {
    "semantic": job_matcher.SEMANTIC_WEIGHT,
    "keyword": job_matcher.KEYWORD_WEIGHT,
    "skill_coverage": job_matcher.SKILL_WEIGHT,
}
COMPONENT_LABELS = {
    "semantic": "semantic fit",
    "keyword": "keyword match",
    "skill_coverage": "must-have skill coverage",
    "deep": "deep analysis",
}


class CandidateMatch(BaseModel):
    """One candidate across the screening rounds: retrieval score and must-have checks
    (round 1), then the deep assessment (round 2) and the recommendation (round 3)."""

    id: str
    name: str
    resume: str
    rank: int = 0  # set by ``screening.rank_matches``; 1 is best
    match_score: int
    failed_requirements: list[str]
    must_have_checks: list[MustHaveCheck]
    matched_skills: list[str]
    missing_must_haves: list[str]
    score_breakdown: ScoreBreakdown
    experience_years: float | None
    highest_degree: str | None
    reasoning: str
    excerpts: list[str]
    assessment: Assessment | None = None  # round 2
    recommendation: Recommendation | None = None  # round 3
    improvements: list[Improvement] | None = None  # borderline candidates only

    @computed_field
    @property
    def passes_must_haves(self) -> bool:
        return not self.failed_requirements

    @computed_field
    @property
    def final_score(self) -> int:
        """The retrieval score, blended with the deep score once round 2 has run."""
        if self.assessment is None:
            return self.match_score
        return round(DEEP_WEIGHT * self.assessment.deep_score + (1 - DEEP_WEIGHT) * self.match_score)

    def retrieval_weights(self) -> dict[str, float]:
        """Effective weight of each retrieval signal. Without must-have skills there is no
        coverage, and its weight is shared out as ``job_matcher.hybrid_score`` does."""
        if self.score_breakdown.skill_coverage is not None:
            return dict(RETRIEVAL_WEIGHTS)
        total = RETRIEVAL_WEIGHTS["semantic"] + RETRIEVAL_WEIGHTS["keyword"]
        return {name: RETRIEVAL_WEIGHTS[name] / total for name in ("semantic", "keyword")}

    def contributions(self) -> dict[str, float]:
        """Points each component adds to the 0-100 final score (they sum to it before rounding)."""
        retrieval_share = 1 - DEEP_WEIGHT if self.assessment else 1
        points = {
            name: 100 * retrieval_share * weight * getattr(self.score_breakdown, name)
            for name, weight in self.retrieval_weights().items()
        }
        if self.assessment is not None:
            points["deep"] = DEEP_WEIGHT * self.assessment.deep_score
        return points

    @property
    def borderline(self) -> bool:
        low, high = BORDERLINE_BAND
        return self.assessment is not None and low <= self.final_score < high

    @property
    def unverified(self) -> list[str]:
        return [c.detail for c in self.must_have_checks if c.status == "unverified"]

    @classmethod
    def from_evaluation(cls, result: dict) -> CandidateMatch:
        """From one ``JobMatcher.evaluate`` result."""
        details = result["details"]
        return cls(
            id=candidate_id(result["resume_path"]),
            name=result["candidate_name"],
            resume=result["resume_path"],
            match_score=result["match_score"],
            failed_requirements=result["failed_requirements"],
            must_have_checks=details["must_have_checks"],
            matched_skills=result["matched_skills"],
            missing_must_haves=details["missing_skills"],
            score_breakdown=ScoreBreakdown(
                semantic=details["semantic_score"],
                keyword=details["keyword_score"],
                skill_coverage=details["skill_coverage"],
            ),
            experience_years=details["experience_years"],
            highest_degree=details["highest_degree"],
            reasoning=result["reasoning"],
            excerpts=result["relevant_excerpts"],
        )


# --- conversation routing (LLM output) ---------------------------------------------------------

Intent = Literal["new_search", "refine", "confirm", "ask", "end"]


class IntentDecision(BaseModel):
    intent: Intent = Field(
        description="new_search: screen for a new role from scratch; refine: change the current "
        "requirements; confirm: accept requirements the assistant asked them to check; ask: anything "
        "about the current candidates, role or files; end: the user is finished."
    )
    resume_paths: list[str] = Field(
        description="Folders or resume files the user wants screened, exactly as written; empty if none."
    )
    job_file: str | None = Field(
        description="A path to a job description file the user gave, exactly as written; else null."
    )


# --- ranking changes after a refinement ------------------------------------------------------------

ChangeKind = Literal["new", "dropped", "up", "down", "same"]


class RankChange(BaseModel):
    """How one candidate's place in the shortlist changed, and why."""

    id: str
    name: str
    kind: ChangeKind
    old_rank: int | None  # overall rank before (None if not screened before)
    new_rank: int | None  # overall rank now
    old_score: int | None
    new_score: int | None
    reasons: list[str]


class RankingDiff(BaseModel):
    requirement_changes: list[str]
    changes: list[RankChange]  # everyone in the new shortlist, then everyone who left it

    def of_kind(self, *kinds: ChangeKind) -> list[RankChange]:
        return [change for change in self.changes if change.kind in kinds]


class ChangeNarrative(BaseModel):
    narrative: str = Field(description="Three to five sentences for the recruiter.")
