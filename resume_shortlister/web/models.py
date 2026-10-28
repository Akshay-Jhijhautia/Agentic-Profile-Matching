"""What the web API returns. Everything a screen needs, already worded for people."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from ..conversation import Conversation, Event
from ..agent.routing import LOAD_COMMANDS
from ..explainability import RUBRIC_NOTE, candidate_report, must_have_cell, score_lines, strengths_and_gaps
from ..pools import split_paths
from ..schemas import (
    CandidateMatch,
    Improvement,
    InterviewQuestion,
    MustHaveCheck,
    Recommendation,
    RequirementEvidence,
    Requirements,
    RiskFlag,
)
from ..sessions import SessionInfo


class Status(BaseModel):
    version: str
    model: str
    embedder: str
    ready: bool
    problem: str | None  # what to fix before screening, when not ready


class SessionSummary(BaseModel):
    id: str
    updated: datetime
    role: str | None
    resumes: str | None
    messages: int

    @classmethod
    def of(cls, info: SessionInfo) -> SessionSummary:
        return cls(id=info.id, updated=info.updated, role=info.role, resumes=info.resumes, messages=info.messages)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    text: str

    @classmethod
    def of(cls, message: HumanMessage | AIMessage) -> Message:
        if isinstance(message, AIMessage):
            return cls(role="assistant", text=message.text)
        return cls(role="user", text=describe_command(message.text))


def describe_command(text: str) -> str:
    """A load command as the user would say it: the web app sends "/resumes <folder>" and
    "/add <staged files>" on their behalf, and the staging paths mean nothing to them."""
    command, _, rest = text.partition(" ")
    if command not in LOAD_COMMANDS or not rest.strip():
        return text
    paths = split_paths(rest)
    if command == "/resumes":
        return f"Load the resumes in {', '.join(paths)}"
    names = ", ".join(Path(p).name for p in paths)
    return f"Upload {len(paths)} file{'s' if len(paths) != 1 else ''}: {names}"


class RequirementsView(BaseModel):
    role_title: str
    must_have: list[str]  # labels, e.g. "React (3+ yrs)"
    nice_to_have: list[str]
    min_total_years: float | None
    min_degree: str | None

    @classmethod
    def of(cls, requirements: Requirements) -> RequirementsView:
        return cls(role_title=requirements.role_title, must_have=[r.label for r in requirements.must_have],
                   nice_to_have=requirements.nice_to_have, min_total_years=requirements.min_total_years,
                   min_degree=requirements.min_degree)


class CandidateRow(BaseModel):
    """One line of the ranked table."""

    rank: int
    id: str
    name: str
    score: int  # final score (retrieval blended with the deep review once it has run)
    retrieval: int
    deep: int | None
    must_haves: str  # "✓ all met", "⚠ 1 unverified", "✗ missing React"…
    passes_must_haves: bool
    decision: Literal["interview", "not_now"] | None
    confidence: str | None
    borderline: bool

    @classmethod
    def of(cls, match: CandidateMatch) -> CandidateRow:
        rec = match.recommendation
        return cls(
            rank=match.rank, id=match.id, name=match.name, score=match.final_score, retrieval=match.match_score,
            deep=match.assessment.deep_score if match.assessment else None, must_haves=must_have_cell(match),
            passes_must_haves=match.passes_must_haves, decision=rec.decision if rec else None,
            confidence=rec.confidence if rec else None, borderline=match.borderline,
        )


class Snapshot(BaseModel):
    """A session as the UI shows it."""

    session_id: str
    messages: list[Message]
    resumes: str | None  # what is being screened: a folder, or "your uploaded resumes"
    requirements: RequirementsView | None
    open_questions: list[str]  # requirements waiting for the recruiter's confirmation
    screened: int
    passing: int
    shortlist: list[CandidateRow]
    report: str | None  # the shortlist report (Markdown)
    waiting: bool  # the agent waits for the next message
    notice: str | None = None  # e.g. "the resumes of this session are gone"

    @classmethod
    def of(cls, conversation: Conversation, notice: str | None = None) -> Snapshot:
        values = conversation.values
        requirements = values.get("requirements")
        candidates = values.get("candidates", [])
        return cls(
            session_id=conversation.session_id,
            messages=[
                Message.of(m) for m in values.get("messages", []) if isinstance(m, (HumanMessage, AIMessage)) and m.text
            ],
            resumes=values.get("resume_folder"),
            requirements=RequirementsView.of(requirements) if requirements else None,
            open_questions=values.get("open_questions", []),
            screened=len(candidates),
            passing=sum(m.passes_must_haves for m in candidates),
            shortlist=[CandidateRow.of(m) for m in values.get("shortlist", [])],
            report=values.get("report"),
            waiting=conversation.waiting,
            notice=notice,
        )


class TurnRequest(BaseModel):
    text: str


class FolderRequest(BaseModel):
    path: str


class ScoreComponent(BaseModel):
    component: str  # "Semantic fit", "Deep analysis"…
    value: str  # the component's own score: "72%", or the deep score "85"
    weight: float  # its share of the final score
    points: float  # what it adds to the final score


class CandidateDetail(BaseModel):
    """Everything behind one candidate's place in the ranking (``explainability`` renders
    the same record as the Markdown report, included for the download)."""

    id: str
    name: str
    rank: int
    score: int
    retrieval: int
    deep: int | None
    resume: str  # the file name
    experience_years: float | None
    highest_degree: str | None
    recommendation: Recommendation | None
    borderline: bool
    knocked_out: list[str]  # the must-haves they fail
    summary: str | None  # the deep review's view of their fit
    score_components: list[ScoreComponent]
    rubric: str | None  # how the deep score is computed, once there is one
    must_have_checks: list[MustHaveCheck]
    evidence: list[RequirementEvidence]  # a verdict per requirement, with a quote from the resume
    strengths: list[str]
    gaps: list[str]
    risk_flags: list[RiskFlag]
    improvements: list[Improvement]  # what would make a borderline candidate a clear interview
    retrieval_reasoning: str
    excerpts: list[str]
    report: str  # Markdown

    @classmethod
    def of(cls, match: CandidateMatch, requirements: Requirements) -> CandidateDetail:
        assessment = match.assessment
        strengths, gaps = strengths_and_gaps(match, requirements)
        return cls(
            id=match.id, name=match.name, rank=match.rank, score=match.final_score, retrieval=match.match_score,
            deep=assessment.deep_score if assessment else None, resume=Path(match.resume).name,
            experience_years=match.experience_years, highest_degree=match.highest_degree,
            recommendation=match.recommendation, borderline=match.borderline,
            knocked_out=match.failed_requirements, summary=assessment.summary if assessment else None,
            score_components=[ScoreComponent(**vars(line)) for line in score_lines(match)],
            rubric=RUBRIC_NOTE if assessment else None, must_have_checks=match.must_have_checks,
            evidence=assessment.requirement_evidence if assessment else [], strengths=strengths, gaps=gaps,
            risk_flags=assessment.risk_flags if assessment else [], improvements=match.improvements or [],
            retrieval_reasoning=match.reasoning, excerpts=match.excerpts,
            report=candidate_report(match, requirements),
        )


class FocusAreas(BaseModel):
    gaps: list[str]
    unverified: list[str]
    strengths: list[str]


class InterviewPlan(BaseModel):
    """Interview questions for one candidate and what they aim at."""

    candidate_id: str
    focus_areas: FocusAreas
    questions: list[InterviewQuestion]

    @classmethod
    def of(cls, result: dict) -> InterviewPlan:
        """From ``analysis.generate_interview_questions``."""
        return cls(candidate_id=result["candidate"]["id"], focus_areas=result["focus_areas"],
                   questions=result["questions"])


def event_payload(event: Event) -> dict:
    return {"kind": event.kind, "text": event.text, "node": event.node}
