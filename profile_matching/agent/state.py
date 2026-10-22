"""The agent's state: what the graph remembers within a conversation.

- conversation history (``messages``);
- its understanding of the job (``job_description``, the current
  ``requirements`` and every earlier version in ``requirements_history``);
- the screening results: ``candidates`` (round 1: every resume scored and
  ranked) and the ``shortlist`` (the top candidates with their reasoning, deep
  assessment and recommendation from rounds 2 and 3), plus the previous
  screening so a refinement's ``ranking_diff`` can explain what moved;
- ``assessments``: round-2 results cached by requirements and candidate, so a
  refinement only re-analyses what its change affects;
- the latest ``report`` and the routing decision for the latest message.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph.message import add_messages

from ..schemas import (
    Assessment,
    CandidateMatch,
    Improvement,
    MustHaveCheck,
    RankChange,
    Recommendation,
    RequirementEvidence,
    RankingDiff,
    Requirements,
    ScoreBreakdown,
    SkillRequirement,
)


def merge_dicts(current: dict | None, update: dict | None) -> dict:
    """Reducer for parallel branches that each add entries to one dict."""
    return {**(current or {}), **(update or {})}


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    job_description: str
    requirements: Requirements
    requirements_history: Annotated[list[Requirements], operator.add]
    candidates: list[CandidateMatch]
    shortlist: list[CandidateMatch]
    previous_candidates: list[CandidateMatch]
    previous_shortlist: list[CandidateMatch]
    ranking_diff: RankingDiff | None  # set by a refinement, cleared by a new search
    assessments: Annotated[dict[str, Assessment], merge_dicts]  # see screening.assessment_key
    report: str
    report_dir: str | None  # where the latest reports were saved
    intent: str  # the latest routing decision (see ``routing.ROUTES``)
    resume_folder: str  # what is being screened: a folder, or "your uploaded resumes"
    pool_folder: str  # the absolute folder behind it, to reload the resumes when a session resumes
    upload_folder: str | None  # this conversation's uploads batch, added to by every upload
    # Set by the router for this turn only: resumes to load and a JD file to read.
    load_paths: list[str] | None
    jd_file: str | None
    load_error: str | None


# Our types stored in checkpoints. LangGraph only restores allow-listed types
# (others are deprecated and will be blocked), so every model the state holds,
# including nested ones, is registered here.
STATE_MODELS = (
    Requirements, SkillRequirement, CandidateMatch, MustHaveCheck, ScoreBreakdown, RankingDiff, RankChange,
    Assessment, RequirementEvidence, Recommendation, Improvement,
)


def serializer() -> JsonPlusSerializer:
    """The checkpoint serializer, allowed to restore the agent's state types."""
    return JsonPlusSerializer(allowed_msgpack_modules=STATE_MODELS)


def make_checkpointer() -> InMemorySaver:
    """An in-memory checkpointer (see ``sessions`` for the one saved to disk)."""
    return InMemorySaver(serde=serializer())
