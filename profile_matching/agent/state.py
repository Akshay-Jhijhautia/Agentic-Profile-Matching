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


# Our types stored in checkpoints. LangGraph only restores allow-listed types
# (others are deprecated and will be blocked), so every model the state holds,
# including nested ones, is registered here.
STATE_MODELS = (
    Requirements, SkillRequirement, CandidateMatch, MustHaveCheck, ScoreBreakdown, RankingDiff, RankChange,
    Assessment, RequirementEvidence, Recommendation, Improvement,
)


def make_checkpointer() -> InMemorySaver:
    """An in-memory checkpointer that can restore the agent's state types."""
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=STATE_MODELS))
