"""The agent's state: what the graph remembers within a conversation.

- conversation history (``messages``);
- its understanding of the job (``job_description``, the current
  ``requirements`` and every earlier version in ``requirements_history``);
- the screening results (``candidates``: every resume scored and ranked) and
  the ``shortlist`` with each candidate's reasoning, plus the previous
  screening so a refinement's ``ranking_diff`` can explain what moved;
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
    CandidateMatch,
    MustHaveCheck,
    RankChange,
    RankingDiff,
    Requirements,
    ScoreBreakdown,
    SkillRequirement,
)


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
    report: str
    intent: str  # the latest routing decision (see ``routing.ROUTES``)


# Our types stored in checkpoints. LangGraph only restores allow-listed types
# (others are deprecated and will be blocked), so every model the state holds,
# including nested ones, is registered here.
STATE_MODELS = (
    Requirements, SkillRequirement, CandidateMatch, MustHaveCheck, ScoreBreakdown, RankingDiff, RankChange,
)


def make_checkpointer() -> InMemorySaver:
    """An in-memory checkpointer that can restore the agent's state types."""
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=STATE_MODELS))
