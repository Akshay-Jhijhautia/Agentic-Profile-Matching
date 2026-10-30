"""Offline stand-ins for the chat model."""

from __future__ import annotations

import json
from collections.abc import Callable

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from resume_shortlister.schemas import (
    Assessment,
    CandidateImprovements,
    DecisionRationale,
    DeepReview,
    DecisionRationales,
    Improvement,
    ImprovementSuggestions,
    RequirementEvidence,
    RequirementReview,
    RiskFlag,
)


class FakeChatModel:
    """Answers ``with_structured_output(schema)`` calls with canned objects, and
    ``bind_tools(...)`` calls with scripted assistant messages.

    ``responses`` maps an output schema to an instance, or to a callable that
    receives the prompt messages and returns one. ``replies`` are returned in
    order by the tool-calling model (an exception in the list is raised
    instead). Every call is recorded in ``calls`` as ``(schema, messages)``,
    with ``"tools"`` as the schema for tool-calling turns.
    """

    def __init__(
        self,
        responses: dict[type[BaseModel], BaseModel | Callable[[list], BaseModel]],
        replies: list[AIMessage | Exception] | None = None,
    ):
        self.responses = responses
        self.replies = list(replies or [])
        self.calls: list[tuple[type[BaseModel] | str, list]] = []

    def bind_tools(self, tools: list, **_kwargs) -> RunnableLambda:
        def reply(messages: list) -> AIMessage:
            self.calls.append(("tools", messages))
            answer = self.replies.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        return RunnableLambda(reply)

    def with_structured_output(self, schema: type[BaseModel], **_kwargs) -> RunnableLambda:
        def respond(messages: list) -> BaseModel:
            self.calls.append((schema, messages))
            response = self.responses[schema]
            return response if isinstance(response, BaseModel) else response(messages)

        return RunnableLambda(respond)


# --- canned screening rounds 2 and 3 -------------------------------------------------------------


def fake_assessment(deep_score: int = 80) -> Assessment:
    """A stored round-2 result, for tests that need a given deep score."""
    return Assessment(
        requirement_evidence=[RequirementEvidence(requirement="Python", kind="must", verdict="strong",
                                                  evidence="5 yrs")],
        strengths=["Shipped production APIs"], gaps=["No GraphQL"], risk_flags=[],
        deep_score=deep_score, summary="Solid backend engineer.",
    )


def fake_review(verdict: str = "strong", risk_flags: list[RiskFlag] | None = None) -> Callable[[list], DeepReview]:
    """A model answering ``verdict`` for every requirement it is asked about, quoting the
    resume's first line long enough to be evidence (it is really in the resume, so the
    evidence check accepts it)."""

    def answer(messages: list) -> DeepReview:
        payload = json.loads(messages[1].content)
        quote = next(line for line in payload["resume_text"].splitlines() if len(line.strip()) >= 30)
        return DeepReview(
            requirements=[
                RequirementReview(requirement_id=item["id"], verdict=verdict,
                                  evidence_quote="" if verdict == "missing" else quote)
                for item in payload["requirements"]
            ],
            strengths=["Shipped production APIs"], gaps=["No GraphQL"], risk_flags=risk_flags or [],
            summary="Solid backend engineer.",
        )

    return answer


def fake_rationales(messages: list) -> DecisionRationales:
    candidates = json.loads(messages[1].content)["candidates"]
    return DecisionRationales(rationales=[
        DecisionRationale(candidate_id=c["candidate_id"], rationale=f"Final score {c['final_score']}.") for c in candidates
    ])


def fake_advisor(messages: list) -> ImprovementSuggestions:
    """One improvement per borderline candidate it is asked about."""
    candidates = json.loads(messages[1].content)["candidates"]
    return ImprovementSuggestions(candidates=[
        CandidateImprovements(candidate_id=c["candidate_id"], improvements=[
            Improvement(gap="No GraphQL", suggestion="Build a GraphQL API", how_to_verify="Ask for the repo"),
        ])
        for c in candidates
    ])


# Answers for every LLM call the screening rounds make; merge into a FakeChatModel's responses.
SCREENING_RESPONSES = {
    DeepReview: fake_review(), DecisionRationales: fake_rationales, ImprovementSuggestions: fake_advisor,
}
