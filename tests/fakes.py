"""Offline stand-ins for the chat model."""

from __future__ import annotations

import json
from collections.abc import Callable

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from profile_matching.schemas import Assessment, HiringDecision, HiringDecisions, RequirementEvidence


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
    return Assessment(
        requirement_evidence=[RequirementEvidence(requirement="Python", verdict="strong", evidence="5 yrs")],
        strengths=["Shipped production APIs"], gaps=["No GraphQL"], risk_flags=[],
        deep_score=deep_score, summary="Solid backend engineer.",
    )


def fake_committee(messages: list) -> HiringDecisions:
    """Hires every candidate with a final score of 75 or more, as the real committee might."""
    candidates = json.loads(messages[1].content)["candidates"]
    return HiringDecisions(decisions=[
        HiringDecision(candidate_id=c["id"], decision="hire" if c["final_score"] >= 75 else "no_hire",
                       confidence="medium", rationale=f"Final score {c['final_score']}.")
        for c in candidates
    ])


# Answers for every LLM call the screening rounds make; merge into a FakeChatModel's responses.
SCREENING_RESPONSES = {Assessment: fake_assessment(), HiringDecisions: fake_committee}
