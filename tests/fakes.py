"""Offline stand-ins for the chat model."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel


class FakeChatModel:
    """Answers ``with_structured_output(schema)`` calls with canned objects.

    ``responses`` maps an output schema to an instance, or to a callable that
    receives the prompt messages and returns one. Every call is recorded in
    ``calls`` as ``(schema, messages)``.
    """

    def __init__(self, responses: dict[type[BaseModel], BaseModel | Callable[[list], BaseModel]]):
        self.responses = responses
        self.calls: list[tuple[type[BaseModel], list]] = []

    def with_structured_output(self, schema: type[BaseModel], **_kwargs) -> RunnableLambda:
        def respond(messages: list) -> BaseModel:
            self.calls.append((schema, messages))
            response = self.responses[schema]
            return response if isinstance(response, BaseModel) else response(messages)

        return RunnableLambda(respond)
