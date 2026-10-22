"""Offline stand-ins for the chat model."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel


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
