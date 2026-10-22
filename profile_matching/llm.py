"""The single path to the chat model.

LLM features ask for a Pydantic object through ``structured`` or for a
tool-calling model through ``with_tools``, so prompts, retries and the model
choice live in one place, and tests replace the model by passing ``model=``
(any object with ``with_structured_output`` / ``bind_tools``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from .config import get_settings

T = TypeVar("T", bound=BaseModel)
# Attempts per structured call: an answer that fails validation is retried with the error.
STRUCTURED_ATTEMPTS = 2


def make_chat_model(model_name: str | None = None) -> BaseChatModel:
    """A chat model with the configured limits; ``model_name`` overrides the configured model."""
    settings = get_settings()
    return ChatOpenAI(
        model=model_name or settings.llm_model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        use_responses_api=settings.llm_responses_api,
    )


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """The configured chat model (created once per process)."""
    return make_chat_model()


def structured(schema: type[T], system: str, user: str, *, model: BaseChatModel | None = None) -> T:
    """Ask the model for an instance of ``schema``.

    An answer that does not validate against the schema is retried once, with
    the validation error shown to the model, before the error is raised.
    """
    runnable = (model or get_chat_model()).with_structured_output(schema)
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    for attempt in range(1, STRUCTURED_ATTEMPTS + 1):
        try:
            return runnable.invoke(messages)
        except (ValidationError, OutputParserException) as exc:
            if attempt == STRUCTURED_ATTEMPTS:
                raise
            messages = [*messages, HumanMessage(
                content=f"Your answer did not match the required schema: {exc}. Answer again, following it exactly."
            )]
    raise AssertionError("unreachable")


def with_tools(tools: list[BaseTool], *, model: BaseChatModel | None = None) -> Runnable:
    """The chat model with ``tools`` bound, for the agent's tool-calling loop."""
    return (model or get_chat_model()).bind_tools(tools)
