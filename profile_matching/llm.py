"""The single path to the chat model.

LLM features ask for a Pydantic object through ``structured`` or for a
tool-calling model through ``with_tools``, so prompts, retries and the model
choice live in one place, and tests replace the model by passing ``model=``
(any object with ``with_structured_output`` / ``bind_tools``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .config import get_settings

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """The configured OpenAI chat model (created once per process)."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )


def structured(schema: type[T], system: str, user: str, *, model: BaseChatModel | None = None) -> T:
    """Ask the model for an instance of ``schema`` (OpenAI strict JSON-schema output)."""
    runnable = (model or get_chat_model()).with_structured_output(schema)
    return runnable.invoke([SystemMessage(content=system), HumanMessage(content=user)])


def with_tools(tools: list[BaseTool], *, model: BaseChatModel | None = None) -> Runnable:
    """The chat model with ``tools`` bound, for the agent's tool-calling loop."""
    return (model or get_chat_model()).bind_tools(tools)
