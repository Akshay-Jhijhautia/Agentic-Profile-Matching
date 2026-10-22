"""The single path to the chat model.

LLM features ask for a Pydantic object through ``structured`` or for a
tool-calling model through ``with_tools``, so prompts, retries and the model
choice live in one place, and tests replace the model by passing ``model=``
(any object with ``with_structured_output`` / ``bind_tools``). The model comes
from the configured provider (see ``providers``); nothing here is specific to one.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from . import providers
from .config import get_settings
from .rag import resume_rag

T = TypeVar("T", bound=BaseModel)
# Attempts per structured call: an answer that fails validation is retried with the error.
STRUCTURED_ATTEMPTS = 2


def make_chat_model(model: str | None = None) -> BaseChatModel:
    """A chat model with the configured limits.

    ``model`` overrides the configured model, as a name of the configured
    provider or as ``provider:model`` (``"anthropic:claude-sonnet-5-5"``).
    """
    settings = get_settings()
    provider, name = providers.split_model_spec(model or settings.llm_model, settings.llm_provider)
    return providers.create_chat_model(provider, name, settings)


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """The configured chat model (created once per process)."""
    return make_chat_model()


def structured(schema: type[T] | dict, system: str, user: str, *, model: BaseChatModel | None = None) -> T | dict:
    """Ask the model for an instance of ``schema`` (a dict for a JSON-schema dict).

    An answer that does not validate against the schema is retried once, with
    the validation error shown to the model, before the error is raised.
    """
    model = model or get_chat_model()
    runnable = model.with_structured_output(schema, **providers.structured_options(model))
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


def metadata_fallback() -> resume_rag.MetadataLLM | None:
    """The configured model as the indexer's fallback for resume fields its rules miss,
    or None when the model cannot be used (no key, integration not installed)."""
    settings = get_settings()
    if providers.missing_key(providers.required_key_vars(settings)):
        return None
    try:
        model = get_chat_model()
    except ValueError:
        return None

    def extract(schema: dict, system: str, text: str) -> dict:
        try:
            return structured(schema, system, text, model=model)
        except (*providers.api_errors(), ValueError) as exc:
            raise resume_rag.MetadataLLMError(providers.error_message(exc)) from exc

    return extract
