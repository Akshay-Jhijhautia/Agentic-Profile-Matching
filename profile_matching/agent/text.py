"""Read the conversation and the current turn from state."""

from __future__ import annotations

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage


def _last_text(messages: list[AnyMessage], kind: type) -> str:
    for message in reversed(messages):
        if isinstance(message, kind) and message.text:
            return message.text
    return ""


def last_human_text(state: dict) -> str:
    return _last_text(state.get("messages", []), HumanMessage)


def last_ai_text(state: dict) -> str:
    """The latest assistant message with text (tool-call-only messages are skipped)."""
    return _last_text(state.get("messages", []), AIMessage)


def refined(state: dict) -> bool:
    """Whether this turn re-ranks after a refinement (rather than a new search)."""
    return state.get("intent") == "refine" and len(state.get("requirements_history", [])) > 1
