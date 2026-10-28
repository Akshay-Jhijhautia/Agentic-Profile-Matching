"""Shared plumbing for agent tools.

``agent_tool`` turns a plain function into a LangChain tool whose name,
description and argument docs come from the function and its Google-style
docstring. Expected failures (bad input, unknown candidate, unreadable file)
come back to the model as ``{"success": False, "error": ...}`` so it can
recover; anything else (bugs, authentication errors) still raises.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

from langchain_core.tools import BaseTool, tool

EXPECTED_ERRORS = (ValueError, LookupError, OSError)
# Longest text field a tool returns, so one result cannot flood the context.
MAX_TOOL_TEXT_CHARS = 15_000


def _truncate(result):
    if not isinstance(result, dict):
        return result
    trimmed = dict(result)
    for key, value in result.items():
        if isinstance(value, str) and len(value) > MAX_TOOL_TEXT_CHARS:
            trimmed[key] = value[:MAX_TOOL_TEXT_CHARS]
            trimmed["truncated"] = f"'{key}' cut to {MAX_TOOL_TEXT_CHARS} of {len(value)} characters"
    return trimmed


def safe_tool(func: Callable) -> Callable:
    """Report expected errors as data and cap long text fields."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return _truncate(func(*args, **kwargs))
        except EXPECTED_ERRORS as exc:
            return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

    return wrapper


def agent_tool(func: Callable) -> BaseTool:
    return tool(parse_docstring=True)(safe_tool(func))
