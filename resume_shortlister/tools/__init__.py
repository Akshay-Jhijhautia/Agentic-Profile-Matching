"""Tool registry: every tool the agent can call, built around one ``Services``."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from ..services import Services
from .analysis_tools import make_analysis_tools
from .file_tools import make_file_tools
from .search_tools import make_search_tools

TOOL_FACTORIES = (make_file_tools, make_search_tools, make_analysis_tools)


def make_tools(services: Services) -> list[BaseTool]:
    """All agent tools, bound to ``services``."""
    return [tool for factory in TOOL_FACTORIES for tool in factory(services)]


__all__ = ["TOOL_FACTORIES", "make_tools"]
