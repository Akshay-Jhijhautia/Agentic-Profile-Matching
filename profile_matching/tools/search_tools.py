"""The Milestone 2 RAG search, exposed to the agent."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from ..candidates import candidate_id
from ..services import Services
from ._base import agent_tool

MAX_SEARCH_RESULTS = 20


def make_search_tools(services: Services) -> list[BaseTool]:
    @agent_tool
    def rag_search(query: str, k: int = 5) -> dict:
        """Search all indexed resumes with hybrid semantic + keyword retrieval.

        Use it for open questions ("who has Kubernetes experience?") rather than
        ranking for the current role. Requirements stated in the query, such as
        "5+ years of Python" or "Master's degree", are applied as filters.

        Args:
            query: What to look for, in natural language or as a job description.
            k: How many candidates to return (1-20).
        """
        if not 1 <= k <= MAX_SEARCH_RESULTS:
            raise ValueError(f"k must be between 1 and {MAX_SEARCH_RESULTS}")
        result = services.matcher.match(query, k=k)
        return {
            "query": query,
            "candidates": [{"id": candidate_id(m["resume_path"])} | m for m in result["top_matches"]],
        }

    return [rag_search]
