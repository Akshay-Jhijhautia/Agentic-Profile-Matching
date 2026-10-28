"""Requirement extraction and candidate analysis tools.

Tools that need the role read it from graph state (``InjectedState``), so the
model never has to pass the requirements around; the argument is invisible
in the schema the model sees.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import BaseTool
from langgraph.prebuilt import InjectedState

from .. import analysis, requirements
from ..services import Services
from ._base import agent_tool


def make_analysis_tools(services: Services) -> list[BaseTool]:
    @agent_tool
    def extract_requirements(jd: str) -> dict:
        """Split a job description or hiring request into must-have and nice-to-have requirements.

        Must-haves reject candidates who lack them; nice-to-haves only affect ranking.

        Args:
            jd: Job description text, a short request ("React and 3+ years"), or a path to a JD file.
        """
        text = requirements.load_job_description(jd)
        return requirements.extract_requirements(text, model=services.model, cache=services.llm_cache).model_dump()

    @agent_tool
    def compare_candidates(candidate_ids: list[str], state: Annotated[dict, InjectedState]) -> dict:
        """Compare 2-5 candidates head to head for the current role: scores, must-have checks and a verdict.

        Also use it to explain why one candidate ranks above another.

        Args:
            candidate_ids: Candidate ids or names, e.g. ["maria_garcia", "John Doe"].
        """
        role = analysis.as_requirements(state.get("requirements"))
        return analysis.compare_candidates(services, candidate_ids, role, state.get("shortlist", []))

    @agent_tool
    def generate_interview_questions(
        candidate_id: str, state: Annotated[dict, InjectedState], count: int = 6
    ) -> dict:
        """Write screening interview questions for one candidate, aimed at their gaps, unverified claims and strengths.

        Args:
            candidate_id: Candidate id or name.
            count: Number of questions (3-10).
        """
        role = analysis.as_requirements(state.get("requirements"))
        return analysis.generate_interview_questions(
            services, candidate_id, role, count, state.get("shortlist", [])
        )

    @agent_tool
    def get_candidate_report(candidate_id: str, state: Annotated[dict, InjectedState]) -> dict:
        """Get the detailed match report of a candidate in the current search: how the score adds up, must-have checks, evidence per requirement, strengths, gaps, risk flags and improvement suggestions.

        Args:
            candidate_id: Candidate id or name.
        """
        report = analysis.detailed_report(
            services, candidate_id, analysis.as_requirements(state.get("requirements")),
            state.get("shortlist", []), state.get("candidates", []),
        )
        return {"candidate": candidate_id, "report": report}

    return [extract_requirements, compare_candidates, generate_interview_questions, get_candidate_report]
