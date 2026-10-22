"""Graph nodes. Each takes the state and returns only the keys it changes."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from .. import reports, screening
from ..requirements import extract_requirements, load_job_description
from ..services import Services
from .state import AgentState


def last_human_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


class PipelineNodes:
    """The screening pipeline, bound to one ``Services``."""

    def __init__(self, services: Services):
        self.services = services

    def parse_jd(self, state: AgentState) -> dict:
        """Load the job description from the latest user message (text or a JD file path)."""
        text = load_job_description(last_human_text(state)).strip()
        if not text:
            raise ValueError("No job description given")
        return {"job_description": text}

    def extract_requirements(self, state: AgentState) -> dict:
        """Split the JD into must-have and nice-to-have requirements."""
        requirements = extract_requirements(state["job_description"], model=self.services.model)
        return {"requirements": requirements, "requirements_history": [requirements]}

    def search_resumes(self, state: AgentState) -> dict:
        """Round 1: score and check every indexed resume against the requirements."""
        return {"candidates": screening.evaluate(self.services, state["requirements"])}

    def rank_candidates(self, state: AgentState) -> dict:
        """Keep the best candidates who pass every must-have."""
        size = self.services.settings.shortlist_size
        return {
            "previous_shortlist": state.get("shortlist", []),
            "shortlist": screening.shortlist(state["candidates"], size),
        }

    def generate_report(self, state: AgentState) -> dict:
        """Write the shortlist report and a short summary for the conversation."""
        parts = (state["requirements"], state["candidates"], state["shortlist"])
        return {
            "report": reports.shortlist_report(*parts),
            "messages": [AIMessage(reports.shortlist_summary(*parts))],
        }
