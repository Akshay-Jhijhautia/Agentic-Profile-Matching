"""Graph nodes. Each takes the state and returns only the keys it changes."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.tools import BaseTool
from langgraph.types import interrupt

from .. import llm, prompts, reports, screening
from ..requirements import extract_requirements, load_job_description
from ..services import Services
from .state import AgentState
from .text import last_ai_text, last_human_text

# Messages of history the assistant sees, cut at a user message so tool calls
# and their results stay together.
ASSISTANT_HISTORY = 40


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


class ConversationNodes:
    """The human feedback loop and the tool-using assistant."""

    def __init__(self, services: Services, tools: list[BaseTool]):
        self.services = services
        self.tools = tools

    @staticmethod
    def human_feedback(state: AgentState) -> dict:
        """Pause the graph until the user replies (resumed with ``Command(resume=text)``)."""
        reply = interrupt({"last_message": last_ai_text(state)})
        return {"messages": [HumanMessage(reply)]}

    def assistant(self, state: AgentState) -> dict:
        """Answer the user, calling tools as needed (loops with the ``tools`` node)."""
        context = reports.conversation_context(
            state.get("requirements"), state.get("candidates", []), state.get("shortlist", [])
        )
        history = trim_messages(
            state["messages"], max_tokens=ASSISTANT_HISTORY, token_counter=len, start_on="human"
        )
        model = llm.with_tools(self.tools, model=self.services.model)
        reply = model.invoke([SystemMessage(prompts.ASSISTANT.format(context=context)), *history])
        return {"messages": [reply]}
