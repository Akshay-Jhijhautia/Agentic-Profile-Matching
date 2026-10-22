"""Intent routing: decide what the user's latest message asks the agent to do."""

from __future__ import annotations

from langgraph.graph import END

from .. import llm, pools, prompts
from ..schemas import IntentDecision
from ..services import Services
from .state import AgentState
from .text import last_ai_text, last_human_text, refined

# Messages that end the conversation without asking the model.
EXIT_WORDS = {"exit", "quit", "bye", "done", "stop", "thanks, bye"}
# "/resumes <folder or files>" and "/add <files>" load resumes without asking the model.
LOAD_COMMANDS = ("/resumes", "/add")
# Where each intent goes next; a folder to load is handled first (see ``next_step``).
ROUTES = {
    "new_search": "parse_jd", "refine": "refine_requirements", "ask": "assistant", "end": END,
    "load_resumes": "load_resumes",
}
# How much of the previous assistant message the router sees (enough to resolve "yes, do it").
CONTEXT_CHARS = 600


class IntentRouter:
    """The ``route_intent`` node: classify the latest user message."""

    def __init__(self, services: Services):
        self.services = services

    def __call__(self, state: AgentState) -> dict:
        text = last_human_text(state).strip()
        if text.lower().rstrip(".!") in EXIT_WORDS:
            return {"intent": "end"}
        command, _, rest = text.partition(" ")
        if command.lower() in LOAD_COMMANDS and rest.strip():
            return {"intent": "ask", "load_paths": pools.split_paths(rest), "jd_file": None}
        requirements = state.get("requirements")
        routing_input = "\n".join([
            f"Current role: {requirements.role_title if requirements else 'none'}",
            f"Previous assistant message: {last_ai_text(state)[:CONTEXT_CHARS] or 'none'}",
            f"Latest user message: {text}",
        ])
        decision = llm.structured(IntentDecision, prompts.ROUTE_INTENT, routing_input, model=self.services.model)
        intent = decision.intent
        if intent == "refine" and requirements is None:
            intent = "new_search"  # nothing to refine yet: the message describes a role
        return {"intent": intent, "load_paths": decision.resume_paths or None, "jd_file": decision.job_file}


def next_step(state: AgentState) -> str:
    """Conditional edge after ``route_intent``: load a requested folder first, otherwise
    the intent names the branch."""
    return "load_resumes" if state.get("load_paths") else state["intent"]


def after_loading(state: AgentState) -> str:
    """Conditional edge after ``load_resumes``: start the search the message asked for,
    re-screen the new folder for the current role, or wait for the user."""
    if state.get("load_error"):
        return "human_feedback"
    if state["intent"] == "new_search":
        return "parse_jd"
    return "search_resumes" if state.get("requirements") else "human_feedback"


def after_ranking(state: AgentState) -> str:
    """Conditional edge after the last screening round: explain changes only after a refinement."""
    return "explain_changes" if refined(state) else "generate_report"
