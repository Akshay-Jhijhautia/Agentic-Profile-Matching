"""Intent routing: decide what the user's latest message asks the agent to do."""

from __future__ import annotations

import re

from langgraph.graph import END

from .. import llm, pools, prompts
from ..schemas import IntentDecision
from ..services import Services
from .state import AgentState
from .text import last_ai_text, last_human_text, refined

# Messages that end the conversation without asking the model.
EXIT_WORDS = {"exit", "quit", "bye", "done", "stop", "thanks, bye"}
# A reply made only of these words accepts proposed requirements without asking the model.
YES_WORDS = {"y", "yes", "yep", "yeah", "ok", "okay", "sure", "correct", "confirm", "confirmed", "fine", "great",
             "perfect", "lgtm", "right", "looks", "sounds", "good", "go", "ahead", "proceed", "please", "do", "it",
             "all", "that's", "thats", "screen", "them", "these"}
# "/resumes <folder or files>" and "/add <files>" load resumes without asking the model.
LOAD_COMMANDS = ("/resumes", "/add")
# Where each intent goes next; a folder to load is handled first (see ``next_step``).
ROUTES = {
    "new_search": "parse_jd", "refine": "refine_requirements", "ask": "assistant", "end": END,
    "load_resumes": "load_resumes", "confirm": "accept_requirements",
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
        pending = bool(state.get("open_questions"))
        words = re.findall(r"[a-z']+", text.lower())
        if pending and words and set(words) <= YES_WORDS:
            return {"intent": "confirm", "load_paths": None, "jd_file": None}
        requirements = state.get("requirements")
        routing_input = "\n".join([
            f"Current role: {requirements.role_title if requirements else 'none'}",
            f"Previous assistant message: {last_ai_text(state)[:CONTEXT_CHARS] or 'none'}",
            f"Latest user message: {text}",
        ])
        decision = llm.structured(IntentDecision, prompts.ROUTE_INTENT, routing_input, model=self.services.model, cache=self.services.llm_cache)
        intent = decision.intent
        if intent == "refine" and requirements is None:
            intent = "new_search"  # nothing to refine yet: the message describes a role
        if intent == "confirm" and not pending:
            intent = "ask"  # nothing waits for a confirmation
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
