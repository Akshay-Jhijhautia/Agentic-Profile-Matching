"""Intent routing: decide what the user's latest message asks the agent to do."""

from __future__ import annotations

from langgraph.graph import END

from .. import llm, prompts
from ..schemas import IntentDecision
from ..services import Services
from .state import AgentState
from .text import last_ai_text, last_human_text, refined

# Messages that end the conversation without asking the model.
EXIT_WORDS = {"exit", "quit", "bye", "done", "stop", "thanks, bye"}
# Where each intent goes next.
ROUTES = {"new_search": "parse_jd", "refine": "refine_requirements", "ask": "assistant", "end": END}
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
        requirements = state.get("requirements")
        routing_input = "\n".join([
            f"Current role: {requirements.role_title if requirements else 'none'}",
            f"Previous assistant message: {last_ai_text(state)[:CONTEXT_CHARS] or 'none'}",
            f"Latest user message: {text}",
        ])
        decision = llm.structured(IntentDecision, prompts.ROUTE_INTENT, routing_input, model=self.services.model)
        if decision.intent == "refine" and requirements is None:
            return {"intent": "new_search"}  # nothing to refine yet: the message describes a role
        return {"intent": decision.intent}


def next_step(state: AgentState) -> str:
    """Conditional edge after ``route_intent``: the intent names the branch."""
    return state["intent"]


def after_ranking(state: AgentState) -> str:
    """Conditional edge after the last screening round: explain changes only after a refinement."""
    return "explain_changes" if refined(state) else "generate_report"
