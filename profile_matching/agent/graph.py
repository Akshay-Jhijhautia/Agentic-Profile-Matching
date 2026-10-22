"""The matching agent's LangGraph graph.

One-shot (``conversational=False``):

    START → parse_jd → extract_requirements → search_resumes (round 1: all resumes)
          → rank_candidates (top N) → assess_candidate ×N in parallel (round 2)
          → finalize_ranking (re-rank + round 3 hire/no-hire)
          → suggest_improvements (borderline candidates) → generate_report → END

Conversational (``conversational=True``), with a human feedback loop:

    START → route_intent ─┬─ new_search → parse_jd → extract_requirements ─┐
                          ├─ refine → refine_requirements ─────────────────┴→ search_resumes
                          ├─ resumes → load_resumes → (search | screen again | wait)
                          ├─ ask → assistant ⇄ tools → human_feedback
                          └─ end → END
    search_resumes → rank_candidates → assess_candidate ×N → finalize_ranking
                   → suggest_improvements → [explain_changes, after a refine] → generate_report
    generate_report → human_feedback (waits for the user) → route_intent
    Without resumes loaded, requirements go to await_resumes instead of search_resumes.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ..services import Services
from ..tools import make_tools
from .nodes import ConversationNodes, PipelineNodes
from .routing import ROUTES, IntentRouter, after_loading, after_ranking, next_step
from .state import AgentState, make_checkpointer

# Linear steps up to the round-1 shortlist; then round 2 fans out, round 3 finalises.
PIPELINE = ("parse_jd", "extract_requirements", "search_resumes", "rank_candidates")
ASSESS, FINALIZE, IMPROVE, REPORT = "assess_candidate", "finalize_ranking", "suggest_improvements", "generate_report"
# Runs between ranking and the report, only when the turn refined the requirements.
EXPLAIN = "explain_changes"
# Replaces screening when requirements are set but no resumes are loaded (chat only).
AWAIT = "await_resumes"


def build_graph(
    services: Services,
    checkpointer: BaseCheckpointSaver | None = None,
    conversational: bool = False,
) -> CompiledStateGraph:
    """The agent graph bound to ``services``.

    The conversational graph pauses for the user, so it always has a
    checkpointer (a fresh in-memory one unless given).
    """
    pipeline = PipelineNodes(services)
    builder = StateGraph(AgentState)
    for name in (*PIPELINE, ASSESS, FINALIZE, IMPROVE, EXPLAIN, REPORT, AWAIT):
        builder.add_node(name, getattr(pipeline, name))
    parse, extract, search, rank = PIPELINE
    builder.add_edge(parse, extract)
    builder.add_conditional_edges(extract, pipeline.resumes_loaded, [search, AWAIT])
    builder.add_edge(search, rank)
    builder.add_conditional_edges(PIPELINE[-1], pipeline.fan_out_assessments, [ASSESS, FINALIZE])
    builder.add_edge(ASSESS, FINALIZE)
    builder.add_edge(FINALIZE, IMPROVE)
    builder.add_conditional_edges(IMPROVE, after_ranking, [EXPLAIN, REPORT])
    builder.add_edge(EXPLAIN, REPORT)

    if not conversational:
        builder.add_edge(START, PIPELINE[0])
        builder.add_edge(REPORT, END)
        builder.add_edge(AWAIT, END)
        return builder.compile(checkpointer=checkpointer)

    tools = make_tools(services)
    conversation = ConversationNodes(services, tools)
    builder.add_node("route_intent", IntentRouter(services))
    builder.add_node("refine_requirements", pipeline.refine_requirements)
    builder.add_node("load_resumes", conversation.load_resumes)
    builder.add_node("assistant", conversation.assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("human_feedback", conversation.human_feedback)

    builder.add_edge(START, "route_intent")
    builder.add_conditional_edges("route_intent", next_step, ROUTES)
    builder.add_conditional_edges("refine_requirements", pipeline.resumes_loaded, [search, AWAIT])
    builder.add_edge(AWAIT, "human_feedback")
    builder.add_conditional_edges(
        "load_resumes", after_loading, ["parse_jd", "search_resumes", "human_feedback"]
    )
    builder.add_edge(REPORT, "human_feedback")
    builder.add_conditional_edges("assistant", tools_condition, {"tools": "tools", END: "human_feedback"})
    builder.add_edge("tools", "assistant")
    builder.add_edge("human_feedback", "route_intent")
    return builder.compile(checkpointer=checkpointer or make_checkpointer())


def screen(services: Services, job_description: str) -> AgentState:
    """Run the pipeline once for a JD (text or file path) and return the final state."""
    return build_graph(services).invoke({"messages": [HumanMessage(job_description)]})
