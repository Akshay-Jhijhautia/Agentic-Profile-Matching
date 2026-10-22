"""The matching agent: a LangGraph graph that screens resumes for a job.

One-shot (``conversational=False``):

    START → parse_jd → extract_requirements → search_resumes
          → rank_candidates → generate_report → END

Conversational (``conversational=True``), with a human feedback loop:

    START → route_intent ─┬─ new_search → parse_jd → … → generate_report → human_feedback
                          ├─ ask → assistant ⇄ tools → human_feedback
                          └─ end → END
    human_feedback (waits for the user) → route_intent

Usage:
    python matching_agent.py                                   interactive chat
    python matching_agent.py --jd-file job_descriptions/frontend_react.txt
    python matching_agent.py "Find me candidates with React and 3+ years experience"
    python main.py chat | screen ...                           same, via the main CLI
"""

from __future__ import annotations

import argparse
import sys
from itertools import pairwise

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from profile_matching import cli
from profile_matching.agent.nodes import ConversationNodes, PipelineNodes
from profile_matching.agent.routing import ROUTES, IntentRouter, next_step
from profile_matching.agent.state import AgentState, make_checkpointer
from profile_matching.rag import fs_tools, resume_rag
from profile_matching.services import Services
from profile_matching.tools import make_tools

PIPELINE = ("parse_jd", "extract_requirements", "search_resumes", "rank_candidates", "generate_report")


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
    for name in PIPELINE:
        builder.add_node(name, getattr(pipeline, name))
    for current, following in pairwise(PIPELINE):
        builder.add_edge(current, following)

    if not conversational:
        builder.add_edge(START, PIPELINE[0])
        builder.add_edge(PIPELINE[-1], END)
        return builder.compile(checkpointer=checkpointer)

    tools = make_tools(services)
    conversation = ConversationNodes(services, tools)
    builder.add_node("route_intent", IntentRouter(services))
    builder.add_node("assistant", conversation.assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("human_feedback", conversation.human_feedback)

    builder.add_edge(START, "route_intent")
    builder.add_conditional_edges("route_intent", next_step, ROUTES)
    builder.add_edge(PIPELINE[-1], "human_feedback")
    builder.add_conditional_edges("assistant", tools_condition, {"tools": "tools", END: "human_feedback"})
    builder.add_edge("tools", "assistant")
    builder.add_edge("human_feedback", "route_intent")
    return builder.compile(checkpointer=checkpointer or make_checkpointer())


def screen(services: Services, job_description: str) -> AgentState:
    """Run the pipeline once for a JD (text or file path) and return the final state."""
    return build_graph(services).invoke({"messages": [HumanMessage(job_description)]})


def _run_screen(services: Services, job_description: str, out: str | None) -> int:
    state = screen(services, job_description)
    print(state["report"])
    if out:
        written = fs_tools.write_file(out, state["report"])
        if not written["success"]:
            print(f"Error: {written['error']}", file=sys.stderr)
            return 1
        print(f"Saved to {resume_rag.display_path(written['filepath'])}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Screen indexed resumes for a job. Without a JD, start an interactive chat."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("jd", nargs="?", help="Job description text or a short request.")
    source.add_argument("--jd-file", help="Job description file (TXT, MD, PDF or DOCX).")
    parser.add_argument("--out", help="Also save the Markdown report to this file (one-shot only).")
    args = parser.parse_args(argv)

    try:
        services = Services.from_settings()
        services.check_ready()
        job_description = args.jd_file or args.jd
        if job_description:
            return _run_screen(services, job_description, args.out)
    except resume_rag.CLI_ERRORS as exc:
        print(f"Error: {resume_rag.cli_error_message(exc)}", file=sys.stderr)
        return 1

    return cli.run(build_graph(services, conversational=True), services.settings.llm_model)


if __name__ == "__main__":
    sys.exit(main())
