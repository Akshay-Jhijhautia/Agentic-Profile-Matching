"""The matching agent: a LangGraph graph that screens resumes for a job.

One-shot (``conversational=False``):

    START → parse_jd → extract_requirements → search_resumes (round 1: all resumes)
          → rank_candidates (top N) → assess_candidate ×N in parallel (round 2)
          → finalize_ranking (re-rank + round 3 hire/no-hire)
          → suggest_improvements (borderline candidates) → generate_report → END

Conversational (``conversational=True``), with a human feedback loop:

    START → route_intent ─┬─ new_search → parse_jd → extract_requirements ─┐
                          ├─ refine → refine_requirements ─────────────────┴→ search_resumes
                          ├─ ask → assistant ⇄ tools → human_feedback
                          └─ end → END
    search_resumes → rank_candidates → assess_candidate ×N → finalize_ranking
                   → suggest_improvements → [explain_changes, after a refine] → generate_report
    generate_report → human_feedback (waits for the user) → route_intent

Usage (``--resumes`` takes a folder or resume files; folders are indexed incrementally):
    python matching_agent.py                                        chat; load resumes in it
    python matching_agent.py --resumes ~/Downloads/backend          chat about a folder
    python matching_agent.py --jd-file job.pdf --resumes jane.pdf raj.docx    one-shot screening
    python main.py chat | screen ...                           same, via the main CLI

Every screening saves its shortlist report and one detailed report per candidate
under REPORTS_DIR (see ``explainability.save_reports``).
"""

from __future__ import annotations

import argparse
import sys

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from profile_matching import cli, pools
from profile_matching.agent.nodes import ConversationNodes, PipelineNodes
from profile_matching.agent.routing import ROUTES, IntentRouter, after_loading, after_ranking, next_step
from profile_matching.agent.state import AgentState, make_checkpointer
from profile_matching.rag import resume_rag
from profile_matching.services import Services
from profile_matching.tools import make_tools

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


def _run_screen(services: Services, job_description: str) -> int:
    state = screen(services, job_description)
    print(state["report"])
    if state.get("report_dir"):
        print(f"Reports saved to {state['report_dir']}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None, mode: str = "auto") -> int:
    """``mode``: "screen" needs a JD, "chat" takes none, "auto" chats when no JD is given."""
    parser = argparse.ArgumentParser(
        description="Screen resumes for a job. Without a JD, start an interactive chat."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("jd", nargs="?", help="Job description text or a short request.")
    source.add_argument("--jd-file", help="Job description file (TXT, MD, PDF or DOCX).")
    parser.add_argument(
        "--resumes", nargs="+", metavar="PATH",
        help="A folder of resumes, or one or more resume files (put this option last).",
    )
    args = parser.parse_args(argv)
    job_description = args.jd_file or args.jd
    if mode == "screen" and not (job_description and args.resumes):
        parser.error("screen needs a job description (text or --jd-file) and --resumes")
    if mode == "chat" and job_description:
        parser.error("chat takes no job description; describe the role in the chat")

    try:
        services = Services.from_settings()
        report, batch = None, None
        if args.resumes:
            # In chat, files start the session's uploads batch, which later /add commands join.
            session_batch = None if job_description else pools.new_batch(services.settings.uploads_dir)
            report, batch = services.use_paths(args.resumes, batch=session_batch)
            print(report.summary(), file=sys.stderr)
        services.check_ready(require_resumes=bool(job_description))
        if job_description:
            return _run_screen(services, job_description)
    except resume_rag.CLI_ERRORS as exc:
        print(f"Error: {resume_rag.cli_error_message(exc)}", file=sys.stderr)
        return 1

    initial = {"resume_folder": report.folder, "upload_folder": str(batch) if batch else None} if report else {}
    graph = build_graph(services, conversational=True)
    return cli.run(graph, services.settings.llm_model, report=report, initial_state=initial)


if __name__ == "__main__":
    sys.exit(main())
