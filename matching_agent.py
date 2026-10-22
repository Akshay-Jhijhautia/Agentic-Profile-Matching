"""The matching agent: a LangGraph pipeline that screens resumes for a job.

    START → parse_jd → extract_requirements → search_resumes
          → rank_candidates → generate_report → END

Usage (one-shot screening):
    python matching_agent.py --jd-file job_descriptions/frontend_react.txt
    python matching_agent.py "Find me candidates with React and 3+ years experience"
    python main.py screen ...   (same options)
"""

from __future__ import annotations

import argparse
import sys
from itertools import pairwise

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from profile_matching.agent.nodes import PipelineNodes
from profile_matching.agent.state import AgentState
from profile_matching.rag import fs_tools, resume_rag
from profile_matching.services import Services

PIPELINE = ("parse_jd", "extract_requirements", "search_resumes", "rank_candidates", "generate_report")


def build_graph(services: Services, checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """The screening pipeline as a compiled graph, bound to ``services``."""
    nodes = PipelineNodes(services)
    builder = StateGraph(AgentState)
    for name in PIPELINE:
        builder.add_node(name, getattr(nodes, name))
    builder.add_edge(START, PIPELINE[0])
    for current, following in pairwise(PIPELINE):
        builder.add_edge(current, following)
    builder.add_edge(PIPELINE[-1], END)
    return builder.compile(checkpointer=checkpointer)


def screen(services: Services, job_description: str) -> AgentState:
    """Run the pipeline once for a JD (text or file path) and return the final state."""
    return build_graph(services).invoke({"messages": [HumanMessage(job_description)]})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Screen indexed resumes for a job description.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("jd", nargs="?", help="Job description text or a short request.")
    source.add_argument("--jd-file", help="Job description file (TXT, MD, PDF or DOCX).")
    parser.add_argument("--out", help="Also save the Markdown report to this file.")
    args = parser.parse_args(argv)

    try:
        services = Services.from_settings()
        services.check_ready()
        state = screen(services, args.jd_file or args.jd)
    except resume_rag.CLI_ERRORS as exc:
        print(f"Error: {resume_rag.cli_error_message(exc)}", file=sys.stderr)
        return 1

    print(state["report"])
    if args.out:
        written = fs_tools.write_file(args.out, state["report"])
        if not written["success"]:
            print(f"Error: {written['error']}", file=sys.stderr)
            return 1
        print(f"Saved to {resume_rag.display_path(written['filepath'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
