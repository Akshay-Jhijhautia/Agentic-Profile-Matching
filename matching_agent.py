"""The matching agent: a LangGraph agent that screens resumes for a job.

The graph is built in ``resume_shortlister/agent/graph.py`` (see its docstring
for the full flow) and re-exported here; the command line lives in
``resume_shortlister/app.py`` and is installed as ``resume-shortlister``.

    python matching_agent.py                                       chat (same as: resume-shortlister chat)
    python matching_agent.py --jd-file job.pdf --resumes ~/cvs/     one-shot screening
"""

from __future__ import annotations

import sys

from resume_shortlister.agent.graph import (
    ASSESS,
    AWAIT,
    EXPLAIN,
    FINALIZE,
    IMPROVE,
    PIPELINE,
    REPORT,
    build_graph,
    screen,
)
from resume_shortlister.app import agent_main as main
from resume_shortlister.services import Services

__all__ = [
    "ASSESS", "AWAIT", "EXPLAIN", "FINALIZE", "IMPROVE", "PIPELINE", "REPORT",
    "Services", "build_graph", "main", "screen",
]


if __name__ == "__main__":
    sys.exit(main())
