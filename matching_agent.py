"""The matching agent: a LangGraph agent that screens resumes for a job.

The graph is built in ``profile_matching/agent/graph.py`` (see its docstring
for the full flow) and re-exported here; the command line lives in
``profile_matching/app.py`` and is installed as ``profile-match``.

    python matching_agent.py                                       chat (same as: profile-match chat)
    python matching_agent.py --jd-file job.pdf --resumes ~/cvs/     one-shot screening
"""

from __future__ import annotations

import sys

from profile_matching.agent.graph import (
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
from profile_matching.app import agent_main as main
from profile_matching.services import Services

__all__ = [
    "ASSESS", "AWAIT", "EXPLAIN", "FINALIZE", "IMPROVE", "PIPELINE", "REPORT",
    "Services", "build_graph", "main", "screen",
]


if __name__ == "__main__":
    sys.exit(main())
