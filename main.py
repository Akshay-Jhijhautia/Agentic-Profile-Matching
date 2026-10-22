"""Entry point for the agentic resume matcher.

    python main.py index [resumes/] [options]   build or refresh the vector index
    python main.py stats | list                 inspect the index
    python main.py match --jd-file JD [options] rank resumes for a job description (no agent)
    python main.py screen --jd-file JD | "request"  run the matching agent's screening (reports saved)
    python main.py chat                         chat with the matching agent

Each command forwards its options to the module that implements it; add
``--help`` after a command to see them.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import matching_agent
from profile_matching.rag import job_matcher, resume_rag

# Command name -> handler taking the remaining argv. Index commands keep their
# name because ``resume_rag.main`` dispatches on it.
COMMANDS: dict[str, Callable[[list[str]], int]] = {
    "index": lambda rest: resume_rag.main(["index", *rest]),
    "stats": lambda rest: resume_rag.main(["stats", *rest]),
    "list": lambda rest: resume_rag.main(["list", *rest]),
    "match": job_matcher.main,
    "screen": matching_agent.main,
    "chat": lambda rest: matching_agent.main([]),
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__.strip())
        return 0 if argv else 1
    command, rest = argv[0], argv[1:]
    handler = COMMANDS.get(command)
    if handler is None:
        print(f"Unknown command '{command}'. Use {', '.join(COMMANDS)}.", file=sys.stderr)
        return 2
    return handler(rest)


if __name__ == "__main__":
    sys.exit(main())
