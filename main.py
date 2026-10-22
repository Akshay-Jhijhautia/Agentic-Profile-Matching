"""Entry point for the agentic resume matcher.

    python main.py chat   [--resumes PATH ...]                         chat with the matching agent
    python main.py screen --jd-file JD | "request" --resumes PATH ...   one-shot screening
    python main.py index  FOLDER [--embedder openai|hashing]           index or refresh a resume folder
    python main.py list   FOLDER                                       candidates indexed from a folder

PATH is a folder of resumes or one or more resume files (files are copied
into UPLOADS_DIR). Every folder has its own index, refreshed incrementally
whenever it is used; add ``--help`` after a command to see its options.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import matching_agent
from profile_matching.rag import resume_rag
from profile_matching.services import Services


def _folder_command(
    name: str, description: str, argv: list[str], with_embedder: bool = False
) -> tuple[Services, argparse.Namespace]:
    """Parse ``[FOLDER]`` (and ``--embedder``), sync that folder and print its intake report."""
    parser = argparse.ArgumentParser(prog=f"main.py {name}", description=description)
    parser.add_argument("folder", help="Resume folder.")
    if with_embedder:
        parser.add_argument("--embedder", choices=["openai", "hashing"],
                            help="Embeddings for a new folder index (default: EMBEDDER, openai).")
    args = parser.parse_args(argv)
    embedder = resume_rag.HashingEmbedder() if getattr(args, "embedder", None) == "hashing" else None
    services = Services.from_settings()
    print(services.use_folder(args.folder, embedder=embedder).summary())
    return services, args


def index(argv: list[str]) -> int:
    _folder_command("index", "Index (or refresh) a folder of resumes.", argv, with_embedder=True)
    return 0


def list_candidates(argv: list[str]) -> int:
    services, _ = _folder_command("list", "List the candidates indexed from a folder.", argv)
    for profile in services.candidates.all():
        meta = profile.metadata
        years = f"{meta['experience_years']:g} yrs" if meta["experience_years"] is not None else "? yrs"
        print(f"  {profile.name:<24} {years:>8}  {meta['highest_degree'] or '-':<11} "
              f"{len(meta['skills']):>2} skills  {profile.summary()['resume']}")
    return 0


# Command name -> handler taking the remaining argv.
COMMANDS: dict[str, Callable[[list[str]], int]] = {
    "chat": lambda rest: matching_agent.main(rest, mode="chat"),
    "screen": lambda rest: matching_agent.main(rest, mode="screen"),
    "index": index,
    "list": list_candidates,
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
    try:
        return handler(rest)
    except resume_rag.CLI_ERRORS as exc:
        print(f"Error: {resume_rag.cli_error_message(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
