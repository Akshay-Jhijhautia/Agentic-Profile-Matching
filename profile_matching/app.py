"""The ``profile-match`` command line.

    profile-match chat     [--resumes PATH ...] [--session ID]    chat with the agent
    profile-match screen   "request" | --jd-file JD --resumes PATH ...   one-shot screening
    profile-match index    FOLDER [--embedder openai|hashing]     index or refresh a resume folder
    profile-match list     FOLDER                                 candidates indexed from a folder
    profile-match sessions [--delete ID]                          saved chat sessions
    profile-match clean    [--older-than DAYS] [--yes]            remove old uploads, indexes, sessions
    profile-match --version

PATH is a folder of resumes or one or more resume files (files are copied
into UPLOADS_DIR). Local data lives in PROFILE_MATCH_HOME (~/.profile-match).
Add ``--help`` after a command to see its options.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import __version__, cli, maintenance, pools, sessions
from .agent.graph import build_graph, screen
from .config import get_settings
from .rag import resume_rag
from .services import Services

DEFAULT_CLEAN_DAYS = 30


def _version() -> str:
    try:
        return version("profile-match")
    except PackageNotFoundError:  # running from a source checkout
        return __version__


# --- chat and screen ---------------------------------------------------------------------------------


def _parser(prog: str, description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog=f"profile-match {prog}", description=description)


def _resumes_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resumes", nargs="+", metavar="PATH",
                        help="A folder of resumes, or one or more resume files (put this option last).")


def run_agent(argv: list[str] | None = None, mode: str = "auto") -> int:
    """Chat, or screen once when a job description is given (``mode`` "chat"/"screen" insist)."""
    parser = _parser(mode if mode != "auto" else "chat", "Screen resumes for a job, or chat with the agent.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("jd", nargs="?", help="Job description text or a short request.")
    source.add_argument("--jd-file", help="Job description file (TXT, MD, PDF or DOCX).")
    parser.add_argument("--session", help="Resume a saved chat session (see: profile-match sessions).")
    _resumes_option(parser)
    args = parser.parse_args(argv)
    job_description = args.jd_file or args.jd
    if mode == "screen" and not (job_description and args.resumes):
        parser.error("screen needs a job description (text or --jd-file) and --resumes")
    if mode == "chat" and job_description:
        parser.error("chat takes no job description; describe the role in the chat")
    if args.session and (job_description or args.resumes):
        parser.error("--session resumes a saved chat as it was; load other resumes inside the chat")

    services = Services.from_settings()
    if job_description:
        if args.resumes:
            report, _ = services.use_paths(args.resumes)
            print(report.summary(), file=sys.stderr)
        services.check_ready()
        state = screen(services, job_description)
        print(state["report"])
        if state.get("report_dir"):
            print(f"Reports saved to {state['report_dir']}", file=sys.stderr)
        return 0
    return _chat(services, args.resumes, args.session)


def _chat(services: Services, paths: list[str] | None, session_id: str | None) -> int:
    saver = sessions.open_saver(services.settings.sessions_db)
    graph = build_graph(services, checkpointer=saver, conversational=True)
    report, initial = None, {}
    if session_id:
        if not sessions.exists(saver, session_id):
            raise ValueError(f"No saved session '{session_id}'. See: profile-match sessions")
        saved = graph.get_state(sessions.config_for(session_id)).values
        folder = saved.get("pool_folder")
        if folder and Path(folder).is_dir():
            # Shown as the session knew it ("your uploaded resumes" rather than a batch path).
            report = replace(services.use_folder(folder), folder=saved.get("resume_folder") or folder)
            print(report.summary(), file=sys.stderr)
        elif folder:
            print(f"The resumes of this session ({folder}) are gone; load new ones in the chat.", file=sys.stderr)
    elif paths:
        # In chat, files start the session's uploads batch, which later /add commands join.
        batch = pools.new_batch(services.settings.uploads_dir)
        report, uploaded_to = services.use_paths(paths, batch=batch)
        print(report.summary(), file=sys.stderr)
        initial = {"resume_folder": report.folder, "pool_folder": str(services.pool.folder),
                   "upload_folder": str(uploaded_to) if uploaded_to else None}
    services.check_ready(require_resumes=False)
    return cli.run(graph, services.settings.llm_model, report=report, initial_state=initial, session_id=session_id)


# --- folders ------------------------------------------------------------------------------------------


def _sync_folder(name: str, description: str, argv: list[str], with_embedder: bool = False) -> Services:
    """Parse ``FOLDER`` (and ``--embedder``), sync that folder and print its intake report."""
    parser = _parser(name, description)
    parser.add_argument("folder", help="Resume folder.")
    if with_embedder:
        parser.add_argument("--embedder", choices=["openai", "hashing"],
                            help="Embeddings for a new folder index (default: EMBEDDER, openai).")
    args = parser.parse_args(argv)
    embedder = resume_rag.HashingEmbedder() if getattr(args, "embedder", None) == "hashing" else None
    services = Services.from_settings()
    print(services.use_folder(args.folder, embedder=embedder).summary())
    return services


def index(argv: list[str]) -> int:
    _sync_folder("index", "Index (or refresh) a folder of resumes.", argv, with_embedder=True)
    return 0


def list_candidates(argv: list[str]) -> int:
    services = _sync_folder("list", "List the candidates indexed from a folder.", argv)
    for profile in services.candidates.all():
        meta = profile.metadata
        years = f"{meta['experience_years']:g} yrs" if meta["experience_years"] is not None else "? yrs"
        print(f"  {profile.name:<24} {years:>8}  {meta['highest_degree'] or '-':<11} "
              f"{len(meta['skills']):>2} skills  {profile.summary()['resume']}")
    return 0


# --- sessions and cleanup ------------------------------------------------------------------------------


def list_sessions(argv: list[str]) -> int:
    parser = _parser("sessions", "List saved chat sessions, newest first, or delete one.")
    parser.add_argument("--delete", metavar="ID", help="Delete a saved session.")
    args = parser.parse_args(argv)
    services = Services.from_settings()
    saver = sessions.open_saver(services.settings.sessions_db)
    if args.delete:
        if not sessions.exists(saver, args.delete):
            raise ValueError(f"No saved session '{args.delete}'")
        sessions.delete_session(saver, args.delete)
        print(f"Deleted session {args.delete}.")
        return 0
    found = sessions.list_sessions(build_graph(services, checkpointer=saver, conversational=True), saver)
    if not found:
        print("No saved sessions yet. Start one with: profile-match chat")
    for info in found:
        print(info.line())
    if found:
        print("\nResume one with: profile-match chat --session <ID>")
    return 0


def clean(argv: list[str]) -> int:
    parser = _parser("clean", "Remove old upload batches and sessions, and indexes of deleted folders.")
    parser.add_argument("--older-than", type=int, default=DEFAULT_CLEAN_DAYS, metavar="DAYS",
                        help=f"Age of uploads and sessions to remove (default: {DEFAULT_CLEAN_DAYS}).")
    parser.add_argument("--yes", action="store_true", help="Delete; without it, only list what would go.")
    args = parser.parse_args(argv)
    settings = get_settings()
    stale = maintenance.find_stale(settings, args.older_than)
    if not stale:
        print("Nothing to clean.")
        return 0
    print("\n".join(item.line() for item in stale))
    if not args.yes:
        print(f"\n{len(stale)} item(s) would be removed. Run again with --yes to delete them.")
        return 0
    maintenance.remove(settings, stale)
    print(f"\nRemoved {len(stale)} item(s).")
    return 0


# --- entry point ----------------------------------------------------------------------------------------

# Command name -> handler taking the remaining arguments.
COMMANDS: dict[str, Callable[[list[str]], int]] = {
    "chat": lambda rest: run_agent(rest, mode="chat"),
    "screen": lambda rest: run_agent(rest, mode="screen"),
    "index": index,
    "list": list_candidates,
    "sessions": list_sessions,
    "clean": clean,
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__.strip())
        return 0 if argv else 1
    if argv[0] in {"-V", "--version"}:
        print(f"profile-match {_version()}")
        return 0
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


def agent_main(argv: list[str] | None = None, mode: str = "auto") -> int:
    """``python matching_agent.py``: chat, or screen when a JD is given."""
    try:
        return run_agent(argv, mode)
    except resume_rag.CLI_ERRORS as exc:
        print(f"Error: {resume_rag.cli_error_message(exc)}", file=sys.stderr)
        return 1
