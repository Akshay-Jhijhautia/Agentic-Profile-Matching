"""The ``resume-shortlister`` command line.

    resume-shortlister chat     [--resumes PATH ...] [--session ID]    chat with the agent
    resume-shortlister screen   "request" | --jd-file JD --resumes PATH ...   one-shot screening
    resume-shortlister index    FOLDER [--embedder local|openai|hashing]  index or refresh a resume folder
    resume-shortlister list     FOLDER                                 candidates indexed from a folder
    resume-shortlister sessions [--delete ID]                          saved chat sessions
    resume-shortlister clean    [--older-than DAYS] [--yes]            remove old uploads, indexes, sessions, answers
    resume-shortlister cache    [--clear]                              stored model answers (reproducible runs)
    resume-shortlister web      [--port PORT] [--no-browser] [--host]  the app in your browser (this machine only)
    resume-shortlister --version

PATH is a folder of resumes or one or more resume files (files are copied
into UPLOADS_DIR). Local data lives in RESUME_SHORTLISTER_HOME (~/.resume-shortlister).
Add ``--help`` after a command to see its options.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import __version__, cli, maintenance, pools, providers, sessions
from .agent.graph import build_graph, screen
from .conversation import reload_resumes
from .config import get_settings, tool_env
from .llm_cache import ResponseCache
from .rag import resume_rag
from .services import Services

DEFAULT_CLEAN_DAYS = 30


def _version() -> str:
    try:
        return version("resume-shortlister")
    except PackageNotFoundError:  # running from a source checkout
        return __version__


# --- chat and screen ---------------------------------------------------------------------------------


def _parser(prog: str, description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog=f"resume-shortlister {prog}", description=description)


def _resumes_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resumes", nargs="+", metavar="PATH",
                        help="A folder of resumes, or one or more resume files (put this option last).")


def run_agent(argv: list[str] | None = None, mode: str = "auto") -> int:
    """Chat, or screen once when a job description is given (``mode`` "chat"/"screen" insist)."""
    parser = _parser(mode if mode != "auto" else "chat", "Screen resumes for a job, or chat with the agent.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("jd", nargs="?", help="Job description text or a short request.")
    source.add_argument("--jd-file", help="Job description file (TXT, MD, PDF or DOCX).")
    parser.add_argument("--session", help="Resume a saved chat session (see: resume-shortlister sessions).")
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
            raise ValueError(f"No saved session '{session_id}'. See: resume-shortlister sessions")
        report, problem = reload_resumes(services, graph.get_state(sessions.config_for(session_id)).values)
        if report or problem:
            print(report.summary() if report else problem, file=sys.stderr)
    elif paths:
        # In chat, files start the session's uploads batch, which later /add commands join.
        batch = pools.new_batch(services.settings.uploads_dir)
        report, uploaded_to = services.use_paths(paths, batch=batch)
        print(report.summary(), file=sys.stderr)
        initial = {"resume_folder": report.folder, "pool_folder": str(services.pool.folder),
                   "upload_folder": str(uploaded_to) if uploaded_to else None}
    services.check_ready(require_resumes=False)
    return cli.run(graph, services.settings.model_label, report=report, initial_state=initial, session_id=session_id)


# --- folders ------------------------------------------------------------------------------------------


def _sync_folder(name: str, description: str, argv: list[str], with_embedder: bool = False) -> Services:
    """Parse ``FOLDER`` (and ``--embedder``), sync that folder and print its intake report."""
    parser = _parser(name, description)
    parser.add_argument("folder", help="Resume folder.")
    if with_embedder:
        parser.add_argument("--embedder", choices=["local", "openai", "hashing"],
                            help="Embeddings for this folder's index (default: EMBEDDER, local).")
    args = parser.parse_args(argv)
    services = Services.from_settings()
    embedder = pools.make_embedder(services.settings, args.embedder) if getattr(args, "embedder", None) else None
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
        print("No saved sessions yet. Start one with: resume-shortlister chat")
    for info in found:
        print(info.line())
    if found:
        print("\nResume one with: resume-shortlister chat --session <ID>")
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


def cache(argv: list[str]) -> int:
    parser = _parser("cache", "Stored model answers: the same question always gets the same answer.")
    parser.add_argument("--clear", action="store_true", help="Delete every stored answer; the model is asked again.")
    args = parser.parse_args(argv)
    settings = get_settings()
    store = ResponseCache(settings.llm_cache_db)
    if args.clear:
        print(f"Deleted {store.clear()} stored answers.")
        return 0
    stats = store.stats()
    print(f"{stats.answers} stored model answers, reused {stats.reused} times, {stats.size_mb:.1f} MB "
          f"in {resume_rag.display_path(str(store.path))}.")
    for kind, count in stats.by_kind.items():
        print(f"  {kind:<24} {count}")
    if not settings.llm_cache:
        print("LLM_CACHE is off: screenings ask the model every time.")
    return 0


def web(argv: list[str]) -> int:
    from .web.app import DEFAULT_HOST, DEFAULT_PORT, serve

    parser = _parser("web", "Open Resume Shortlister in your browser. It runs on this machine only.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to listen on (default: {DEFAULT_PORT}).")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser.")
    parser.add_argument("--host", default=tool_env("WEB_HOST") or DEFAULT_HOST,
                        help="Address to listen on (default: 127.0.0.1). Use 0.0.0.0 only inside a container.")
    args = parser.parse_args(argv)
    serve(args.port, open_browser=not args.no_browser, host=args.host)
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
    "cache": cache,
    "web": web,
}


# The command's name before 1.2; it still works, for now.
LEGACY_COMMAND = "profile-match"


def main(argv: list[str] | None = None) -> int:
    if argv is None and Path(sys.argv[0]).name == LEGACY_COMMAND:
        print(f"Note: {LEGACY_COMMAND} is now resume-shortlister; the old name will be removed in a later version.",
              file=sys.stderr)
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__.strip())
        return 0 if argv else 1
    if argv[0] in {"-V", "--version"}:
        print(f"resume-shortlister {_version()}")
        return 0
    command, rest = argv[0], argv[1:]
    handler = COMMANDS.get(command)
    if handler is None:
        print(f"Unknown command '{command}'. Use {', '.join(COMMANDS)}.", file=sys.stderr)
        return 2
    try:
        return handler(rest)
    except providers.cli_errors() as exc:
        print(f"Error: {providers.error_message(exc)}", file=sys.stderr)
        return 1


def agent_main(argv: list[str] | None = None, mode: str = "auto") -> int:
    """``python matching_agent.py``: chat, or screen when a JD is given."""
    try:
        return run_agent(argv, mode)
    except providers.cli_errors() as exc:
        print(f"Error: {providers.error_message(exc)}", file=sys.stderr)
        return 1
