"""Interactive terminal chat with the matching agent.

Each message from the user resumes the graph's human feedback loop. Progress
(requirements, screening, tool calls) is shown as the graph streams, and the
assistant's Markdown answers and reports are rendered with rich.

Commands: /help, /report, /reset, /exit
"""

from __future__ import annotations

import json
from collections.abc import Callable

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .agent.text import last_ai_text
from .pools import IntakeReport
from .rag import resume_rag
from .sessions import config_for, new_session_id

# Graph steps allowed per user message (a tool loop is two steps per call).
RECURSION_LIMIT = 60
# What a new conversation inherits from the current one: the resumes being screened.
CARRIED_OVER = ("resume_folder", "pool_folder", "upload_folder")
# Tools whose result is a Markdown report shown to the user verbatim, under this key.
DIRECT_REPORT_TOOLS = {"get_candidate_report": "report"}
MAX_ARG_CHARS = 80
EXAMPLES = (
    "Find me candidates with React and 3+ years experience",
    "Compare the top 3 matches side by side",
    "Why did Akira rank higher than Maria?",
    "Write interview questions for the top candidate",
)
HELP = """\
**Start a search** by describing the role, pasting a job description or giving a JD file path.
**Load resumes** with `/resumes <folder>`, `/add <files>`, or in words: "screen the resumes in ~/Downloads/backend".
**Then ask** for comparisons, explanations, interview questions or candidate details.

| Command | Effect |
| --- | --- |
| `/resumes <folder>` | Screen a folder of resumes (indexed incrementally) |
| `/add <file> ...` | Upload resume files (drag them into the terminal); they are screened together |
| `/report` | Show the latest shortlist report again |
| `/reset` | Start a new conversation |
| `/help` | Show this help |
| `/exit` | Quit (also Ctrl-D) |
"""


def _brief(args: dict) -> str:
    text = ", ".join(f"{key}={json.dumps(value)}" for key, value in args.items())
    return text if len(text) <= MAX_ARG_CHARS else text[: MAX_ARG_CHARS - 1] + "…"


def _tool_failed(message: ToolMessage) -> bool:
    try:
        result = json.loads(message.text)
    except json.JSONDecodeError:
        return message.status == "error"
    return isinstance(result, dict) and result.get("success") is False


def close_dangling_tool_calls(messages: list[AnyMessage], error: str) -> list[ToolMessage]:
    """Tool results for calls the last assistant message made but never got answered.

    The OpenAI API rejects a history with unanswered tool calls, so a turn
    that failed mid-tool-loop must close them before the conversation goes on.
    """
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    calls = last_ai.tool_calls if last_ai else []
    return [
        ToolMessage(json.dumps({"success": False, "error": error}), tool_call_id=call["id"], name=call["name"])
        for call in calls
        if call["id"] not in answered
    ]


class ChatSession:
    """One conversation (a saved session) with the agent graph."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        console: Console,
        initial_state: dict | None = None,
        session_id: str | None = None,
    ):
        self.graph = graph
        self.console = console
        # Seeds the first turn of a new conversation, e.g. the resumes loaded at startup.
        self.initial_state = initial_state or {}
        self.reset(session_id)

    @property
    def values(self) -> dict:
        return self.graph.get_state(self.config).values

    def reset(self, session_id: str | None = None) -> None:
        """Continue session ``session_id``, or start a new one that keeps the loaded resumes."""
        if getattr(self, "config", None):
            current = self.values
            self.initial_state = {key: current[key] for key in CARRIED_OVER if current.get(key)} or self.initial_state
        self.session_id = session_id or new_session_id()
        self.config = {**config_for(self.session_id), "recursion_limit": RECURSION_LIMIT}
        self.report: str | None = self.values.get("report")
        self.finished = False

    def send(self, text: str) -> None:
        """Deliver one user message and render what the agent does with it."""
        snapshot = self.graph.get_state(self.config)
        if snapshot.interrupts:
            payload = Command(resume=text)
        else:
            first_turn = not snapshot.values
            payload = {"messages": [("user", text)]} | (self.initial_state if first_turn else {})
        try:
            with self.console.status("Working…"):
                for update in self.graph.stream(payload, self.config, stream_mode="updates"):
                    self._render(update)
        except resume_rag.CLI_ERRORS as exc:
            self._recover(resume_rag.cli_error_message(exc))
        self.finished = not self.graph.get_state(self.config).next

    def _recover(self, error: str) -> None:
        """Record a failed turn as the assistant's reply, then wait for the next message."""
        self.console.print(f"[red]Error:[/red] {error}")
        messages = self.graph.get_state(self.config).values.get("messages", [])
        closing = [*close_dangling_tool_calls(messages, error), AIMessage(f"Sorry, that failed: {error}")]
        self.graph.update_state(self.config, {"messages": closing}, as_node="assistant")
        for _ in self.graph.stream(None, self.config, stream_mode="updates"):
            pass  # runs human_feedback up to its interrupt

    # --- rendering, one handler per graph node -------------------------------------------------

    def _render(self, update: dict) -> None:
        for node, change in update.items():
            handler = getattr(self, f"_on_{node}", None)
            if handler and change:
                handler(change)

    def _note(self, text: str) -> None:
        self.console.print(f"[dim]{text}[/dim]")

    def _on_await_resumes(self, change: dict) -> None:
        self.console.print(Markdown(change["messages"][-1].text))

    def _on_load_resumes(self, change: dict) -> None:
        self.console.print(Markdown(change["messages"][-1].text))

    def _on_route_intent(self, change: dict) -> None:
        if change["intent"] == "new_search":
            self._note("Starting a new search…")

    def _on_extract_requirements(self, change: dict) -> None:
        requirements = change["requirements"]
        self._note(f"Must have: {', '.join(r.label for r in requirements.must_have) or 'none'}")
        self._note(f"Nice to have: {', '.join(requirements.nice_to_have) or 'none'}")

    def _on_refine_requirements(self, change: dict) -> None:
        self._note("Updated the requirements:")
        self._on_extract_requirements(change)

    def _on_explain_changes(self, change: dict) -> None:
        diff = change["ranking_diff"]
        counts = {kind: len(diff.of_kind(kind)) for kind in ("new", "dropped", "up", "down")}
        self._note("Ranking changes: " + ", ".join(f"{n} {kind}" for kind, n in counts.items()))

    def _on_search_resumes(self, change: dict) -> None:
        candidates = change["candidates"]
        passing = sum(m.passes_must_haves for m in candidates)
        self._note(f"Screened {len(candidates)} resumes; {passing} pass every must-have.")

    def _on_assess_candidate(self, change: dict) -> None:
        for key, assessment in change["assessments"].items():
            self._note(f"Deep analysis of {key.split(':', 1)[1]}: {assessment.deep_score}")

    def _on_finalize_ranking(self, change: dict) -> None:
        decided = [m for m in change["shortlist"] if m.recommendation is not None]
        if decided:
            hires = sum(m.recommendation.decision == "hire" for m in decided)
            self._note(f"Final round: {hires} hire, {len(decided) - hires} no hire.")

    def _on_suggest_improvements(self, change: dict) -> None:
        improved = [m.name for m in change["shortlist"] if m.improvements]
        if improved:
            self._note(f"Improvement suggestions for borderline: {', '.join(improved)}")

    def _on_generate_report(self, change: dict) -> None:
        self.report = change["report"]
        self.show_report()
        self.console.print(Markdown(change["messages"][-1].text))

    def _on_assistant(self, change: dict) -> None:
        message = change["messages"][-1]
        for call in message.tool_calls:
            self._note(f"→ {call['name']}({_brief(call['args'])})")
        if message.text:
            self.console.print(Markdown(message.text))

    def _on_tools(self, change: dict) -> None:
        for message in change["messages"]:
            if _tool_failed(message):
                self.console.print(f"[yellow]← {message.name} failed: {message.text[:200]}[/yellow]")
                continue
            self._note(f"← {message.name} done")
            key = DIRECT_REPORT_TOOLS.get(message.name)
            if key:  # computed reports are shown as-is, never retold by the model
                report = json.loads(message.text)[key]
                self.console.print(Panel(Markdown(report), title="Candidate report", border_style="cyan"))

    def show_report(self) -> None:
        if self.report:
            self.console.print(Panel(Markdown(self.report), title="Shortlist report", border_style="cyan"))
        else:
            self.console.print("No report yet. Describe a role to start a search.")


# --- REPL ------------------------------------------------------------------------------------------

STOP = object()
COMMANDS: dict[str, Callable[[ChatSession], object]] = {
    "/exit": lambda session: STOP,
    "/quit": lambda session: STOP,
    "/help": lambda session: session.console.print(Markdown(HELP)),
    "/report": ChatSession.show_report,
    "/reset": lambda session: (
        session.reset(),
        session.console.print(f"Started a new conversation (session {session.session_id}); loaded resumes stay loaded."),
    ),
}


def run(
    graph: CompiledStateGraph,
    model_name: str,
    console: Console | None = None,
    report: IntakeReport | None = None,
    initial_state: dict | None = None,
    session_id: str | None = None,
) -> int:
    """Chat until the user exits; returns the process exit code. ``session_id`` resumes
    a saved session."""
    console = console or Console()
    session = ChatSession(graph, console, initial_state, session_id)
    examples = "\n".join(f"  • {example}" for example in EXAMPLES)
    loaded = (
        f"Screening {report.ready} resumes from {report.folder}." if report
        else "No resumes loaded yet: use /resumes <folder> or /add <files> (drag files into the terminal)."
    )
    console.print(Panel(
        f"Session {session.session_id}. {loaded}\n\n"
        "Describe a role, paste a job description or give a JD file path. For example:\n"
        f"{examples}\n\nCommands: /resumes <folder> /add <files> /help /report /reset /exit",
        title=f"Matching agent ({model_name})", border_style="cyan",
    ))
    if session_id and session.values:
        requirements = session.values.get("requirements")
        role = requirements.role_title if requirements else "no role yet"
        console.print(f"[dim]Resumed: {role}. Last reply:[/dim]")
        console.print(Markdown(last_ai_text(session.values) or "(none)"))
    while True:
        try:
            text = console.input("\n[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        command = COMMANDS.get(text.lower())
        if command:
            if command(session) is STOP:
                break
            continue
        session.send(text)
        if session.finished:
            break
    console.print("Goodbye!")
    return 0
