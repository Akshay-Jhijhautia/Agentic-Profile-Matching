"""Interactive terminal chat with the matching agent.

A ``conversation.Conversation`` whose events are printed with rich: progress
(requirements, screening, tool calls) as the graph streams, the assistant's
Markdown answers, and reports in panels.

Commands: /help, /report, /reset, /exit
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .agent.text import last_ai_text
from .conversation import Conversation, Event
from .pools import IntakeReport

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


class ChatSession(Conversation):
    """A conversation printed to the terminal."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        console: Console,
        initial_state: dict | None = None,
        session_id: str | None = None,
    ):
        self.console = console
        super().__init__(graph, initial_state, session_id)

    def send(self, text: str) -> None:
        """Deliver one user message and print what the agent does with it."""
        with self.console.status("Working…"):
            for event in self.turn(text):
                self.show(event)

    def show(self, event: Event) -> None:
        if event.kind == "note":
            self.console.print(f"[dim]{event.text}[/dim]")
        elif event.kind == "message":
            self.console.print(Markdown(event.text))
        elif event.kind == "report":
            self.show_report()
        elif event.kind == "candidate_report":
            self.console.print(Panel(Markdown(event.text), title="Candidate report", border_style="cyan"))
        elif event.kind == "tool_failed":
            self.console.print(f"[yellow]← {event.text}[/yellow]")
        elif event.kind == "error":
            self.console.print(f"[red]Error:[/red] {event.text}")

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
