"""A conversation with the agent, independent of how it is shown.

``Conversation`` delivers one user message per ``turn`` and yields ``Event``s
as the graph works: progress notes, the assistant's answers, the shortlist
report, candidate reports and errors. The terminal chat (``cli``) prints them;
the web app (``web``) streams them to the browser. A failed turn is recorded as
the assistant's reply, so the conversation can always go on.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from . import providers
from .pools import IntakeReport
from .services import Services
from .sessions import config_for, new_session_id

# Graph steps allowed per user message (a tool loop is two steps per call).
RECURSION_LIMIT = 60
# What a new conversation inherits from the current one: the resumes being screened.
CARRIED_OVER = ("resume_folder", "pool_folder", "upload_folder")
# Tools whose result is a Markdown report shown to the user verbatim, under this key.
DIRECT_REPORT_TOOLS = {"get_candidate_report": "report"}
MAX_ARG_CHARS = 80


@dataclass(frozen=True)
class Event:
    """Something the user should see. ``kind`` is one of:

    - ``note``: progress ("Screened 100 resumes; 10 pass every must-have.")
    - ``message``: an answer or question from the agent (Markdown)
    - ``report``: the shortlist report (Markdown)
    - ``candidate_report``: one candidate's detailed report (Markdown)
    - ``tool_failed``: a tool the agent called failed (it can recover)
    - ``error``: the turn failed; the conversation goes on
    """

    kind: str
    text: str
    node: str = ""


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


class Conversation:
    """One conversation (a saved session) with the agent graph."""

    def __init__(self, graph: CompiledStateGraph, initial_state: dict | None = None, session_id: str | None = None):
        self.graph = graph
        # Seeds the first turn of a new conversation, e.g. the resumes loaded at startup.
        self.initial_state = initial_state or {}
        self.reset(session_id)

    @property
    def values(self) -> dict:
        return self.graph.get_state(self.config).values

    @property
    def waiting(self) -> bool:
        """Whether the agent waits for the user (always, between turns, until they say goodbye)."""
        return bool(self.graph.get_state(self.config).next) or not self.values

    def reset(self, session_id: str | None = None) -> None:
        """Continue session ``session_id``, or start a new one that keeps the loaded resumes."""
        if getattr(self, "config", None):
            current = self.values
            self.initial_state = {key: current[key] for key in CARRIED_OVER if current.get(key)} or self.initial_state
        self.session_id = session_id or new_session_id()
        self.config = {**config_for(self.session_id), "recursion_limit": RECURSION_LIMIT}
        self.report: str | None = self.values.get("report")
        self.finished = False

    def turn(self, text: str) -> Iterator[Event]:
        """Deliver one user message and yield what the agent does with it."""
        snapshot = self.graph.get_state(self.config)
        if snapshot.interrupts:
            payload = Command(resume=text)
        else:
            first_turn = not snapshot.values
            payload = {"messages": [("user", text)]} | (self.initial_state if first_turn else {})
        try:
            for update in self.graph.stream(payload, self.config, stream_mode="updates"):
                for node, change in update.items():
                    for event in describe(node, change):
                        if event.kind == "report":
                            self.report = event.text
                        yield event
        except providers.cli_errors() as exc:
            error = providers.error_message(exc)
            yield Event("error", error)
            self._recover(error)
        self.finished = not self.graph.get_state(self.config).next

    def _recover(self, error: str) -> None:
        """Record a failed turn as the assistant's reply, then wait for the next message."""
        messages = self.graph.get_state(self.config).values.get("messages", [])
        closing = [*close_dangling_tool_calls(messages, error), AIMessage(f"Sorry, that failed: {error}")]
        self.graph.update_state(self.config, {"messages": closing}, as_node="assistant")
        for _ in self.graph.stream(None, self.config, stream_mode="updates"):
            pass  # runs human_feedback up to its interrupt


def reload_resumes(services: Services, values: dict) -> tuple[IntakeReport | None, str | None]:
    """Reactivate the resumes a saved session was screening.

    Returns ``(intake report, None)``, ``(None, why not)`` when its folder is
    gone, or ``(None, None)`` when it had no resumes.
    """
    folder = values.get("pool_folder")
    if not folder:
        return None, None
    if not Path(folder).is_dir():
        return None, f"The resumes of this session ({folder}) are gone; load new ones."
    # Shown as the session knew it ("your uploaded resumes" rather than a batch path).
    return replace(services.use_folder(folder), folder=values.get("resume_folder") or folder), None


# --- graph steps as events --------------------------------------------------------------------------


def _brief(args: dict) -> str:
    text = ", ".join(f"{key}={json.dumps(value)}" for key, value in args.items())
    return text if len(text) <= MAX_ARG_CHARS else text[: MAX_ARG_CHARS - 1] + "…"


def _tool_failed(message: ToolMessage) -> bool:
    try:
        result = json.loads(message.text)
    except json.JSONDecodeError:
        return message.status == "error"
    return isinstance(result, dict) and result.get("success") is False


def _requirement_notes(requirements) -> list[str]:
    return [
        f"Must have: {', '.join(r.label for r in requirements.must_have) or 'none'}",
        f"Nice to have: {', '.join(requirements.nice_to_have) or 'none'}",
    ]


def describe(node: str, change: dict | None) -> list[Event]:
    """What one graph step's update means to the user."""
    if not change:
        return []
    notes: list[str] = []
    events: list[Event] = []
    if node in ("await_resumes", "load_resumes", "confirm_requirements"):
        events.append(Event("message", change["messages"][-1].text, node))
    elif node == "route_intent" and change["intent"] == "new_search":
        notes.append("Starting a new search…")
    elif node == "extract_requirements":
        notes += _requirement_notes(change["requirements"])
    elif node == "refine_requirements":
        notes += ["Updated the requirements:", *_requirement_notes(change["requirements"])]
    elif node == "accept_requirements":
        notes.append("Screening with these requirements…")
    elif node == "explain_changes":
        counts = {kind: len(change["ranking_diff"].of_kind(kind)) for kind in ("new", "dropped", "up", "down")}
        notes.append("Ranking changes: " + ", ".join(f"{n} {kind}" for kind, n in counts.items()))
    elif node == "search_resumes":
        candidates = change["candidates"]
        passing = sum(m.passes_must_haves for m in candidates)
        notes.append(f"Screened {len(candidates)} resumes; {passing} pass every must-have.")
    elif node == "assess_candidate":
        # Keys are "pool:requirements:candidate" (``screening.assessment_key``).
        notes += [f"Deep analysis of {key.rsplit(':', 1)[1]}: {a.deep_score}" for key, a in change["assessments"].items()]
    elif node == "finalize_ranking":
        decided = [m for m in change["shortlist"] if m.recommendation is not None]
        if decided:
            advance = sum(m.recommendation.decision == "interview" for m in decided)
            notes.append(f"Final round: {advance} recommended for interview, {len(decided) - advance} not now.")
    elif node == "suggest_improvements":
        improved = [m.name for m in change["shortlist"] if m.improvements]
        if improved:
            notes.append(f"Improvement suggestions for borderline: {', '.join(improved)}")
    elif node == "generate_report":
        events += [Event("report", change["report"], node), Event("message", change["messages"][-1].text, node)]
    elif node == "assistant":
        message = change["messages"][-1]
        notes += [f"→ {call['name']}({_brief(call['args'])})" for call in message.tool_calls]
        if message.text:
            events.append(Event("message", message.text, node))
    elif node == "tools":
        for message in change["messages"]:
            if _tool_failed(message):
                events.append(Event("tool_failed", f"{message.name} failed: {message.text[:200]}", node))
                continue
            events.append(Event("note", f"← {message.name} done", node))
            key = DIRECT_REPORT_TOOLS.get(message.name)
            if key:  # computed reports are shown as-is, never retold by the model
                events.append(Event("candidate_report", json.loads(message.text)[key], node))
    return [Event("note", note, node) for note in notes] + events
