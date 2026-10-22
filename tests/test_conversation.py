import builtins
import io
import json

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from rich.console import Console

import matching_agent
from profile_matching import cli
from profile_matching.agent.routing import IntentRouter
from profile_matching.schemas import (
    CandidateNote,
    ChangeNarrative,
    ComparisonVerdict,
    IntentDecision,
    Requirements,
)
from tests.fakes import SCREENING_RESPONSES, FakeChatModel

ROLE = Requirements.model_validate({
    "role_title": "Python Backend Engineer", "must_have": [{"skill": "Python", "min_years": None}],
    "nice_to_have": ["Django"], "min_total_years": None, "min_degree": None,
    "ideal_profile": "Python backend engineer.",
})
VERDICT = ComparisonVerdict(
    ranking=["john_doe"], notes=[CandidateNote(candidate_id="john_doe", strengths=["Django"], weaknesses=[])],
    summary="John fits best.",
)


def route_by_keyword(messages) -> IntentDecision:
    """Stand-in router: the latest user message decides the intent."""
    latest = messages[1].content.rsplit("Latest user message: ", 1)[1].lower()
    if "find" in latest or "engineer" in latest:
        return IntentDecision(intent="new_search")
    if "goodbye" in latest:
        return IntentDecision(intent="end")
    return IntentDecision(intent="ask")


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture
def chat(services):
    """A chat session over the hashing index; set ``services.model.replies`` per test."""
    services.model = FakeChatModel(
        {Requirements: ROLE, IntentDecision: route_by_keyword, ComparisonVerdict: VERDICT, **SCREENING_RESPONSES}
    )
    graph = matching_agent.build_graph(services, conversational=True)
    return cli.ChatSession(graph, Console(file=io.StringIO(), width=120))


def output(session: cli.ChatSession) -> str:
    return session.console.file.getvalue()


def state(session: cli.ChatSession) -> dict:
    return session.graph.get_state(session.config).values


# --- graph shape ------------------------------------------------------------------------------------


def test_conversational_graph_has_the_feedback_loop(services):
    services.model = FakeChatModel({})
    graph = matching_agent.build_graph(services, conversational=True).get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    assert {
        ("__start__", "route_intent"),
        ("route_intent", "parse_jd"), ("route_intent", "assistant"), ("route_intent", "__end__"),
        ("generate_report", "human_feedback"),
        ("assistant", "tools"), ("tools", "assistant"), ("assistant", "human_feedback"),
        ("human_feedback", "route_intent"),
    } <= edges


# --- turns ------------------------------------------------------------------------------------------


def test_first_request_runs_the_pipeline_then_waits_for_feedback(chat):
    chat.send("Find me a Python backend engineer")
    assert chat.graph.get_state(chat.config).next == ("human_feedback",)
    assert not chat.finished
    assert state(chat)["requirements"] == ROLE and chat.report.startswith("# Shortlist")
    text = output(chat)
    assert "Starting a new search" in text and "Must have: Python" in text
    assert "Screened 12 resumes" in text and "Shortlist report" in text


def test_compare_request_uses_the_tool_loop(chat, services):
    chat.send("Find me a Python backend engineer")
    top3 = [m.id for m in state(chat)["shortlist"][:3]]
    services.model.replies = [
        tool_call("compare_candidates", {"candidate_ids": top3}),
        AIMessage("| | John | ... |\n|---|---|---|\n\nJohn fits best."),
    ]
    chat.send("Compare the top 3 matches side by side")

    messages = state(chat)["messages"]
    tool_result = next(m for m in messages if isinstance(m, ToolMessage))
    result = json.loads(tool_result.text)
    assert result["role"] == "Python Backend Engineer"  # the tool read the role from state
    assert [c["id"] for c in result["candidates"]] and result["verdict"]["summary"] == "John fits best."
    assert messages[-1].text.endswith("John fits best.")

    # The assistant saw the shortlist in its system prompt, so it could pick the top 3 ids.
    _, first_prompt = next(call for call in services.model.calls if call[0] == "tools")
    assert isinstance(first_prompt[0], SystemMessage)
    assert f"1. {state(chat)['shortlist'][0].name} ({top3[0]})" in first_prompt[0].content

    text = output(chat)
    assert "→ compare_candidates" in text and "← compare_candidates done" in text
    assert chat.graph.get_state(chat.config).next == ("human_feedback",)


def test_a_question_before_any_search_goes_to_the_assistant(chat, services):
    services.model.replies = [AIMessage("Describe a role to start.")]
    chat.send("What can you do?")
    _, prompt = services.model.calls[-1]
    assert "No role is set yet" in prompt[0].content
    assert state(chat)["messages"][-1].text == "Describe a role to start."


@pytest.mark.parametrize("farewell", ["bye", "Exit", "done!", "goodbye for now"])
def test_ending_the_conversation(chat, farewell):
    chat.send("Find me a Python backend engineer")
    chat.send(farewell)
    assert chat.finished


def test_tool_errors_are_returned_to_the_model(chat, services):
    services.model.replies = [
        tool_call("compare_candidates", {"candidate_ids": ["Nobody Here", "john_doe"]}),
        AIMessage("I could not find that candidate."),
    ]
    chat.send("Compare Nobody Here with John")
    assert "← compare_candidates failed" in output(chat)
    assert state(chat)["messages"][-1].text == "I could not find that candidate."


# --- errors -----------------------------------------------------------------------------------------


def test_failed_turn_is_recorded_and_the_chat_continues(chat, services):
    request = httpx.Request("POST", "https://api.openai.com")
    services.model.replies = [openai.APIConnectionError(request=request), AIMessage("Back online.")]
    chat.send("Tell me about John")
    assert "could not reach the OpenAI API" in output(chat)
    assert state(chat)["messages"][-1].text.startswith("Sorry, that failed")
    assert chat.graph.get_state(chat.config).interrupts and not chat.finished

    chat.send("Tell me about John again")
    assert state(chat)["messages"][-1].text == "Back online."


def test_close_dangling_tool_calls():
    messages = [
        HumanMessage("compare"),
        AIMessage("", tool_calls=[{"name": "a", "args": {}, "id": "1"}, {"name": "b", "args": {}, "id": "2"}]),
        ToolMessage("{}", tool_call_id="1", name="a"),
    ]
    (closing,) = cli.close_dangling_tool_calls(messages, "boom")
    assert closing.tool_call_id == "2" and json.loads(closing.text)["error"] == "boom"
    assert cli.close_dangling_tool_calls(messages[:1], "boom") == []


# --- routing -----------------------------------------------------------------------------------------


def test_router_gives_the_model_role_and_previous_reply(services):
    services.model = FakeChatModel({IntentDecision: IntentDecision(intent="ask")})
    router = IntentRouter(services)
    update = router({
        "requirements": ROLE,
        "messages": [AIMessage("Shall I compare them?"), HumanMessage("yes please")],
    })
    assert update == {"intent": "ask"}
    _, messages = services.model.calls[0]
    assert messages[1].content.splitlines() == [
        "Current role: Python Backend Engineer",
        "Previous assistant message: Shall I compare them?",
        "Latest user message: yes please",
    ]


def test_router_handles_exit_words_without_the_model(services):
    services.model = FakeChatModel({})
    assert IntentRouter(services)({"messages": [HumanMessage("Quit.")]}) == {"intent": "end"}
    assert services.model.calls == []


# --- REPL ---------------------------------------------------------------------------------------------


def test_repl_commands_and_conversation(chat, monkeypatch):
    inputs = iter(["", "/help", "/report", "Find me a Python backend engineer", "/reset", "/report", "/exit"])
    monkeypatch.setattr(builtins, "input", lambda *_: next(inputs))
    console = chat.console
    assert cli.run(chat.graph, "gpt-test", console=console) == 0
    text = output(chat)
    assert "Matching agent (gpt-test)" in text
    assert "/reset" in text and "Start a new conversation" in text  # help table
    assert text.count("No report yet") == 2  # before the search and after /reset
    assert "Shortlist report" in text and "Started a new conversation." in text
    assert text.rstrip().endswith("Goodbye!")


def test_repl_stops_when_the_conversation_ends(chat, monkeypatch):
    inputs = iter(["Find me a Python backend engineer", "goodbye"])
    monkeypatch.setattr(builtins, "input", lambda *_: next(inputs))
    assert cli.run(chat.graph, "gpt-test", console=chat.console) == 0
    assert output(chat).rstrip().endswith("Goodbye!")


def test_repl_exits_on_ctrl_d(chat, monkeypatch):
    def eof(*_):
        raise EOFError

    monkeypatch.setattr(builtins, "input", eof)
    assert cli.run(chat.graph, "gpt-test", console=chat.console) == 0


# --- refinement --------------------------------------------------------------------------------------


DJANGO_MUST = ROLE.model_copy(update={
    "must_have": [*ROLE.must_have, ROLE.must_have[0].model_copy(update={"skill": "Django"})],
    "nice_to_have": [],
})


def refine_router(messages) -> IntentDecision:
    latest = messages[1].content.rsplit("Latest user message: ", 1)[1].lower()
    return IntentDecision(intent="refine" if "must-have" in latest else route_by_keyword(messages).intent)


def test_refinement_reranks_and_explains_the_changes(chat, services):
    services.model.responses[IntentDecision] = refine_router
    services.model.responses[ChangeNarrative] = ChangeNarrative(narrative="Django is now required.")
    chat.send("Find me a Python backend engineer")
    before = state(chat)["shortlist"]
    services.model.responses[Requirements] = DJANGO_MUST
    chat.send("Make Django a must-have")

    values = state(chat)
    assert values["intent"] == "refine"
    assert values["requirements"] == DJANGO_MUST
    assert values["requirements_history"] == [ROLE, DJANGO_MUST]
    assert values["previous_shortlist"] == before
    assert all("Django" in m.matched_skills for m in values["shortlist"])

    # The refine prompt carried the current requirements and the instruction.
    refine_call = [c for c in services.model.calls if c[0] is Requirements][-1]
    assert '"role_title":"Python Backend Engineer"' in refine_call[1][1].content
    assert refine_call[1][1].content.endswith("Instruction: Make Django a must-have")

    diff = values["ranking_diff"]
    assert diff.requirement_changes == ["Django is now a must-have"]
    dropped = diff.of_kind("dropped")
    assert dropped and all(any("now fails" in r for r in c.reasons) for c in dropped)

    assert "## What changed" in values["report"] and "Django is now a must-have" in values["report"]
    summary = values["messages"][-1].text
    assert summary.startswith("Django is now required.") and "Screened 12 resumes" in summary
    assert "Updated the requirements" in output(chat) and "Ranking changes:" in output(chat)


def test_new_search_after_a_refinement_has_no_changes_section(chat, services):
    services.model.responses[IntentDecision] = refine_router
    services.model.responses[ChangeNarrative] = ChangeNarrative(narrative="Django is now required.")
    chat.send("Find me a Python backend engineer")
    services.model.responses[Requirements] = DJANGO_MUST
    chat.send("Make Django a must-have")
    chat.send("Find me a Python backend engineer")  # extraction now answers DJANGO_MUST again
    assert "## What changed" not in state(chat)["report"]


def test_refine_without_a_role_starts_a_new_search(services):
    services.model = FakeChatModel({IntentDecision: IntentDecision(intent="refine")})
    update = IntentRouter(services)({"messages": [HumanMessage("Make React a must-have")]})
    assert update == {"intent": "new_search"}
