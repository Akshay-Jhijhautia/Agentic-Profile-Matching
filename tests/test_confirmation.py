"""Confirming requirements the job description leaves open, before screening."""

import io

import pytest
from langchain_core.messages import AIMessage
from rich.console import Console

import matching_agent
from profile_matching import cli
from profile_matching.schemas import IntentDecision, Requirements
from tests.fakes import SCREENING_RESPONSES, FakeChatModel
from tests.test_conversation import ROLE, intent, output, route_by_keyword, state

IOS_JD = """Mobile engineer
Requirements:
- 4+ years of professional iOS development in Swift
- SwiftUI"""
# The model kept Swift and left iOS out: the job description does not say which it means.
IOS_ROLE = Requirements.model_validate({
    "role_title": "Mobile Engineer", "must_have": [{"skill": "Swift", "min_years": 4}, {"skill": "SwiftUI", "min_years": None}],
    "nice_to_have": ["iOS"], "min_total_years": None, "min_degree": None,
})
IOS_BOTH = IOS_ROLE.model_copy(update={"must_have": [*IOS_ROLE.must_have, IOS_ROLE.must_have[0].model_copy(
    update={"skill": "iOS"})], "nice_to_have": []})
QUESTION = '"4+ years of professional iOS development in Swift": **Swift** must-have, **iOS** only nice-to-have'


@pytest.fixture
def chat(services):
    services.model = FakeChatModel({Requirements: IOS_ROLE, IntentDecision: route_by_keyword, **SCREENING_RESPONSES})
    return cli.ChatSession(matching_agent.build_graph(services, conversational=True), Console(file=io.StringIO(), width=200))


def _router_calls(services) -> int:
    return sum(schema is IntentDecision for schema, _ in services.model.calls)


def _ask_ios(chat):
    chat.send(IOS_JD)
    assert chat.graph.get_state(chat.config).next == ("human_feedback",)
    assert "candidates" not in state(chat) and state(chat)["open_questions"] == [QUESTION]


def test_an_open_requirement_is_asked_about_before_screening(chat, services):
    _ask_ios(chat)
    text = output(chat)
    assert "Before screening, please check what the job description leaves open" in text
    assert 'make iOS a must-have' in text  # the example names the skill in question

    routed = _router_calls(services)
    chat.send("Yes, looks good")
    assert _router_calls(services) == routed  # a plain yes needs no model call
    values = state(chat)
    assert values["intent"] == "confirm" and values["open_questions"] == []
    assert len(values["candidates"]) == 12 and "Screening with these requirements" in output(chat)
    assert "Please check" not in values["report"]  # the recruiter settled it


def test_a_change_is_applied_before_screening(chat, services):
    _ask_ios(chat)
    services.model.responses[IntentDecision] = lambda messages: intent(
        "refine" if "must-have" in messages[1].content.lower() else "ask")
    services.model.responses[Requirements] = IOS_BOTH
    chat.send("Make iOS a must-have too")
    values = state(chat)
    assert values["intent"] == "refine" and values["open_questions"] == []
    assert "iOS" in values["requirements"].must_have_skills and len(values["candidates"]) == 12
    # Nothing was screened before, so there are no ranking changes to explain.
    assert values["ranking_diff"] is None and "Ranking changes" not in output(chat)


def test_questions_can_be_asked_while_the_confirmation_waits(chat, services):
    _ask_ios(chat)
    services.model.replies = [AIMessage("It means iOS app development, written in Swift.")]
    chat.send("What does iOS development cover here?")
    assert state(chat)["open_questions"] == [QUESTION] and "candidates" not in state(chat)
    chat.send("ok, go ahead")
    assert len(state(chat)["candidates"]) == 12


def test_yes_without_a_pending_question_goes_to_the_router(chat, services):
    services.model.replies = [AIMessage("Happy to help.")]
    chat.send("yes")
    assert _router_calls(services) == 1 and state(chat)["intent"] == "ask"


def test_clear_requirements_screen_straight_away(chat, services):
    services.model.responses[Requirements] = ROLE
    chat.send("Find me a Python backend engineer; Django is a plus")
    assert state(chat)["open_questions"] == [] and len(state(chat)["candidates"]) == 12


def test_one_shot_screening_lists_the_open_points_in_the_report(services):
    services.model = FakeChatModel({Requirements: IOS_ROLE, IntentDecision: route_by_keyword, **SCREENING_RESPONSES})
    result = matching_agent.screen(services, IOS_JD)
    assert len(result["candidates"]) == 12  # nobody to ask: screened as proposed
    assert f"**Please check:** the job description leaves these open, so the screening assumed:\n- {QUESTION}" \
        in result["report"]
