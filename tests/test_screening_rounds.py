import dataclasses
import json

import pytest

import matching_agent
from profile_matching import changes, guardrails, screening
from profile_matching.schemas import (
    DEEP_WEIGHT,
    DecisionRationale,
    DeepReview,
    HiringRationales,
    Requirements,
)
from tests.fakes import SCREENING_RESPONSES, FakeChatModel, fake_assessment, fake_review

ROLE = Requirements.model_validate({
    "role_title": "Python Backend Engineer", "must_have": [{"skill": "Python", "min_years": None}],
    "nice_to_have": ["Django"], "min_total_years": None, "min_degree": None,
})


def _calls(services, schema) -> list:
    return [messages for called, messages in services.model.calls if called is schema]


@pytest.fixture
def run(services):
    def _run(rounds: int = 3, size: int = 4, responses: dict | None = None):
        services.model = FakeChatModel({Requirements: ROLE, **SCREENING_RESPONSES, **(responses or {})})
        services.settings = dataclasses.replace(services.settings, screening_rounds=rounds, shortlist_size=size)
        return matching_agent.screen(services, "Python backend engineer")

    return _run


# --- the rounds in the graph -------------------------------------------------------------------------


def test_three_rounds_review_each_candidate_then_decide_by_rule(run, services):
    state = run()
    shortlist = state["shortlist"]
    assert len(shortlist) == 4
    assert len(_calls(services, DeepReview)) == 4  # one deep review per candidate (in parallel)
    assert len(_calls(services, HiringRationales)) == 1  # one call explains every decision
    for match in shortlist:
        # Every verdict is strong (grounded quotes), no risks: the rubric gives 100.
        assert match.assessment.deep_score == 100
        assert match.final_score == round(DEEP_WEIGHT * 100 + (1 - DEEP_WEIGHT) * match.match_score)
        rule = guardrails.decide(match)
        assert (match.recommendation.decision, match.recommendation.rule) == (rule.decision, rule.rule)
        assert match.recommendation.rationale == f"Final score {match.final_score}."
    assert [m.rank for m in shortlist] == [1, 2, 3, 4]

    # The review got numbered requirements, the round-1 facts and the resume, but no scores.
    payload = json.loads(_calls(services, DeepReview)[0][1].content)
    assert payload["requirements"] == [
        {"id": "M1", "requirement": "Python", "kind": "must-have"},
        {"id": "N1", "requirement": "Django", "kind": "nice-have"},
    ]
    assert payload["first_round"]["must_have_checks"] and "match_score" not in payload["first_round"]
    assert payload["resume_text"]


def test_deep_reviews_can_reorder_the_shortlist(run):
    round_one = run(rounds=1)["shortlist"]
    last_name = round_one[-1].name

    def by_name(messages):
        resume = json.loads(messages[1].content)["resume_text"]
        return fake_review("strong" if resume.startswith(last_name) else "missing")(messages)

    final = run(rounds=2, responses={DeepReview: by_name})["shortlist"]
    assert final[0].name == last_name
    assert all(m.recommendation is None for m in final)  # round 3 did not run


def test_round_one_only_makes_no_deep_calls(run, services):
    state = run(rounds=1)
    assert _calls(services, DeepReview) == [] and _calls(services, HiringRationales) == []
    assert all(m.assessment is None and m.final_score == m.match_score for m in state["shortlist"])


def test_report_and_summary_show_every_round(run):
    state = run()
    report = state["report"]
    assert "| # | Candidate | Score | Retrieval | Deep | Must-haves | Recommendation |" in report
    first = state["shortlist"][0]
    assert "**Deep analysis (100):** Solid backend engineer." in report
    assert f"**Retrieval ({first.match_score}):**" in report
    assert "*(Rule: final score" in report
    summary = state["messages"][-1].text
    assert "Recommended to advance:" in summary or "No candidate is recommended to advance." in summary


# --- caching ------------------------------------------------------------------------------------------


def test_requirements_key_ignores_wording_but_not_requirements():
    reworded = ROLE.model_copy(update={"role_title": "Backend Dev"})
    stricter = ROLE.model_copy(update={"min_total_years": 5})
    assert screening.requirements_key(reworded) == screening.requirements_key(ROLE)
    assert screening.requirements_key(stricter) != screening.requirements_key(ROLE)


def test_cached_assessments_are_reused(services):
    services.model = FakeChatModel({Requirements: ROLE, **SCREENING_RESPONSES})
    services.settings = dataclasses.replace(services.settings, shortlist_size=3)
    graph = matching_agent.build_graph(services)
    state = graph.invoke({"messages": [("user", "Python backend engineer")]})
    assert len(_calls(services, DeepReview)) == 3

    # Same hard requirements, cache already filled: round 2 is skipped, round 3 still runs.
    again = graph.invoke({"messages": [("user", "Python backend engineer")], "assessments": state["assessments"]})
    assert len(_calls(services, DeepReview)) == 3
    assert len(_calls(services, HiringRationales)) == 2
    assert all(m.assessment is not None for m in again["shortlist"])


# --- round 3 ---------------------------------------------------------------------------------------------


def _assessed(services, names: list[str], deep_score: int = 90):
    profiles = services.candidates.resolve_many(names)
    return screening.apply_assessments(
        screening.evaluate(services, ROLE, profiles), ROLE,
        {screening.assessment_key(ROLE, p.id): fake_assessment(deep_score) for p in profiles},
    )


def test_rationales_are_mapped_by_id_or_name_and_never_change_the_decision(services):
    shortlist = _assessed(services, ["John Doe", "Daniel Kim", "Wei Chen"])
    services.model = FakeChatModel({HiringRationales: HiringRationales(rationales=[
        DecisionRationale(candidate_id="john_doe", rationale="Strong Python."),
        DecisionRationale(candidate_id="Daniel Kim", rationale="Solid."),
    ])})
    decided = {m.id: m.recommendation for m in screening.decide(services, ROLE, shortlist)}
    assert decided["john_doe"].rationale == "Strong Python."
    assert decided["daniel_kim"].rationale == "Solid."  # answered by name
    # Left out by the model: the rule's outcome stands as the rationale.
    assert decided["wei_chen"].rationale == decided["wei_chen"].rule
    for match in shortlist:
        assert decided[match.id].decision == guardrails.decide(match).decision
    assert screening.decide(services, ROLE, []) == []


def test_borderline_band(services):
    (match,) = screening.evaluate(services, ROLE, services.candidates.resolve_many(["John Doe"]))
    assert not match.borderline  # no deep analysis yet
    for deep, expected in [(40, False), (70, True), (95, False)]:
        assessed = match.model_copy(update={"assessment": fake_assessment(deep), "match_score": 70})
        assert assessed.borderline is (60 <= assessed.final_score < 75) is expected


# --- explaining moves caused by round 2 ---------------------------------------------------------------


def test_diff_names_the_deep_analysis_when_it_moved_a_candidate(services):
    (base,) = screening.evaluate(services, ROLE, services.candidates.resolve_many(["John Doe"]))
    old = base.model_copy(update={"assessment": fake_assessment(60)})
    new = base.model_copy(update={"assessment": fake_assessment(90)})
    assert changes.reasons_for(old, new) == [
        f"score {old.final_score} → {new.final_score} (deep analysis 60 → 90)"
    ]
