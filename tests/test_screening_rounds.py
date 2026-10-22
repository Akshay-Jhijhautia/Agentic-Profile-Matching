import dataclasses
import json

import pytest

import matching_agent
from profile_matching import changes, screening
from profile_matching.schemas import (
    DEEP_WEIGHT,
    Assessment,
    HiringDecision,
    HiringDecisions,
    Recommendation,
    Requirements,
)
from tests.fakes import SCREENING_RESPONSES, FakeChatModel, fake_assessment

ROLE = Requirements.model_validate({
    "role_title": "Python Backend Engineer", "must_have": [{"skill": "Python", "min_years": None}],
    "nice_to_have": ["Django"], "min_total_years": None, "min_degree": None,
    "ideal_profile": "Python backend engineer.",
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


def test_three_rounds_assess_each_shortlisted_candidate_then_decide(run, services):
    state = run()
    shortlist = state["shortlist"]
    assert len(shortlist) == 4
    assert len(_calls(services, Assessment)) == 4  # one deep analysis per candidate (in parallel)
    assert len(_calls(services, HiringDecisions)) == 1  # one committee call for the cohort
    for match in shortlist:
        assert match.assessment.deep_score == 80
        assert match.final_score == round(DEEP_WEIGHT * 80 + (1 - DEEP_WEIGHT) * match.match_score)
        assert match.recommendation.decision == ("hire" if match.final_score >= 75 else "no_hire")
    assert [m.rank for m in shortlist] == [1, 2, 3, 4]
    assert len(state["assessments"]) == 4

    # The deep review saw the full resume and the round-1 checks.
    payload = json.loads(_calls(services, Assessment)[0][1].content)
    assert payload["requirements"]["must_have"][0]["skill"] == "Python"
    assert payload["first_round"]["must_have_checks"] and payload["resume_text"]


def test_deep_scores_can_reorder_the_shortlist(run):
    def by_name(messages):
        # The last candidate of round 1 gets a perfect deep review, everyone else a weak one.
        resume = json.loads(messages[1].content)["resume_text"]
        return fake_assessment(100 if resume.startswith(last_name) else 40)

    round_one = run(rounds=1)["shortlist"]
    last_name = round_one[-1].name
    final = run(rounds=2, responses={Assessment: by_name})["shortlist"]
    assert final[0].name == last_name
    assert all(m.recommendation is None for m in final)  # round 3 did not run


def test_round_one_only_makes_no_deep_calls(run, services):
    state = run(rounds=1)
    assert _calls(services, Assessment) == [] and _calls(services, HiringDecisions) == []
    assert all(m.assessment is None and m.final_score == m.match_score for m in state["shortlist"])


def test_report_and_summary_show_every_round(run):
    state = run()
    report = state["report"]
    assert "| # | Candidate | Score | Retrieval | Deep | Must-haves | Recommendation |" in report
    first = state["shortlist"][0]
    assert f"**Deep analysis (80):** Solid backend engineer." in report
    assert f"**Retrieval ({first.match_score}):**" in report
    summary = state["messages"][-1].text
    assert "Recommended to advance:" in summary or "No candidate is recommended to advance." in summary


# --- caching ------------------------------------------------------------------------------------------


def test_requirements_key_ignores_wording_but_not_requirements():
    reworded = ROLE.model_copy(update={"role_title": "Backend Dev", "ideal_profile": "Someone else."})
    stricter = ROLE.model_copy(update={"min_total_years": 5})
    assert screening.requirements_key(reworded) == screening.requirements_key(ROLE)
    assert screening.requirements_key(stricter) != screening.requirements_key(ROLE)


def test_cached_assessments_are_reused(services):
    services.model = FakeChatModel({Requirements: ROLE, **SCREENING_RESPONSES})
    services.settings = dataclasses.replace(services.settings, shortlist_size=3)
    graph = matching_agent.build_graph(services)
    state = graph.invoke({"messages": [("user", "Python backend engineer")]})
    assert len(_calls(services, Assessment)) == 3

    # Same hard requirements, cache already filled: round 2 is skipped, round 3 still runs.
    again = graph.invoke({"messages": [("user", "Python backend engineer")], "assessments": state["assessments"]})
    assert len(_calls(services, Assessment)) == 3
    assert len(_calls(services, HiringDecisions)) == 2
    assert all(m.assessment is not None for m in again["shortlist"])


# --- round 3 ---------------------------------------------------------------------------------------------


def test_committee_answers_are_mapped_by_id_or_name_with_a_fallback(services):
    profiles = services.candidates.resolve_many(["John Doe", "Daniel Kim", "Wei Chen"])
    shortlist = screening.apply_assessments(
        screening.evaluate(services, ROLE, profiles), ROLE,
        {screening.assessment_key(ROLE, p.id): fake_assessment(90) for p in profiles},
    )
    services.model = FakeChatModel({HiringDecisions: HiringDecisions(decisions=[
        HiringDecision(candidate_id="john_doe", decision="hire", confidence="high", rationale="Strong."),
        HiringDecision(candidate_id="Daniel Kim", decision="no_hire", confidence="medium", rationale="Junior."),
    ])})
    decided = {m.id: m.recommendation for m in screening.decide(services, ROLE, shortlist)}
    assert decided["john_doe"] == Recommendation(decision="hire", confidence="high", rationale="Strong.")
    assert decided["daniel_kim"].decision == "no_hire"  # answered by name
    fallback = decided["wei_chen"]  # left out by the committee
    assert fallback.confidence == "low" and fallback.decision == "hire"  # final score >= 75
    assert screening.decide(services, ROLE, []) == []


def test_borderline_band(services):
    (match,) = screening.evaluate(services, ROLE, services.candidates.resolve_many(["John Doe"]))
    assert not match.borderline  # no deep analysis yet
    for deep, expected in [(40, False), (70, True), (95, False)]:
        assessed = match.model_copy(update={"assessment": fake_assessment(deep), "match_score": 70})
        low, high = 60, 75
        assert assessed.borderline is (low <= assessed.final_score < high) is expected


# --- explaining moves caused by round 2 ---------------------------------------------------------------


def test_diff_names_the_deep_analysis_when_it_moved_a_candidate(services):
    (base,) = screening.evaluate(services, ROLE, services.candidates.resolve_many(["John Doe"]))
    old = base.model_copy(update={"assessment": fake_assessment(60)})
    new = base.model_copy(update={"assessment": fake_assessment(90)})
    assert changes.reasons_for(old, new) == [
        f"score {old.final_score} → {new.final_score} (deep analysis 60 → 90)"
    ]
