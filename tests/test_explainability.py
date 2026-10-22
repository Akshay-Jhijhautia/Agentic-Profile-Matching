import dataclasses
import json
from pathlib import Path

import pytest

import matching_agent
from profile_matching import explainability, screening
from profile_matching.schemas import (
    DEEP_WEIGHT,
    ImprovementSuggestions,
    Improvement,
    Recommendation,
    Requirements,
)
from profile_matching.tools import make_tools
from tests.fakes import SCREENING_RESPONSES, FakeChatModel, fake_assessment

ROLE = Requirements.model_validate({
    "role_title": "Python Backend Engineer",
    "must_have": [{"skill": "Python", "min_years": None}, {"skill": "Redis", "min_years": 2}],
    "nice_to_have": ["Django", "GraphQL"], "min_total_years": None, "min_degree": None,
    "ideal_profile": "Python backend engineer.",
})


@pytest.fixture
def john(services):
    """John Doe screened against ROLE: passes Python, Redis years unverified, has Django."""
    (match,) = screening.evaluate(services, ROLE, services.candidates.resolve_many(["John Doe"]))
    return match


def full_record(match):
    """``match`` after every round, with a deep score that puts it in the borderline band."""
    deep = round((67 - (1 - DEEP_WEIGHT) * match.match_score) / DEEP_WEIGHT)  # final score ≈ 67
    assessment = fake_assessment(deep)
    evidence = [assessment.requirement_evidence[0].model_copy(update={"evidence": "Python | Django, 6 yrs"})]
    return match.model_copy(update={
        "assessment": assessment.model_copy(update={
            "risk_flags": ["Short tenure at Acme"], "requirement_evidence": evidence,
        }),
        "recommendation": Recommendation(decision="no_hire", confidence="medium", rationale="Thin Redis."),
        "improvements": [Improvement(gap="Redis years", suggestion="Show Redis work", how_to_verify="Ask")],
    })


# --- score breakdown --------------------------------------------------------------------------------


def test_points_add_up_to_the_final_score(john):
    for match in (john, full_record(john)):
        points = match.contributions()
        assert sum(points.values()) == pytest.approx(match.final_score, abs=1.0)
    assert set(full_record(john).contributions()) == {"semantic", "keyword", "skill_coverage", "deep"}


def test_without_must_have_skills_coverage_weight_is_shared(john):
    no_coverage = john.model_copy(update={
        "score_breakdown": john.score_breakdown.model_copy(update={"skill_coverage": None})
    })
    weights = no_coverage.retrieval_weights()
    assert set(weights) == {"semantic", "keyword"} and sum(weights.values()) == pytest.approx(1.0)


def test_score_rows_show_weights_and_the_final_score(john):
    rows = explainability.score_rows(full_record(john))
    assert [row[0] for row in rows] == [
        "Semantic fit", "Keyword match", "Must-have skill coverage", "Deep analysis", "**Final score**",
    ]
    assert rows[3][1:3] == [full_record(john).assessment.deep_score, f"{DEEP_WEIGHT:.2f}"]
    assert rows[-1][-1] == f"**{full_record(john).final_score}**"


# --- strengths and gaps -------------------------------------------------------------------------------


def test_round_one_strengths_and_gaps_come_from_the_checks(john):
    strengths, gaps = explainability.strengths_and_gaps(john, ROLE)
    assert any(s.startswith("Meets") and "Python" in s for s in strengths)
    assert "Has nice-to-have Django" in strengths
    assert any(g.startswith("Unverified:") and "Redis" in g for g in gaps)
    assert "No GraphQL (nice to have)" in gaps


def test_deep_analysis_strengths_and_gaps_take_precedence(john):
    strengths, gaps = explainability.strengths_and_gaps(full_record(john), ROLE)
    assert (strengths, gaps) == (["Shipped production APIs"], ["No GraphQL"])


# --- reports -------------------------------------------------------------------------------------------


def test_full_candidate_report(john):
    report = explainability.candidate_report(full_record(john), ROLE)
    assert report.startswith("# John Doe: Python Backend Engineer")
    for section in ("## How the score adds up", "## Must-have checks", "## Evidence per requirement",
                    "## Strengths", "## Gaps", "## Risk flags", "## Improvement suggestions",
                    "## Retrieval reasoning"):
        assert section in report
    assert "**Recommendation:** No hire (medium confidence). Thin Redis." in report
    assert "· borderline" in report
    assert "| Python | strong | Python \\| Django, 6 yrs |" in report  # pipes escaped in table cells
    assert "- Short tenure at Acme" in report


def test_round_one_candidate_report_has_no_deep_sections(john):
    report = explainability.candidate_report(john, ROLE)
    assert "## Evidence per requirement" not in report and "## Improvement suggestions" not in report
    assert "**Recommendation:**" not in report and "## Gaps" in report


def test_knocked_out_candidate_report(services):
    (alex,) = screening.evaluate(services, ROLE, services.candidates.resolve_many(["Alex Lee"]))
    report = explainability.candidate_report(alex, ROLE)
    assert "**Knocked out:** fails" in report and "✗ fails" not in report.split("\n")[0]


def test_summary_section_lists_improvements(john):
    section = "\n".join(explainability.summary_section(full_record(john), ROLE))
    assert "**To become a clear hire:**" in section
    assert "- Show Redis work (verify: Ask)" in section


def test_save_reports_writes_a_new_folder_each_time(john, tmp_path):
    shortlist = [full_record(john).model_copy(update={"rank": 1})]
    first = explainability.save_reports(str(tmp_path), ROLE, shortlist, "# Summary\n")
    second = explainability.save_reports(str(tmp_path), ROLE, shortlist, "# Summary\n")
    assert first != second and first.name.startswith("python-backend-engineer-")
    assert (first / "summary.md").read_text() == "# Summary\n"
    assert (first / "candidates" / "01-john_doe.md").read_text().startswith("# John Doe")


# --- improvement suggestions --------------------------------------------------------------------------


def test_only_borderline_candidates_get_suggestions(services, john):
    clear = john.model_copy(update={"assessment": fake_assessment(95), "match_score": 95})
    borderline = full_record(john).model_copy(update={"improvements": None})
    assert not clear.borderline and borderline.borderline
    services.model = FakeChatModel(SCREENING_RESPONSES)

    (unchanged,) = screening.suggest_improvements(services, ROLE, [clear])
    assert unchanged.improvements is None and services.model.calls == []

    (improved,) = screening.suggest_improvements(services, ROLE, [borderline])
    assert improved.improvements[0].suggestion == "Build a GraphQL API"
    _, messages = services.model.calls[0]
    assert [c["id"] for c in json.loads(messages[1].content)["candidates"]] == ["john_doe"]


# --- the candidate report tool ------------------------------------------------------------------------


@pytest.fixture
def report_tool(services):
    return {tool.name: tool for tool in make_tools(services)}["get_candidate_report"]


def test_report_tool_prefers_the_shortlisted_record(report_tool, services, john):
    shortlisted = full_record(john).model_copy(update={"rank": 1})
    state = {"requirements": ROLE, "shortlist": [shortlisted], "candidates": [john]}
    result = report_tool.invoke({"candidate_id": "John", "state": state})
    assert "## Evidence per requirement" in result["report"]


def test_report_tool_falls_back_to_round_one(report_tool, john):
    result = report_tool.invoke({"candidate_id": "john_doe", "state": {"requirements": ROLE, "candidates": [john]}})
    assert result["report"].startswith("# John Doe") and "## Evidence per requirement" not in result["report"]


@pytest.mark.parametrize(
    "state, error",
    [({}, "No search has run yet"), ({"requirements": ROLE, "candidates": []}, "was not screened")],
)
def test_report_tool_errors(report_tool, state, error):
    result = report_tool.invoke({"candidate_id": "John Doe", "state": state})
    assert result["success"] is False and error in result["error"]


# --- end to end ---------------------------------------------------------------------------------------------


def _screen(services, deep_score: int, **settings):
    services.model = FakeChatModel({Requirements: ROLE, **SCREENING_RESPONSES})
    services.model.responses[type(fake_assessment())] = fake_assessment(deep_score)
    services.settings = dataclasses.replace(services.settings, shortlist_size=3, **settings)
    return matching_agent.screen(services, "Python backend engineer")


def test_pipeline_suggests_improvements_and_saves_reports(services):
    state = _screen(services, deep_score=50)  # final scores land in the borderline band
    borderline = [m for m in state["shortlist"] if m.borderline]
    assert borderline and all(m.improvements for m in borderline)
    assert sum(c is ImprovementSuggestions for c, _ in services.model.calls) == 1
    assert "**To become a clear hire:**" in state["report"]

    folder = Path(services.settings.reports_dir) / Path(state["report_dir"]).name
    assert (folder / "summary.md").read_text() == state["report"]
    assert len(list((folder / "candidates").iterdir())) == len(state["shortlist"])
    assert f"Reports saved to `{state['report_dir']}`." in state["messages"][-1].text


def test_pipeline_without_saving(services):
    state = _screen(services, deep_score=90, save_reports=False)
    assert state["report_dir"] is None and not Path(services.settings.reports_dir).exists()
    assert "Reports saved" not in state["messages"][-1].text
