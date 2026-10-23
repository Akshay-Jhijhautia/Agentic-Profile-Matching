import dataclasses
import warnings
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import matching_agent
from profile_matching import reports, screening
from profile_matching.agent.state import make_checkpointer
from profile_matching.pools import ResumePool
from profile_matching.rag import resume_rag
from profile_matching.schemas import CandidateMatch, Requirements
from tests.fakes import SCREENING_RESPONSES, FakeChatModel
from tests.paths import JD_DIR


def _role(**overrides) -> Requirements:
    fields = {"role_title": "Python Backend Engineer", "must_have": [], "nice_to_have": [],
              "min_total_years": None, "min_degree": None, "ideal_profile": "Python backend engineer."}
    return Requirements.model_validate(fields | overrides)


PYTHON_ROLE = _role(must_have=[{"skill": "Python", "min_years": None}], nice_to_have=["Django"])
RUST_ROLE = _role(role_title="Rust Engineer", must_have=[{"skill": "Rust", "min_years": None}])


@pytest.fixture
def run(services):
    """Run the pipeline once with a fake LLM answering ``requirements``.

    Round 1 only unless ``screening_rounds`` is given (the deep rounds have their own tests).
    """

    def _run(requirements: Requirements, jd: str = "Python backend engineer; Django is a plus", **settings):
        services.model = FakeChatModel({Requirements: requirements, **SCREENING_RESPONSES})
        services.settings = dataclasses.replace(services.settings, **{"screening_rounds": 1} | settings)
        return matching_agent.screen(services, jd)

    return _run


# --- graph -------------------------------------------------------------------------------------------


def test_graph_is_the_pipeline(services):
    graph = matching_agent.build_graph(services).get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    assert edges == {
        ("__start__", "parse_jd"), ("parse_jd", "extract_requirements"),
        # screening needs resumes; without them the requirements are acknowledged instead
        ("extract_requirements", "search_resumes"), ("extract_requirements", "await_resumes"),
        ("await_resumes", "__end__"), ("search_resumes", "rank_candidates"),
        # round 2 fans out per candidate, or is skipped when nothing needs assessing
        ("rank_candidates", "assess_candidate"), ("rank_candidates", "finalize_ranking"),
        ("assess_candidate", "finalize_ranking"), ("finalize_ranking", "suggest_improvements"),
        # explain_changes sits between ranking and the report, but only runs after a refinement
        ("suggest_improvements", "explain_changes"), ("explain_changes", "generate_report"),
        ("suggest_improvements", "generate_report"),
        ("generate_report", "__end__"),
    }


def test_pipeline_screens_every_resume_and_shortlists_passing_ones(run):
    state = run(PYTHON_ROLE, shortlist_size=3)
    assert state["requirements"] == PYTHON_ROLE
    assert state["requirements_history"] == [PYTHON_ROLE]

    candidates = state["candidates"]
    assert len(candidates) == 12  # the whole index, not a retrieval pool
    assert [m.rank for m in candidates] == list(range(1, 13))
    passing = [m for m in candidates if m.passes_must_haves]
    assert passing and candidates[: len(passing)] == passing  # knock-outs ranked last

    shortlist = state["shortlist"]
    assert shortlist == passing[:3]
    assert all("Python" in m.matched_skills for m in shortlist)
    assert state["previous_shortlist"] == []


def test_pipeline_reports_and_summarises(run):
    state = run(PYTHON_ROLE, shortlist_size=3)
    report = state["report"]
    assert report.startswith("# Shortlist: Python Backend Engineer")
    assert "- **Must have:** Python\n- **Nice to have:** Django" in report
    for match in state["shortlist"]:
        assert f"{match.name} (`{match.id}`)" in report and match.reasoning in report

    human, summary = state["messages"]
    assert isinstance(human, HumanMessage) and human.text == "Python backend engineer; Django is a plus"
    assert isinstance(summary, AIMessage) and summary.text.startswith("Screened 12 resumes")


def test_pipeline_reads_a_jd_file(run, services):
    path = JD_DIR / "java_backend.txt"
    state = run(PYTHON_ROLE, jd=str(path))
    assert state["job_description"] == path.read_text().strip()
    _, messages = services.model.calls[0]
    assert messages[1].content == path.read_text().strip()


def test_no_candidate_passes(run):
    state = run(RUST_ROLE, jd="Rust engineer")
    assert state["shortlist"] == []
    assert "**No candidate meets every must-have.**" in state["report"]
    assert "## Closest knocked-out candidates" in state["report"]
    assert "Consider relaxing" in state["messages"][-1].text


def test_empty_job_description_is_rejected(run):
    with pytest.raises(ValueError, match="No job description"):
        run(PYTHON_ROLE, jd="   ")


def test_state_round_trips_through_the_checkpointer_without_warnings(services):
    services.model = FakeChatModel({Requirements: PYTHON_ROLE, **SCREENING_RESPONSES})
    graph = matching_agent.build_graph(services, checkpointer=make_checkpointer())
    config = {"configurable": {"thread_id": "t1"}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        graph.invoke({"messages": [HumanMessage("Python backend engineer")]}, config)
        values = graph.get_state(config).values
    assert isinstance(values["requirements"], Requirements)
    assert all(isinstance(m, CandidateMatch) for m in values["shortlist"])


# --- screening and reports ---------------------------------------------------------------------------


def test_rank_matches_puts_knockouts_last(services):
    matches = screening.evaluate(services, PYTHON_ROLE)
    shuffled = list(reversed(matches))
    reranked = screening.rank_matches(shuffled)
    assert [m.id for m in reranked] == [m.id for m in matches]
    scores = [m.match_score for m in reranked if m.passes_must_haves]
    assert scores == sorted(scores, reverse=True)


def test_screening_can_be_limited_to_profiles(services):
    profiles = services.candidates.resolve_many(["John Doe", "Alex Lee"])
    assert {m.id for m in screening.evaluate(services, PYTHON_ROLE, profiles)} == {"john_doe", "alex_lee"}


def test_summary_with_nothing_indexed():
    assert reports.shortlist_summary(PYTHON_ROLE, [], []).endswith("No resumes are indexed yet.")


# --- readiness and CLI ---------------------------------------------------------------------------------


def test_check_ready_needs_an_index(services, tmp_path):
    empty = resume_rag.ResumeRAG(str(tmp_path / "empty"), embedder=resume_rag.HashingEmbedder())
    services.pool = ResumePool(tmp_path, empty)
    with pytest.raises(ValueError, match="No readable resumes"):
        services.check_ready()


def test_check_ready_needs_a_key_for_the_real_model(services, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    services.model = FakeChatModel({})
    services.check_ready()  # hashing index + fake model: nothing to authenticate
    services.model = None
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        services.check_ready()


def test_cli_prints_the_report_and_saves_it(services, monkeypatch, capsys):
    services.model = FakeChatModel({Requirements: PYTHON_ROLE, **SCREENING_RESPONSES})
    services.settings = dataclasses.replace(services.settings, screening_rounds=1, shortlist_size=2)
    monkeypatch.setattr(matching_agent.Services, "from_settings", lambda: services)
    assert matching_agent.main(["Python backend engineer"]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("# Shortlist: Python Backend Engineer")
    (folder,) = Path(services.settings.reports_dir).iterdir()
    assert f"Reports saved to {resume_rag.display_path(str(folder))}" in captured.err
    assert (folder / "summary.md").read_text() == captured.out.removesuffix("\n")
    assert len(list((folder / "candidates").iterdir())) == 2


def test_cli_reports_errors(services, monkeypatch, capsys):
    services.model = None
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(matching_agent.Services, "from_settings", lambda: services)
    assert matching_agent.main(["--jd-file", str(JD_DIR / "java_backend.txt")]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err
