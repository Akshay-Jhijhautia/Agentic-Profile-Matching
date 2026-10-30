import json
from pathlib import Path

import pytest

from resume_shortlister import analysis, screening
from resume_shortlister.requirements import load_job_description
from resume_shortlister.schemas import (
    CandidateNote,
    ComparisonVerdict,
    InterviewQuestion,
    InterviewQuestions,
    Requirements,
)
from resume_shortlister.tools import make_tools
from resume_shortlister.tools._base import MAX_TOOL_TEXT_CHARS, safe_tool
from tests.fakes import FakeChatModel, fake_assessment
from tests.paths import JD_DIR, RESUMES_DIR


def _role(**overrides) -> Requirements:
    fields = {"role_title": "Backend Engineer", "must_have": [], "nice_to_have": [],
              "min_total_years": None, "min_degree": None, "ideal_profile": "Python backend engineer."}
    return Requirements.model_validate(fields | overrides)


DJANGO_ROLE = _role(must_have=[{"skill": "Django", "min_years": None}])


@pytest.fixture
def tools(services):
    return {tool.name: tool for tool in make_tools(services)}


def _payload(fake: FakeChatModel) -> dict:
    """The JSON the analysis sent to the model in its last call."""
    _, messages = fake.calls[-1]
    return json.loads(messages[1].content)


# --- registry and safe_tool -------------------------------------------------------------------


def test_registry_exposes_every_tool(tools):
    assert set(tools) == {
        "read_file", "list_files", "write_file", "search_in_file", "rag_search",
        "extract_requirements", "compare_candidates", "generate_interview_questions", "get_candidate_report",
    }


def test_injected_state_is_hidden_from_the_model(tools):
    for name in ("compare_candidates", "generate_interview_questions", "get_candidate_report"):
        assert "state" not in tools[name].tool_call_schema.model_json_schema()["properties"]


def test_safe_tool_reports_expected_errors():
    @safe_tool
    def fails():
        raise LookupError("no such candidate")

    assert fails() == {"success": False, "error": "LookupError: no such candidate"}


def test_safe_tool_lets_unexpected_errors_raise():
    @safe_tool
    def broken():
        raise RuntimeError("bug")

    with pytest.raises(RuntimeError):
        broken()


def test_safe_tool_truncates_long_text():
    @safe_tool
    def long():
        return {"content": "x" * (MAX_TOOL_TEXT_CHARS + 10), "name": "a"}

    result = long()
    assert len(result["content"]) == MAX_TOOL_TEXT_CHARS and result["name"] == "a"
    assert "truncated" in result


# --- file tools ---------------------------------------------------------------------------------


def test_read_list_and_search(tools):
    path = str(RESUMES_DIR / "resume_alex_lee.txt")
    assert "Alex Lee" in tools["read_file"].invoke({"filepath": path})["content"]
    listed = tools["list_files"].invoke({"directory": str(RESUMES_DIR), "extension": "pdf"})
    assert listed["files"] and all(f["extension"] == ".pdf" for f in listed["files"])
    assert tools["search_in_file"].invoke({"filepath": path, "keyword": "Kafka"})["match_count"] >= 1


def test_list_files_error_is_reported(tools):
    result = tools["list_files"].invoke({"directory": "/no/such/dir"})
    assert result["success"] is False and "FileNotFoundError" in result["error"]


def test_write_file_stays_in_reports_dir(tools, services):
    reports = Path(services.settings.reports_dir).resolve()
    assert tools["write_file"].invoke({"filepath": "a/report.md", "content": "# Hi"})["success"]
    assert (reports / "a" / "report.md").read_text() == "# Hi"
    inside = str(reports / "b.md")
    assert tools["write_file"].invoke({"filepath": inside, "content": "ok"})["success"]
    for outside in ("../escape.md", "/tmp/escape.md"):
        result = tools["write_file"].invoke({"filepath": outside, "content": "x"})
        assert result["success"] is False and "PermissionError" in result["error"]


# --- search and requirements --------------------------------------------------------------------


def test_rag_search_returns_candidate_ids(tools):
    result = tools["rag_search"].invoke({"query": "Kubernetes and Terraform infrastructure", "k": 3})
    assert len(result["candidates"]) == 3
    assert result["candidates"][0]["id"] == "fatima_ahmed"


@pytest.mark.parametrize("k", [0, 21])
def test_rag_search_validates_k(tools, k):
    assert tools["rag_search"].invoke({"query": "Python", "k": k})["success"] is False


def test_extract_requirements_reads_a_jd_file(tools, services):
    services.model = FakeChatModel({Requirements: _role(must_have=[{"skill": "Java", "min_years": None}])})
    path = JD_DIR / "java_backend.txt"
    result = tools["extract_requirements"].invoke({"jd": str(path)})
    # The model's reading, completed by the rules: "Experience with Kafka ..." names Kafka alone.
    assert [r["skill"] for r in result["must_have"]] == ["Java", "Kafka"] and result["min_total_years"] == 6
    _, messages = services.model.calls[0]
    assert messages[1].content == path.read_text().strip()


def test_load_job_description():
    assert load_job_description("React and 3+ years") == "React and 3+ years"
    with pytest.raises(FileNotFoundError):
        load_job_description("missing_jd.txt")


# --- candidate facts ---------------------------------------------------------------------------


def test_candidate_facts_rank_passing_candidates_first(services):
    profiles = services.candidates.resolve_many(["Maria Garcia", "John Doe"])
    facts = analysis.candidate_facts(services, profiles, DJANGO_ROLE)
    assert [f["id"] for f in facts] == ["john_doe", "maria_garcia"]
    assert [f["rank"] for f in facts] == [1, 2]
    assert facts[0]["passes_must_haves"] and not facts[1]["passes_must_haves"]
    assert set(facts[0]["score_breakdown"]) == {"semantic", "keyword", "skill_coverage"}


def test_candidate_facts_without_role_are_profiles_only(services):
    profiles = services.candidates.resolve_many(["Maria Garcia", "John Doe"])
    facts = analysis.candidate_facts(services, profiles, None)
    assert [f["id"] for f in facts] == ["maria_garcia", "john_doe"]
    assert "match_score" not in facts[0]


# --- compare_candidates ----------------------------------------------------------------------------


def _verdict(ranking, note_ids):
    return ComparisonVerdict(
        ranking=ranking,
        notes=[CandidateNote(candidate_id=i, strengths=["s"], weaknesses=["w"]) for i in note_ids],
        summary="John fits best.",
    )


def test_compare_candidates_uses_role_from_state(tools, services):
    # The model knows the candidates by label; its answer is mapped back to ids and names.
    verdict = _verdict(["Candidate 1"], ["Candidate 1", "candidate_2"])
    services.model = FakeChatModel({ComparisonVerdict: verdict.model_copy(update={"summary": "Candidate 1 fits best."})})
    result = tools["compare_candidates"].invoke(
        {"candidate_ids": ["maria", "john_doe"], "state": {"requirements": DJANGO_ROLE.model_dump()}}
    )
    assert result["role"] == "Backend Engineer"
    assert [c["id"] for c in result["candidates"]] == ["john_doe", "maria_garcia"]
    assert result["verdict"]["ranking"] == ["john_doe", "maria_garcia"]
    assert [n["candidate_id"] for n in result["verdict"]["notes"]] == ["john_doe", "maria_garcia"]
    assert result["verdict"]["summary"] == "John Doe fits best."

    payload = _payload(services.model)
    assert payload["requirements"]["must_have"][0]["skill"] == "Django"
    assert [c["candidate_id"] for c in payload["candidates"]] == ["Candidate 1", "Candidate 2"]
    assert not any({"id", "name", "resume"} & set(c) for c in payload["candidates"])
    assert all(c["resume_text"] and "Maria" not in c["resume_text"] for c in payload["candidates"])


def test_compare_candidates_without_role(tools, services):
    services.model = FakeChatModel({ComparisonVerdict: _verdict(["alex_lee", "john_doe"], [])})
    result = tools["compare_candidates"].invoke({"candidate_ids": ["John Doe", "Alex Lee"], "state": {}})
    assert result["role"] is None and "match_score" not in result["candidates"][0]
    assert _payload(services.model)["requirements"] is None


@pytest.mark.parametrize("ids", [["maria"], ["maria", "maria_garcia"], ["a", "b", "c", "d", "e", "f"]])
def test_compare_candidates_needs_two_to_five(tools, ids):
    result = tools["compare_candidates"].invoke({"candidate_ids": ids, "state": {}})
    assert result["success"] is False


def test_compare_candidates_unknown_name(tools):
    result = tools["compare_candidates"].invoke({"candidate_ids": ["maria", "Nobody Here"], "state": {}})
    assert result["success"] is False and "No candidate matches" in result["error"]


# --- generate_interview_questions --------------------------------------------------------------


def _questions(n):
    return InterviewQuestions(questions=[
        InterviewQuestion(question=f"Q{i}", focus="Django", purpose="assess_depth", strong_answer="A")
        for i in range(n)
    ])


def test_interview_questions_target_gaps_and_unverified_claims(tools, services):
    services.model = FakeChatModel({InterviewQuestions: _questions(8)})
    role = _role(
        must_have=[{"skill": "Django", "min_years": None}, {"skill": "Redis", "min_years": 2},
                   {"skill": "Kubernetes", "min_years": None}],
        nice_to_have=["GraphQL", "Docker"],
    )
    result = tools["generate_interview_questions"].invoke(
        {"candidate_id": "John Doe", "count": 4, "state": {"requirements": role}}
    )
    assert result["candidate"] == {"id": "john_doe", "name": "John Doe"}
    focus = result["focus_areas"]
    assert any("Kubernetes" in gap for gap in focus["gaps"])
    assert "nice to have: GraphQL" in focus["gaps"] and "nice to have: Docker" not in focus["gaps"]
    assert any("Redis" in claim for claim in focus["unverified"])
    assert len(result["questions"]) == 4  # the model wrote 8; trimmed to the requested count

    payload = _payload(services.model)
    assert payload["question_count"] == 4 and payload["focus_areas"] == focus
    assert payload["candidate"]["candidate_id"] == "Candidate 1" and "name" not in payload["candidate"]
    assert "John" not in payload["resume_text"] and "example.com" not in payload["resume_text"]


@pytest.mark.parametrize("count", [2, 11])
def test_interview_questions_validate_count(tools, count):
    result = tools["generate_interview_questions"].invoke(
        {"candidate_id": "John Doe", "count": count, "state": {}}
    )
    assert result["success"] is False


def test_compare_uses_the_shortlisted_record_and_its_final_score(tools, services):
    profiles = services.candidates.resolve_many(["John Doe", "Maria Garcia"])
    john = next(m for m in screening.evaluate(services, DJANGO_ROLE, profiles) if m.id == "john_doe")
    # John's deep analysis is poor, so his final score falls well below his retrieval score.
    shortlisted = john.model_copy(update={"assessment": fake_assessment(10)})
    services.model = FakeChatModel({ComparisonVerdict: _verdict(["john_doe"], [])})
    result = tools["compare_candidates"].invoke({
        "candidate_ids": ["John Doe", "Maria Garcia"],
        "state": {"requirements": DJANGO_ROLE, "shortlist": [shortlisted]},
    })
    facts = {c["id"]: c for c in result["candidates"]}
    assert facts["john_doe"]["deep_score"] == 10
    assert facts["john_doe"]["final_score"] == shortlisted.final_score < john.match_score
    assert facts["maria_garcia"]["deep_score"] is None  # not shortlisted: screened in round 1
