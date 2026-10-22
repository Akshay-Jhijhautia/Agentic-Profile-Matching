import typing

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from profile_matching import prompts
from profile_matching.rag import job_matcher, resume_rag
from profile_matching.requirements import (
    alternative_skills,
    apply_rule_floor,
    degree_required,
    extract_requirements,
)
from profile_matching.schemas import DegreeLevel, Requirements
from tests.fakes import FakeChatModel
from tests.paths import JD_DIR


def _requirements(**overrides) -> Requirements:
    fields = {
        "role_title": "Frontend Engineer",
        "must_have": [],
        "nice_to_have": [],
        "min_total_years": None,
        "min_degree": None,
        "ideal_profile": "Frontend engineer building accessible React apps.",
    }
    return Requirements.model_validate(fields | overrides)


def _fake(requirements: Requirements) -> FakeChatModel:
    return FakeChatModel({Requirements: requirements})


# --- Requirements model ------------------------------------------------------------------


def test_degree_levels_match_resume_rag():
    assert set(typing.get_args(DegreeLevel)) == set(resume_rag.DEGREE_LEVELS)


def test_skills_are_canonical_and_deduplicated():
    req = _requirements(
        must_have=[
            {"skill": "react.js", "min_years": 2},
            {"skill": "React", "min_years": 3},
            {"skill": "postgres", "min_years": None},
        ],
        nice_to_have=["tailwind", "PostgreSQL", "k8s", "Tailwind CSS"],
    )
    assert [(r.skill, r.min_years) for r in req.must_have] == [("React", 3), ("PostgreSQL", None)]
    assert req.nice_to_have == ["Tailwind CSS", "Kubernetes"]


def test_durations_and_degrees_are_not_skills():
    req = _requirements(
        must_have=[{"skill": "Java", "min_years": None}, {"skill": "6 years", "min_years": None}],
        nice_to_have=["PhD", "Master's degree", "MLflow", "3+ yrs"],
    )
    assert req.must_have_skills == ["Java"]
    assert req.nice_to_have == ["MLflow"]


def test_unknown_skills_are_kept_as_written():
    req = _requirements(must_have=[{"skill": " WCAG ", "min_years": None}])
    assert req.must_have_skills == ["WCAG"]


def test_to_must_haves_uses_matcher_format():
    req = _requirements(
        must_have=[{"skill": "React", "min_years": 3}, {"skill": "CSS", "min_years": None}],
        min_total_years=4,
        min_degree="Bachelor's",
    )
    checks = [{k: v for k, v in c.items() if k != "source"} for c in req.to_must_haves()]
    assert checks == [
        {"type": "skill_years", "skill": "React", "min_years": 3},
        {"type": "skill", "skill": "CSS"},
        {"type": "total_years", "min_years": 4},
        {"type": "degree", "degree": "Bachelor's"},
    ]


def test_must_haves_are_checked_by_the_matcher():
    req = _requirements(must_have=[{"skill": "React", "min_years": 3}, {"skill": "Go", "min_years": None}])
    metadata = {
        "skills": ["React", "CSS"], "skill_years": {"React": 4.0}, "experience_years": 4.0,
        "education": [], "highest_degree": None, "degree_level": 0,
    }
    statuses = [c["status"] for c in job_matcher.check_must_haves(metadata, req.to_must_haves())]
    assert statuses == ["met", "not_met"]


def test_query_text_reads_like_a_job_description():
    req = _requirements(
        must_have=[{"skill": "React", "min_years": 3}],
        nice_to_have=["GraphQL"],
        min_total_years=5,
        min_degree="Master's",
    )
    text = req.query_text()
    assert text.splitlines()[0] == "Frontend Engineer"
    assert "Must have: React (3+ yrs)" in text
    assert "Nice to have: GraphQL" in text
    assert "Experience: 5+ years" in text
    assert "Education: Master's degree or higher" in text
    parsed = req.to_parsed()
    assert parsed["text"] == text
    assert parsed["required_skills"] == ["React"] and parsed["preferred_skills"] == ["GraphQL"]


# --- extraction ------------------------------------------------------------------------------


def test_extract_sends_prompt_and_text_to_the_model():
    jd = "Backend engineer. Must have Go."
    fake = _fake(_requirements(must_have=[{"skill": "Go", "min_years": None}]))
    extract_requirements(f"  {jd}\n", model=fake)
    ((schema, messages),) = fake.calls
    assert schema is Requirements
    assert isinstance(messages[0], SystemMessage) and messages[0].content == prompts.EXTRACT_REQUIREMENTS
    assert isinstance(messages[1], HumanMessage) and messages[1].content == jd


def test_rule_floor_adds_years_and_missed_skills_as_nice_to_have():
    jd = (JD_DIR / "frontend_react.txt").read_text()
    llm_answer = _requirements(
        must_have=[
            {"skill": "React", "min_years": None},
            {"skill": "TypeScript", "min_years": 3},
            {"skill": "CSS", "min_years": None},
        ],
        nice_to_have=["Tailwind CSS", "Jest", "Next.js", "GraphQL"],
    )
    req = extract_requirements(jd, model=_fake(llm_answer))
    # "3+ years of experience with React and TypeScript": the rules tie 3 years to React.
    assert [(r.skill, r.min_years) for r in req.must_have] == [("React", 3), ("TypeScript", 3), ("CSS", None)]
    # The LLM skipped Cypress; the floor keeps it, but only as nice-to-have.
    assert "Cypress" in req.nice_to_have and "Cypress" not in req.must_have_skills


def test_rule_floor_never_adds_thresholds_the_llm_left_out():
    jd = "Python developer.\nRequirements:\n- 5+ years of Python\n- Bachelor's degree or equivalent experience"
    llm_answer = _requirements(must_have=[{"skill": "Python", "min_years": 5}])
    req = extract_requirements(jd, model=_fake(llm_answer))
    assert req.min_degree is None
    assert req.min_total_years is None


def test_rule_floor_keeps_llm_years_over_rule_years():
    rules = {"required_skills": ["Python"], "preferred_skills": [],
             "must_haves": [{"type": "skill_years", "skill": "Python", "min_years": 2}]}
    req = apply_rule_floor(_requirements(must_have=[{"skill": "Python", "min_years": 6}]), rules)
    assert req.must_have[0].min_years == 6


# --- knockout policy -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hands-on experience with PyTorch or TensorFlow and scikit-learn", {"pytorch", "tensorflow"}),
        ("Kafka and relational databases (Oracle or PostgreSQL)", {"oracle", "postgresql"}),
        ("Testing with Jest or Cypress", {"jest", "cypress"}),
        ("Strong Java and Spring Boot skills", set()),
        ("Bachelor's or Master's degree", set()),
    ],
)
def test_alternative_skills(text, expected):
    assert alternative_skills(text) == expected


@pytest.mark.parametrize(
    "jd, required",
    [
        ("Requirements:\n- Bachelor's degree in Computer Science", True),
        ("Requirements:\n- Bachelor's degree in Computer Science or equivalent", False),
        ("Requirements:\n- Master's degree preferred", False),
        ("Requirements:\n- Python\nNice to have:\n- PhD in a quantitative field", False),
        ("Find me React developers with a master's degree", True),
    ],
)
def test_degree_required(jd, required):
    assert degree_required(job_matcher.parse_job_description(jd)) is required


def test_policy_corrects_llm_knockouts_on_java_jd():
    """The live gpt-4o-mini answer for java_backend.txt, before the policy ran."""
    jd = (JD_DIR / "java_backend.txt").read_text()
    llm_answer = _requirements(
        role_title="Backend Java Engineer (Payments)",
        must_have=[{"skill": s, "min_years": None}
                   for s in ["Java", "Spring Boot", "Kafka", "Oracle", "PostgreSQL", "CI/CD"]],
        nice_to_have=["Kubernetes"],
        min_total_years=6,
        min_degree="Bachelor's",
    )
    req = extract_requirements(jd, model=_fake(llm_answer))
    assert req.must_have_skills == ["Java", "Spring Boot", "Kafka", "CI/CD"]
    assert {"Oracle", "PostgreSQL", "Kubernetes"} <= set(req.nice_to_have)
    assert req.min_degree is None  # "Bachelor's degree ... or equivalent"
    assert req.min_total_years == 6


@pytest.mark.parametrize("text", ["", "   \n"])
def test_extract_rejects_empty_text(text):
    with pytest.raises(ValueError):
        extract_requirements(text, model=_fake(_requirements()))


# --- ranking from requirements ---------------------------------------------------------------


def test_match_parsed_applies_requirements_as_knockouts(matcher):
    req = _requirements(must_have=[{"skill": "React", "min_years": 3}])
    result = matcher.match_parsed(req.to_parsed(), k=5, include_filtered=True)
    assert result["top_matches"], "expected React candidates in the fixture resumes"
    assert all("React" in m["matched_skills"] for m in result["top_matches"])
    failed = [f for out in result["filtered_out"] for f in out["failed_requirements"]]
    assert any("React" in reason for reason in failed)


def test_match_delegates_to_match_parsed(matcher):
    jd = (JD_DIR / "java_backend.txt").read_text()
    parsed = job_matcher.parse_job_description(jd)
    assert matcher.match(jd, k=3, details=True) == matcher.match_parsed(parsed, k=3, details=True)
