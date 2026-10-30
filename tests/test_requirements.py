import typing

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from resume_shortlister import prompts
from resume_shortlister.rag import job_matcher, resume_rag, skills_vocab
from resume_shortlister.requirements import (
    alternative_skills,
    as_stated,
    degree_required,
    extract_requirements,
    is_named,
    open_questions,
)
from resume_shortlister.schemas import DegreeLevel, Requirements
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
    req = _requirements(must_have=[{"skill": " Hotwire ", "min_years": None}])
    assert req.must_have_skills == ["Hotwire"]


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
    # Only structured requirements: no role title or model-written prose.
    assert text.splitlines()[0] == "Must have: React (3+ yrs)"
    assert "Frontend Engineer" not in text
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


def _must(*skills, years=None) -> list[dict]:
    years = years or {}
    return [{"skill": s, "min_years": years.get(s)} for s in skills]


def _decided(req: Requirements) -> tuple:
    """What the rules make stable: must-haves with years, skill nice-to-haves, total years, degree."""
    return (tuple((r.skill, r.min_years) for r in req.must_have),
            tuple(s for s in req.nice_to_have if s in skills_vocab.SKILLS), req.min_total_years, req.min_degree)


# Answers models actually gave for the same text (gpt-5.6-terra and gpt-6-luna, live): they
# differ in years, total years, optional skills and wording, and must all come out the same.
DIFFERENT_READINGS = {
    "fullstack_engineer.txt": [
        dict(must_have=_must("React", "Node.js", "TypeScript", years={"React": 3, "Node.js": 3}),
             nice_to_have=["PostgreSQL", "MongoDB", "Docker", "AWS", "GraphQL", "Next.js"]),
        dict(must_have=_must("React", "Node.js", "TypeScript"), nice_to_have=["GraphQL"], min_total_years=3),
    ],
    "ios_engineer.txt": [
        dict(must_have=_must("Swift", "SwiftUI", "XCTest", years={"Swift": 4}), nice_to_have=["iOS development", "Firebase"]),
        dict(must_have=_must("SwiftUI", "XCTest", "Swift"), nice_to_have=["Objective-C"], min_degree="Master's"),
    ],
    "java_backend.txt": [
        dict(must_have=_must("Java", "Spring Boot", "Kafka", "Oracle", "PostgreSQL", "CI/CD"), nice_to_have=["Kubernetes"],
             min_total_years=6, min_degree="Bachelor's"),
        dict(must_have=_must("Java", "Spring Boot", "Kafka", "Relational Databases"), nice_to_have=["trade settlement"]),
    ],
    "senior_python_ml.txt": [
        dict(must_have=_must("Machine Learning", "NLP", "Python", "scikit-learn", years={"Python": 5})),
        dict(must_have=_must("Python", "scikit-learn", "NLP", "Machine Learning", "LLMs", years={"Python": 6}),
             nice_to_have=["model deployment"], min_total_years=5),
    ],
    "data_engineer.txt": [
        dict(must_have=_must("Python", "SQL", "Airflow", years={"Python": 4}), nice_to_have=["cloud data warehousing"]),
        dict(must_have=_must("Python", "SQL", "Snowflake"), nice_to_have=["Airflow", "dbt"]),  # Airflow stands alone
    ],
}


@pytest.mark.parametrize("jd", sorted(DIFFERENT_READINGS))
def test_rules_make_different_readings_agree(jd):
    text = (JD_DIR / jd).read_text()
    results = {_decided(extract_requirements(text, model=_fake(_requirements(**reading))))
               for reading in DIFFERENT_READINGS[jd]}
    assert len(results) == 1, results


def test_rules_on_the_frontend_jd():
    jd = (JD_DIR / "frontend_react.txt").read_text()
    llm_answer = _requirements(must_have=_must("React", "TypeScript", "CSS", years={"TypeScript": 2}),
                               nice_to_have=["Tailwind CSS", "web accessibility (WCAG)", "Design systems"])
    req = extract_requirements(jd, model=_fake(llm_answer))
    # "3+ years of experience with React and TypeScript": the years go to both skills it names.
    # "A strong eye for accessibility (WCAG)" names one skill, but a practice is never a knockout.
    assert [(r.skill, r.min_years) for r in req.must_have] == [("React", 3), ("TypeScript", 3), ("CSS", None)]
    # Every other skill in the text, in text order; the model's "web accessibility (WCAG)" is the
    # same skill, and "Design systems", which the text never says, is dropped.
    assert req.nice_to_have == ["Tailwind CSS", "Jest", "Cypress", "Accessibility", "Next.js", "GraphQL"]


@pytest.mark.parametrize("text, must, skill_years, total", [
    ("Find me candidates with React and 3+ years experience", ["React"], {"React": 3}, None),
    ("Senior Python backend engineer with Django and PostgreSQL, 5+ years", ["Python", "Django", "PostgreSQL"], {}, 5),
    ("Requirements:\n- 6+ years of professional software development experience\n- Java", ["Java"], {}, 6),
    ("Requirements:\n- 3+ years of full-stack experience with React and Node.js", ["React", "Node.js"],
     {"React": 3, "Node.js": 3}, None),
    ("You have at least 5 years of experience operating infrastructure, and you have run Kubernetes.",
     ["Kubernetes"], {}, 5),
    ("Requirements:\n- Python: 4+ years\n- 2+ years of Go is a plus", ["Python"], {"Python": 4}, None),
])
def test_years_come_from_the_text(text, must, skill_years, total):
    req = extract_requirements(text, model=_fake(_requirements(must_have=_must(*must), min_total_years=99)))
    assert {r.skill: r.min_years for r in req.must_have if r.min_years} == skill_years
    assert req.min_total_years == total


def test_years_written_in_words_keep_the_models_reading():
    llm_answer = _requirements(must_have=_must("Python", years={"Python": 5}), min_total_years=8)
    req = extract_requirements("Python developer with five years of Python and eight overall", model=_fake(llm_answer))
    assert req.must_have[0].min_years == 5 and req.min_total_years == 8


@pytest.mark.parametrize("line", [
    "Familiarity with Kubernetes", "Kubernetes is a plus", "Kubernetes or similar orchestration",
    "Experience with an orchestrator such as Kubernetes or Nomad", "Exposure to Kubernetes would be great",
])
def test_optional_or_alternative_skills_are_never_must_haves(line):
    req = extract_requirements(f"Requirements:\n- Python\n- {line}", model=_fake(_requirements(must_have=_must("Kubernetes"))))
    assert req.must_have_skills == ["Python"] and "Kubernetes" in req.nice_to_have


def test_only_the_requirements_section_makes_must_haves():
    jd = (JD_DIR / "senior_python_ml.txt").read_text()  # "About the role" mentions NLP and LLMs
    req = extract_requirements(jd, model=_fake(_requirements(must_have=_must("LLMs", "Python"))))
    assert "LLMs" not in req.must_have_skills and "LLMs" in req.nice_to_have
    # Requirements naming one skill demand it even when the model left it out.
    assert req.must_have_skills == ["Python", "scikit-learn"]


def test_in_prose_a_lone_mention_is_left_to_the_model():
    jd = (JD_DIR / "devops_sre.txt").read_text()
    req = extract_requirements(jd, model=_fake(_requirements(must_have=_must("Terraform"))))
    assert req.must_have_skills == ["Terraform"] and req.min_total_years == 5
    assert {"Kubernetes", "Python", "Go", "Helm"} <= set(req.nice_to_have)


@pytest.mark.parametrize("jd, model_must, questions", [
    # The model kept Swift, not iOS: open.
    ("ios_engineer.txt", ["Swift", "SwiftUI", "XCTest"], [("4+ years of professional iOS development in Swift",
                                                         ("Swift",), ("iOS",))]),
    # None of the three became must-haves: open.
    ("senior_python_ml.txt", ["Python", "scikit-learn", "NLP", "Machine Learning"],
     [("Experience deploying models with Docker and Kubernetes on AWS", (), ("Docker", "Kubernetes", "AWS"))]),
    # Every multi-skill requirement kept all its skills: nothing to ask.
    ("java_backend.txt", ["Java", "Spring Boot", "Kafka"], []),
    ("frontend_react.txt", ["React", "TypeScript", "CSS"], []),
])
def test_open_questions_are_the_models_own_must_have_choices(jd, model_must, questions):
    text = (JD_DIR / jd).read_text()
    req = extract_requirements(text, model=_fake(_requirements(must_have=_must(*model_must))))
    assert [(q.requirement, q.must, q.nice) for q in open_questions(text, req)] == questions


def test_rule_floor_never_adds_thresholds_the_llm_left_out():
    jd = "Python developer.\nRequirements:\n- 5+ years of Python\n- Bachelor's degree or equivalent experience"
    llm_answer = _requirements(must_have=[{"skill": "Python", "min_years": 5}], min_degree="Bachelor's")
    req = extract_requirements(jd, model=_fake(llm_answer))
    assert req.min_degree is None
    assert req.min_total_years is None


# --- knockout policy -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hands-on experience with PyTorch or TensorFlow and scikit-learn", {"pytorch", "tensorflow"}),
        ("Kafka and relational databases (Oracle or PostgreSQL)", {"oracle", "postgresql"}),
        ("Testing with Jest or Cypress", {"jest", "cypress"}),
        ("A warehouse such as Snowflake or BigQuery", {"snowflake", "bigquery"}),
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
    """The live gpt-4o-mini answer for java_backend.txt, before the rules ran."""
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
    # Oracle / PostgreSQL are options; "Familiarity with Kubernetes and CI/CD" is softened.
    assert req.must_have_skills == ["Java", "Spring Boot", "Kafka"]
    assert {"Oracle", "PostgreSQL", "Kubernetes", "CI/CD"} <= set(req.nice_to_have)
    assert req.min_degree is None  # "Bachelor's degree ... or equivalent"
    assert req.min_total_years == 6


def test_policy_demotes_concepts_the_parser_cannot_check():
    """gpt-6-luna names broad concepts as must-haves; no resume lists them as skills."""
    jd = (JD_DIR / "java_backend.txt").read_text()
    llm_answer = _requirements(
        must_have=[{"skill": s, "min_years": None}
                   for s in ["Java", "Spring Boot", "Kafka", "Relational Databases"]],
    )
    req = extract_requirements(jd, model=_fake(llm_answer))
    assert req.must_have_skills == ["Java", "Spring Boot", "Kafka"]
    assert "Relational databases" in req.nice_to_have  # as the text writes it


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


@pytest.mark.parametrize("preference, stated", [
    ("Financial Services", "Financial services"),
    ("trade settlement   systems", "Trade settlement systems"),
    ("Cloud Data Warehousing", None),  # the text says "a cloud data warehouse"
    # Words that only qualify a skill go: what resumes write is the skill.
    ("Microsoft Excel", "Excel"), ("Advanced Excel", "Excel"), ("Excel skills", "Excel"),
    ("CompTIA certification", "CompTIA"),
    ("Microsoft 365", "Microsoft 365"),  # "Microsoft" stays when no name remains without it
])
def test_preferences_count_only_in_the_texts_own_words(preference, stated):
    text = ("- Experience in financial services or trade settlement systems\n- A cloud data warehouse\n"
            "- Advanced Excel skills; ITIL or CompTIA certification; Microsoft 365 administration")
    assert as_stated(preference, text) == stated


# --- skills outside the vocabulary ------------------------------------------------------------------

HR_SYSTEMS_JD = """HR Systems Analyst

Requirements:
- 3+ years of experience administering Workday and Ceridian Dayforce
- Strong knowledge of payroll compliance
- Advanced Excel skills
- 5+ years of human resources experience
- Networking fundamentals: VPN and SFTP

Nice to have:
- SuccessFactors or UKG
- Experience with benefits administration
"""


def test_any_skill_the_text_names_can_be_a_requirement():
    proposed = _requirements(
        role_title="HR Systems Analyst",
        must_have=_must("Workday", "Ceridian Dayforce", "Payroll Compliance", "Microsoft Excel", "Human Resources",
                        years={"Workday": 3, "Human Resources": 5}),
        nice_to_have=["VPN", "SFTP", "SuccessFactors", "UKG", "Benefits Administration", "Cloud HR Platforms"],
    )
    req = extract_requirements(HR_SYSTEMS_JD, model=_fake(proposed))
    # Names (Workday, Ceridian Dayforce) are checkable must-haves, with the years their line states;
    # "Microsoft Excel" is the vocabulary's Excel.
    assert [(r.skill, r.min_years) for r in req.must_have] == [
        ("Workday", 3), ("Ceridian Dayforce", 3), ("Excel", None)]
    # Concepts resumes paraphrase are nice-to-haves, and their years are total experience;
    # "Cloud HR Platforms", which the text never says, is dropped.
    assert req.nice_to_have == ["Payroll compliance", "Human resources", "VPN", "SFTP", "SuccessFactors", "UKG",
                                "Benefits administration"]
    assert req.min_total_years == 5
    # The model left a line naming several skills to nice-to-have: the recruiter is asked.
    (question,) = open_questions(HR_SYSTEMS_JD, req)
    assert question.nice == ("VPN", "SFTP")


@pytest.mark.parametrize("term, named", [
    ("Workday", True), ("Ceridian Dayforce", True), ("VPN", True),
    ("Payroll compliance", False), ("Human resources", False),
    ("Networking", False),  # capitalised only because it starts the line
])
def test_names_are_told_from_concepts_by_how_the_text_writes_them(term, named):
    assert is_named(term, HR_SYSTEMS_JD) is named


def test_a_word_the_text_also_writes_in_lower_case_is_a_concept():
    text = "Requirements:\n- 5+ years of accounting experience\n- Bachelor's degree in Accounting or Finance"
    assert not is_named("Accounting", text)
    assert is_named(".NET", "Requirements:\n- 3+ years of .net development") and is_named("TCP/IP", "- tcp/ip")
