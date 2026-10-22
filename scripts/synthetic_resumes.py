"""Template-driven synthetic resumes for screening at scale.

Each resume is assembled from a role family (titles, skill pools, project
nouns), a seniority level and a seeded random generator, so the same seed
always yields the same resumes. The variety is deliberate, to exercise the
RAG pipeline and the agent's screening rounds:

- 11 role families with overlapping skills (React appears in frontend and
  full-stack resumes, Python in five families);
- junior to lead seniority, with skills picked up at different points in a
  career so per-skill years differ from total years;
- occasional career changers, missing degrees, certifications and projects;
- several spellings of section headers and date ranges.

The output uses the same shape as ``generate_sample_resumes.RESUMES``:
``{stem: {"format", "name", "contact", "sections": {heading: [lines]}}}``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

SEED = 2026
# Newest start date used for roles; the current role always ends "Present".
REFERENCE_YEAR, REFERENCE_MONTH = 2026, 6
FORMATS = ("pdf", "docx", "txt")
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# Share of resumes with an edge case, kept low so most candidates are "normal".
CAREER_CHANGE_RATE = 0.15
NO_DEGREE_RATE = 0.08
CERTIFICATION_RATE = 0.25
JUNIOR_PROJECTS_RATE = 0.6


@dataclass(frozen=True)
class Seniority:
    level: int  # index into RoleFamily.titles
    min_years: float
    max_years: float
    max_roles: int
    weight: int  # relative share of the dataset


SENIORITIES = (
    Seniority(0, 0.8, 3.0, 2, 25),
    Seniority(1, 3.0, 6.0, 3, 35),
    Seniority(2, 6.0, 10.0, 3, 30),
    Seniority(3, 10.0, 15.0, 4, 10),
)


@dataclass(frozen=True)
class RoleFamily:
    key: str
    titles: tuple[str, str, str, str]  # junior, mid, senior, lead
    core: tuple[str, ...]  # always on the resume
    choose: tuple[dict[str, int], ...] = ()  # one weighted pick per group
    extras: tuple[str, ...] = ()  # a seniority-dependent sample of these
    things: tuple[str, ...] = ("internal platform",)  # what the candidate built
    reduce: tuple[str, ...] = ("latency",)  # metrics where lower is better
    increase: tuple[str, ...] = ()  # metrics where higher is better
    fields: tuple[str, ...] = ("Computer Science",)  # degree subjects
    phd_rate: float = 0.02


FAMILIES = (
    RoleFamily(
        "frontend",
        ("Junior Frontend Developer", "Frontend Engineer", "Senior Frontend Engineer",
         "Lead Frontend Engineer"),
        core=("JavaScript", "HTML", "CSS"),
        choose=({"React": 3, "Vue.js": 1, "Angular": 1}, {"TypeScript": 3, "Redux": 1}),
        extras=("Next.js", "Redux", "Tailwind CSS", "GraphQL", "Jest", "Cypress", "Figma",
                "Storybook", "Accessibility (WCAG)", "Git", "CI/CD"),
        things=("design system", "checkout flow", "analytics dashboard", "customer portal",
                "marketing site"),
        reduce=("page load time", "bundle size"), increase=("conversion rate", "Lighthouse score"),
        fields=("Computer Science", "Software Engineering", "Information Technology"),
    ),
    RoleFamily(
        "fullstack",
        ("Junior Full Stack Developer", "Full Stack Engineer", "Senior Full Stack Engineer",
         "Lead Full Stack Engineer"),
        core=("JavaScript", "Node.js"),
        choose=({"React": 3, "Vue.js": 1}, {"Express": 2, "GraphQL": 1},
                {"PostgreSQL": 1, "MongoDB": 1}),
        extras=("TypeScript", "Next.js", "Redux", "Docker", "AWS", "Jest", "REST APIs", "Redis",
                "CI/CD", "Tailwind CSS", "Git"),
        things=("booking platform", "SaaS billing app", "admin console", "public REST API",
                "real-time chat feature"),
        reduce=("response time", "infrastructure cost", "customer churn"),
        increase=("release frequency",),
        fields=("Computer Science", "Software Engineering", "Information Systems"),
    ),
    RoleFamily(
        "python_backend",
        ("Junior Backend Developer", "Backend Engineer", "Senior Backend Engineer",
         "Staff Backend Engineer"),
        core=("Python", "SQL"),
        choose=({"Django": 2, "FastAPI": 2, "Flask": 1}, {"PostgreSQL": 3, "MySQL": 1}),
        extras=("Django REST Framework", "Celery", "Redis", "Docker", "AWS", "Kubernetes",
                "REST APIs", "Microservices", "pytest", "Git", "Kafka", "GraphQL"),
        things=("payments API", "order management service", "notification service",
                "partner integration layer", "reporting backend"),
        reduce=("p95 latency", "error rate", "cloud spend", "deployment time"),
        fields=("Computer Science", "Software Engineering", "Mathematics"),
    ),
    RoleFamily(
        "java_backend",
        ("Junior Java Developer", "Java Developer", "Senior Java Engineer",
         "Principal Java Engineer"),
        core=("Java", "SQL"),
        choose=({"Spring Boot": 3, "Spring": 1}, {"PostgreSQL": 1, "Oracle": 1, "MySQL": 1}),
        extras=("Hibernate", "Kafka", "Microservices", "Maven", "JUnit", "Docker", "Kubernetes",
                "AWS", "REST APIs", "Jenkins", "Redis", "Kotlin"),
        things=("trading gateway", "claims processing system", "core banking service",
                "inventory platform", "event-driven ledger"),
        reduce=("batch run time", "incident count", "p99 latency"), increase=("throughput",),
        fields=("Computer Science", "Computer Engineering", "Information Technology"),
    ),
    RoleFamily(
        "data_science",
        ("Junior Data Scientist", "Data Scientist", "Senior Data Scientist",
         "Lead Data Scientist"),
        core=("Python", "SQL", "Statistics", "pandas", "scikit-learn"),
        choose=({"XGBoost": 2, "PyTorch": 1, "TensorFlow": 1}, {"Tableau": 1, "Power BI": 1}),
        extras=("NumPy", "A/B Testing", "Spark", "Airflow", "R", "Machine Learning",
                "Deep Learning", "Snowflake", "Databricks", "Git"),
        things=("churn model", "demand forecast", "pricing experiment", "customer segmentation model",
                "fraud scoring model"),
        reduce=("false-positive rate",), increase=("forecast accuracy", "retention", "marketing ROI"),
        fields=("Statistics", "Mathematics", "Economics", "Computer Science"),
        phd_rate=0.12,
    ),
    RoleFamily(
        "ml_engineer",
        ("Junior Machine Learning Engineer", "Machine Learning Engineer",
         "Senior Machine Learning Engineer", "Staff Machine Learning Engineer"),
        core=("Python", "Machine Learning"),
        choose=({"PyTorch": 2, "TensorFlow": 1}, {"NLP": 1, "Computer Vision": 1, "LLMs": 1}),
        extras=("Docker", "Kubernetes", "MLflow", "Hugging Face", "AWS", "AWS SageMaker",
                "Deep Learning", "RAG", "Kubeflow", "FastAPI", "CUDA", "scikit-learn", "Spark"),
        things=("recommendation model", "document classifier", "semantic search service",
                "image tagging pipeline", "model serving platform"),
        reduce=("inference latency", "GPU cost", "training time"), increase=("model accuracy",),
        fields=("Computer Science", "Machine Learning", "Electrical Engineering"),
        phd_rate=0.2,
    ),
    RoleFamily(
        "data_engineer",
        ("Junior Data Engineer", "Data Engineer", "Senior Data Engineer", "Lead Data Engineer"),
        core=("Python", "SQL", "ETL"),
        choose=({"Spark": 2, "Databricks": 1}, {"Airflow": 2, "dbt": 1},
                {"Snowflake": 1, "AWS Glue": 1, "PostgreSQL": 1}),
        extras=("Kafka", "AWS", "Scala", "Docker", "Hadoop", "GCP", "Terraform", "Git"),
        things=("event ingestion pipeline", "customer data warehouse", "streaming platform",
                "data quality framework", "finance reporting mart"),
        reduce=("pipeline run time", "storage cost", "failed job rate"),
        increase=("data freshness",),
        fields=("Computer Science", "Information Systems", "Mathematics"),
    ),
    RoleFamily(
        "devops",
        ("Junior DevOps Engineer", "DevOps Engineer", "Senior Site Reliability Engineer",
         "Lead Platform Engineer"),
        core=("Linux", "Docker", "Kubernetes"),
        choose=({"AWS": 3, "GCP": 1, "Azure": 1}, {"Terraform": 2, "Ansible": 1}),
        extras=("Helm", "Prometheus", "Grafana", "Jenkins", "GitHub Actions", "CI/CD", "Bash",
                "Python", "Go", "GitLab"),
        things=("Kubernetes platform", "CI/CD pipeline", "observability stack",
                "multi-region failover setup", "infrastructure-as-code module"),
        reduce=("deployment time", "mean time to recovery", "cloud spend"),
        increase=("deployment frequency",),
        fields=("Computer Science", "Information Technology", "Computer Engineering"),
    ),
    RoleFamily(
        "ios",
        ("Junior iOS Developer", "iOS Engineer", "Senior iOS Engineer", "Lead iOS Engineer"),
        core=("Swift", "iOS"),
        choose=({"SwiftUI": 2, "Objective-C": 1},),
        extras=("XCTest", "Firebase", "Git", "CI/CD", "REST APIs", "Figma", "Agile"),
        things=("mobile banking app", "fitness tracking app", "offline sync layer",
                "in-app purchase flow", "push notification service"),
        reduce=("crash rate", "app start time"), increase=("App Store rating", "session length"),
        fields=("Computer Science", "Software Engineering"),
    ),
    RoleFamily(
        "android",
        ("Junior Android Developer", "Android Engineer", "Senior Android Engineer",
         "Lead Android Engineer"),
        core=("Kotlin", "Android"),
        choose=({"Jetpack Compose": 2, "Java": 1},),
        extras=("Firebase", "Git", "CI/CD", "REST APIs", "JUnit", "Agile", "Flutter", "Dart"),
        things=("ride-hailing app", "media player", "offline maps feature", "checkout flow",
                "loyalty app"),
        reduce=("crash rate", "app size", "battery usage"), increase=("Play Store rating",),
        fields=("Computer Science", "Software Engineering"),
    ),
    RoleFamily(
        "qa",
        ("QA Analyst", "QA Automation Engineer", "Senior QA Automation Engineer",
         "QA Lead"),
        core=("Selenium", "SQL"),
        choose=({"Python": 1, "Java": 1, "JavaScript": 1}, {"pytest": 1, "JUnit": 1, "Cypress": 1}),
        extras=("Jenkins", "CI/CD", "Git", "Agile", "Scrum", "REST APIs", "Docker", "Jest"),
        things=("regression suite", "API contract test suite", "smoke test pack",
                "end-to-end test framework", "load testing tool"),
        reduce=("escaped defects", "test run time", "release cycle time", "flaky test rate"),
        fields=("Computer Science", "Information Technology"),
    ),
)

FIRST_NAMES = (
    "Aarav", "Abigail", "Adrian", "Aisha", "Akira", "Amara", "Andre", "Anika", "Ben", "Bianca",
    "Caleb", "Camila", "Chen", "Chloe", "Dev", "Diana", "Elena", "Emeka", "Ethan", "Farah",
    "Gabriel", "Grace", "Hana", "Hassan", "Ian", "Ines", "Isaac", "Jasmine", "Javier", "Kai",
    "Kavya", "Kenji", "Lara", "Leo", "Lucia", "Mateo", "Maya", "Mohammed", "Nadia", "Nikhil",
    "Noah", "Nora", "Omar", "Paula", "Rafael", "Riya", "Ryan", "Samuel", "Sara", "Tariq",
    "Tomas", "Uma", "Victor", "Yara", "Yusuf", "Zara", "Zoe",
)
LAST_NAMES = (
    "Adeyemi", "Alvarez", "Andersen", "Bauer", "Bose", "Carter", "Castillo", "Chowdhury",
    "Costa", "Dubois", "Evans", "Fischer", "Fernandes", "Gupta", "Haddad", "Hughes", "Ivanova",
    "Jensen", "Kapoor", "Khan", "Kowalski", "Laurent", "Lim", "Lopez", "Mehta", "Morales",
    "Murphy", "Nakamura", "Nguyen", "Novak", "Okafor", "Park", "Petrov", "Quinn", "Reddy",
    "Reyes", "Santos", "Schmidt", "Shah", "Silva", "Singh", "Sato", "Tanaka", "Torres",
    "Turner", "Varga", "Wagner", "Walsh", "Yamamoto", "Zhang",
)
COMPANIES = (
    "Acme Payments", "Bluefin Analytics", "Brightside Labs", "Cobalt Health", "Contoso Retail",
    "Driftwood Media", "Evergreen Bank", "Fabrikam Logistics", "Globex Insurance",
    "Helios Energy", "Initech", "Juniper Travel", "Keystone Telecom", "Lumen Robotics",
    "Meridian Foods", "Northwind Traders", "Orbital Games", "Pinnacle Software", "Quartz AI",
    "Redwood Mobility", "Summit Learning", "Tidal Streaming", "Umbra Security",
    "Vertex Commerce", "Willow Biotech", "Zephyr Cloud",
)
UNIVERSITIES = (
    "University of Texas at Austin", "University of Washington", "Georgia Institute of Technology",
    "University of Michigan", "Purdue University", "University of Toronto", "Arizona State University",
    "University of Illinois Urbana-Champaign", "Imperial College London", "National University of Singapore",
    "University of Melbourne", "Technical University of Munich", "IIT Bombay", "University of Waterloo",
)
BOOTCAMPS = ("General Assembly", "Le Wagon", "App Academy", "Flatiron School")
CITIES = (
    "Austin, TX", "Seattle, WA", "Chicago, IL", "New York, NY", "Boston, MA", "Denver, CO",
    "Toronto, ON", "London, UK", "Berlin, Germany", "Bangalore, India", "Singapore", "Remote",
)
CERTIFICATIONS = {
    "AWS": "AWS Certified Solutions Architect",
    "Kubernetes": "Certified Kubernetes Administrator (CKA)",
    "Azure": "Microsoft Certified: Azure Developer",
    "GCP": "Google Cloud Professional Data Engineer",
    "Scrum": "Professional Scrum Master I",
    "Terraform": "HashiCorp Certified: Terraform Associate",
}

HEADERS = {
    "Summary": ("Summary", "Professional Summary", "Profile"),
    "Skills": ("Skills", "Technical Skills", "Core Competencies"),
    "Experience": ("Experience", "Work Experience", "Professional Experience", "Employment History"),
    "Projects": ("Projects", "Personal Projects"),
    "Education": ("Education", "Academic Background"),
    "Certifications": ("Certifications",),
}
# Placeholders: {p1}-{p3} are primary stack skills (languages, frameworks,
# databases), {s1}-{s2} any other skill of the role, {outcome} a measured result.
SUMMARIES = (
    "{title} with {years} years of experience building {thing}s with {p1} and {p2}.",
    "{title} with {years}+ years of experience, specialising in {p1}, {p2} and {p3}.",
    "{title} who has spent {years} years shipping {thing}s; strongest in {p1} and {s1}.",
)
BULLETS = (
    "- Delivered a new {thing} using {p1} and {s1}, resulting in {outcome}.",
    "- Rebuilt the {thing} on {p1}, delivering {outcome}.",
    "- Introduced {s1} on the {thing} team, contributing to {outcome}.",
    "- Owned the {thing} end to end with {p1}, {s1} and {s2}.",
)
LEAD_BULLETS = (
    "- Mentored {n} engineers and set {p1} standards across the team.",
    "- Led a team of {n} engineers delivering the {thing}.",
)
PROJECT_LINES = (
    "{Thing} (personal project) - built with {p1} and {s1}; open source on GitHub.",
    "Hackathon winner: prototype {thing} using {p1} in 48 hours.",
)
OUTCOMES = {"reduce": "a {pct}% reduction in {metric}", "increase": "a {pct}% increase in {metric}"}


@dataclass
class _Role:
    title: str
    company: str
    start: tuple[int, int]  # (year, month)
    end: tuple[int, int] | None  # None means Present
    stack: list[str] = field(default_factory=list)  # primary skills used in this role
    extras: list[str] = field(default_factory=list)  # supporting skills used in this role


def _months_before(year: int, month: int, months: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) - months
    return index // 12, index % 12 + 1


def _format_date(date: tuple[int, int] | None, style: str) -> str:
    if date is None:
        return "Present"
    year, month = date
    if style == "year":
        return str(year)
    if style == "numeric":
        return f"{month:02d}/{year}"
    return f"{MONTH_NAMES[month - 1]} {year}"


def _weighted(rng: random.Random, options: dict[str, int]) -> str:
    return rng.choices(list(options), weights=list(options.values()))[0]


def _unique_names(rng: random.Random, count: int, exclude: set[str]) -> list[str]:
    pairs = [f"{first} {last}" for first in FIRST_NAMES for last in LAST_NAMES]
    pairs = [name for name in pairs if name not in exclude]
    if count > len(pairs):
        raise ValueError(f"Can only build {len(pairs)} unique names, asked for {count}")
    return rng.sample(pairs, count)


def _pick_skills(rng: random.Random, family: RoleFamily, seniority: Seniority) -> tuple[list[str], list[str]]:
    """``(stack, extras)``: the core plus one pick per choice group, then a sample of extras."""
    stack = list(dict.fromkeys([*family.core, *(_weighted(rng, group) for group in family.choose)]))
    extra_count = min(len(family.extras), 2 + 2 * seniority.level + rng.randint(0, 2))
    extras = [s for s in rng.sample(family.extras, extra_count) if s not in stack]
    return stack, extras


def _plan_roles(
    rng: random.Random, family: RoleFamily, seniority: Seniority, stack: list[str], extras: list[str]
) -> list[_Role]:
    """Roles newest first. The stack runs through the whole career; extras are
    picked up along the way, so older roles use fewer of them."""
    total_months = round(rng.uniform(seniority.min_years, seniority.max_years) * 12)
    role_count = max(1, min(seniority.max_roles, total_months // 18, rng.randint(1, seniority.max_roles)))
    cuts = sorted(rng.sample(range(6, total_months - 5), role_count - 1)) if role_count > 1 else []
    lengths = [b - a for a, b in zip([0, *cuts], [*cuts, total_months])][::-1]  # newest first

    companies = rng.sample(COMPANIES, role_count)
    roles, end = [], None
    cursor = (REFERENCE_YEAR, REFERENCE_MONTH)
    for i, length in enumerate(lengths):
        start = _months_before(*cursor, length)
        level = max(0, seniority.level - i)
        roles.append(_Role(family.titles[level], companies[i], start, end))
        # The previous job ended a month before this one started, sometimes with a short break.
        cursor = end = _months_before(*start, rng.choice((1, 1, 1, 2, 3, 5)))

    for i, role in enumerate(roles):
        share = 1.0 - i / (len(roles) + 1)
        role.stack = list(stack)
        role.extras = extras[: max(1, round(len(extras) * share))]
    return roles


def _sample(rng: random.Random, items: list[str], k: int) -> list[str]:
    """``k`` distinct items when there are enough, otherwise cycle through them."""
    return rng.sample(items, k) if len(items) >= k else (items * k)[:k]


def _outcome(rng: random.Random, family: RoleFamily) -> str:
    direction = rng.choice([d for d in OUTCOMES if getattr(family, d)])
    metric = rng.choice(getattr(family, direction))
    return OUTCOMES[direction].format(pct=rng.randint(12, 60), metric=metric)


def _fill(
    template: str, rng: random.Random, family: RoleFamily, stack: list[str], extras: list[str], **extra
) -> str:
    p1, p2, p3 = _sample(rng, stack, 3)
    s1, s2 = _sample(rng, [s for s in extras if s != p1] or [s for s in stack if s != p1] or stack, 2)
    thing = rng.choice(family.things)
    return template.format(
        p1=p1, p2=p2, p3=p3, s1=s1, s2=s2, thing=thing, Thing=thing.capitalize(),
        outcome=_outcome(rng, family), n=rng.randint(3, 9), **extra,
    )


def _career_change(rng: random.Random, roles: list[_Role], family: RoleFamily) -> None:
    """Turn the oldest role into one from another family (e.g. QA before frontend)."""
    other = rng.choice([f for f in FAMILIES if f.key != family.key])
    oldest = roles[-1]
    oldest.title = other.titles[0]
    oldest.stack, oldest.extras = list(other.core), []


def _education(rng: random.Random, family: RoleFamily, grad_year: int, seniority: Seniority) -> list[str]:
    university = rng.choice(UNIVERSITIES)
    subject = rng.choice(family.fields)
    roll = rng.random()
    if roll < NO_DEGREE_RATE:
        return [f"Full-Stack Web Development Bootcamp, {rng.choice(BOOTCAMPS)}, {grad_year}"]
    if roll < NO_DEGREE_RATE + family.phd_rate and seniority.level >= 1:
        return [
            f"Ph.D. {subject}, {university}, {grad_year}",
            f"B.S. {subject}, {rng.choice(UNIVERSITIES)}, {grad_year - 5}",
        ]
    if roll < 0.35:
        degree = rng.choice(("M.S. {s}", "Master of Science in {s}", "M.Sc. {s}"))
    else:
        degree = rng.choice(("B.S. {s}", "Bachelor of Science in {s}", "B.Sc. {s}", "B.Tech {s}"))
    return [f"{degree.format(s=subject)}, {university}, {grad_year}"]


def build_resume(rng: random.Random, name: str, family: RoleFamily, fmt: str) -> dict:
    """One resume in the ``generate_sample_resumes`` data shape."""
    seniority = rng.choices(SENIORITIES, weights=[s.weight for s in SENIORITIES])[0]
    stack, extras = _pick_skills(rng, family, seniority)
    roles = _plan_roles(rng, family, seniority, stack, extras)
    if len(roles) > 1 and rng.random() < CAREER_CHANGE_RATE:
        _career_change(rng, roles, family)

    # PDFs use the core latin-1 font, so they keep plain ASCII hyphens.
    dash = " - " if fmt == "pdf" else rng.choice((" – ", " - ", " to "))
    date_style = rng.choice(("month", "year", "numeric"))
    experience = []
    for role in roles:
        dates = f"{_format_date(role.start, date_style)}{dash}{_format_date(role.end, date_style)}"
        experience.append(f"{role.title}, {role.company} ({dates})")
        bullet_count = rng.randint(1, 2) + (1 if role is roles[0] else 0)
        for template in rng.sample(BULLETS, bullet_count):
            experience.append(_fill(template, rng, family, role.stack, role.extras))
        if role is roles[0] and seniority.level >= 2:
            experience.append(_fill(rng.choice(LEAD_BULLETS), rng, family, role.stack, role.extras))

    total_years = REFERENCE_YEAR - roles[-1].start[0]
    title = roles[0].title
    summary = _fill(
        rng.choice(SUMMARIES), rng, family, stack, extras, title=title, years=max(1, total_years)
    )

    sections = {
        rng.choice(HEADERS["Summary"]): [summary],
        rng.choice(HEADERS["Skills"]): [", ".join(stack + extras)],
        rng.choice(HEADERS["Experience"]): experience,
    }
    if seniority.level == 0 and rng.random() < JUNIOR_PROJECTS_RATE:
        sections[rng.choice(HEADERS["Projects"])] = [
            _fill(line, rng, family, stack, extras) for line in PROJECT_LINES
        ]
    sections[rng.choice(HEADERS["Education"])] = _education(
        rng, family, roles[-1].start[0] - rng.choice((0, 0, 1)), seniority
    )
    certifiable = [CERTIFICATIONS[s] for s in stack + extras if s in CERTIFICATIONS]
    if certifiable and rng.random() < CERTIFICATION_RATE:
        sections["Certifications"] = [f"{rng.choice(certifiable)} ({rng.randint(2019, REFERENCE_YEAR)})"]

    first, last = name.split(" ", 1)
    contact = (
        f"{first.lower()}.{last.lower()}@example.com | (555) 0{rng.randint(10, 99)}-"
        f"{rng.randint(1000, 9999)} | {rng.choice(CITIES)}"
    )
    return {"format": fmt, "name": name, "contact": contact, "family": family.key, "sections": sections}


def build_resumes(count: int, seed: int = SEED, exclude_names: set[str] | None = None) -> dict[str, dict]:
    """``count`` synthetic resumes keyed by file stem, identical for the same ``seed``.

    Role families are dealt round-robin (in a seeded order) so every family is
    represented evenly; file formats rotate through PDF, DOCX and TXT.
    """
    rng = random.Random(seed)
    names = _unique_names(rng, count, exclude_names or set())
    families = rng.sample(FAMILIES, len(FAMILIES))
    resumes = {}
    for i, name in enumerate(names):
        stem = "resume_" + name.lower().replace(" ", "_")
        resumes[stem] = build_resume(rng, name, families[i % len(families)], FORMATS[i % len(FORMATS)])
    return resumes
