"""Skills vocabulary shared by the resume indexer and the job matcher.

Maps canonical skill names to the spellings that appear in resumes and job
descriptions, so that "sklearn", "scikit learn" and "scikit-learn" all count as
the same skill on both sides of a match.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Canonical name -> aliases (lower-case; the canonical name itself is always an alias).
SKILLS: dict[str, list[str]] = {
    # Languages
    "Python": ["python3"],
    "Java": [],
    "JavaScript": ["js", "ecmascript"],
    "TypeScript": [],
    "Go": ["golang"],
    "Rust": [],
    "C++": ["cpp"],
    "C#": ["csharp", "c sharp"],
    "C": [],
    "Scala": [],
    "Kotlin": [],
    "Swift": [],
    "Objective-C": ["objective c", "objc"],
    "Ruby": [],
    "PHP": [],
    "R": [],
    "SQL": [],
    "Bash": ["shell scripting", "shell"],
    "Dart": [],
    "MATLAB": [],
    "HTML": ["html5"],
    "CSS": ["css3"],
    # Web and backend frameworks
    "Django": [],
    "Django REST Framework": ["drf", "django rest"],
    "Flask": [],
    "FastAPI": ["fast api"],
    "Spring": [],
    "Spring Boot": ["springboot"],
    "Hibernate": [],
    "Node.js": ["nodejs", "node js", "node"],
    "Express": ["express.js", "expressjs"],
    "GraphQL": [],
    "REST APIs": ["rest", "rest api", "restful", "restful apis", "rest apis"],
    "Microservices": ["microservice", "micro-services"],
    "Celery": [],
    # Frontend and mobile
    "React": ["react.js", "reactjs"],
    "Next.js": ["nextjs"],
    "Redux": [],
    "Angular": ["angularjs"],
    "Vue.js": ["vue", "vuejs"],
    "Tailwind CSS": ["tailwind"],
    "SwiftUI": [],
    "Jetpack Compose": [],
    "Flutter": [],
    "Android": [],
    "iOS": [],
    "Firebase": [],
    "Figma": [],
    # Data and ML
    "Machine Learning": ["ml", "machine-learning"],
    "Deep Learning": ["deep-learning"],
    "NLP": ["natural language processing"],
    "Computer Vision": [],
    "LLMs": ["llm", "large language model", "large language models"],
    "Generative AI": ["genai", "gen ai"],
    "RAG": ["retrieval-augmented generation", "retrieval augmented generation"],
    "MLOps": [],
    "Statistics": ["statistical modeling", "statistical modelling"],
    "A/B Testing": ["a/b tests", "ab testing"],
    "pandas": [],
    "NumPy": [],
    "scikit-learn": ["sklearn", "scikit learn"],
    "PyTorch": ["torch"],
    "TensorFlow": ["tensorflow serving"],
    "Keras": [],
    "Hugging Face": ["huggingface", "hugging face transformers", "transformers"],
    "XGBoost": [],
    "OpenCV": [],
    "CUDA": [],
    "MLflow": [],
    "Kubeflow": [],
    "AWS SageMaker": ["sagemaker", "amazon sagemaker"],
    "Spark": ["apache spark", "pyspark"],
    "Hadoop": [],
    "Airflow": ["apache airflow"],
    "Kafka": ["apache kafka"],
    "Kinesis": ["amazon kinesis", "aws kinesis"],
    "Databricks": [],
    "Snowflake": [],
    "BigQuery": ["google bigquery"],
    "dbt": [],
    "AWS Glue": [],
    "ETL": ["elt"],
    "Tableau": [],
    "Power BI": ["powerbi"],
    # Databases
    "PostgreSQL": ["postgres"],
    "MySQL": [],
    "SQLite": [],
    "MongoDB": ["mongo"],
    "Redis": [],
    "Oracle": [],
    "Elasticsearch": ["elastic search"],
    "Cassandra": [],
    "DynamoDB": [],
    # Cloud, DevOps and tooling
    "AWS": ["amazon web services"],
    "GCP": ["google cloud", "google cloud platform"],
    "Azure": ["microsoft azure"],
    "Docker": [],
    "Kubernetes": ["k8s"],
    "Helm": [],
    "Argo CD": ["argocd"],
    "Terraform": [],
    "Ansible": [],
    "Jenkins": [],
    "GitHub Actions": [],
    "GitLab": ["gitlab ci"],
    "CI/CD": ["ci cd", "continuous integration", "continuous delivery"],
    "Git": [],
    "Linux": [],
    "Prometheus": [],
    "Grafana": [],
    "Maven": [],
    # Testing
    "pytest": [],
    "JUnit": [],
    "Jest": [],
    "Cypress": [],
    "Selenium": [],
    "XCTest": [],
    # Microsoft stack and IT operations: the common tools' other spellings, found in real resumes
    ".NET": [".net framework", ".net core", "dotnet"],
    "ASP.NET": ["asp .net", "asp.net mvc", "asp.net core"],
    "VB.NET": ["vb .net", "visual basic .net"],
    "SQL Server": ["mssql", "ms sql", "ms sql server", "microsoft sql server"],
    "PowerShell": ["power shell"],
    "Windows Server": [
        "windows nt server", "windows 2000 server", "windows 2003 server", "windows 2008 server",
        "windows 2012 server", "server 2003", "server 2008", "server 2012", "server 2016", "server 2019",
        "server 2022",
    ],
    "Active Directory": ["ad ds", "active directory domain services"],
    "VMware": ["vsphere", "esxi", "vcenter"],
    "Hyper-V": ["hyperv", "hyper v"],
    # Business and finance tools
    "Microsoft Office": ["ms office", "microsoft office suite", "ms office suite"],
    "Excel": ["microsoft excel", "ms excel"],
    "GAAP": ["us gaap", "u.s. gaap", "generally accepted accounting principles"],
    "IFRS": ["international financial reporting standards"],
    "QuickBooks": ["quick books", "quickbooks online"],
    "SAP": [],
    # Practices
    "Agile": [],
    "Scrum": [],
    "Accessibility": ["web accessibility", "wcag", "a11y"],
    # Certifications
    "CKA": ["certified kubernetes administrator"],
}

# Practices and certifications a resume often does not name even when the candidate has
# them: they rank candidates but never knock one out (``requirements`` keeps them nice-to-have).
NICE_ONLY = {"Accessibility", "CKA"}

# Aliases that are ordinary words: in free text they only count when spelled exactly
# like this, so "react to incidents" is not read as React but "Staff ML Engineer" is ML.
CASE_SENSITIVE = {
    "Go", "Rust", "Swift", "Ruby", "Dart", "Spring", "Flask", "Express", "React", "Redux",
    "Angular", "Spark", "Helm", "Jest", "Oracle", "Agile", "Scrum", "Snowflake", "Android",
    "Keras", "ML", "REST", "JS", "Excel", "SAP",
}
_CASE_SENSITIVE_BY_ALIAS = {spelling.lower(): spelling for spelling in CASE_SENSITIVE}
# Too short or ambiguous to find in prose at all; only recognised as an item of a skills list.
LIST_ONLY = {"c", "r", "shell", "node", "torch", "transformers", "elt"}

_ALIAS_TO_SKILL: dict[str, str] = {}
for _canonical, _aliases in SKILLS.items():
    for _alias in [_canonical.lower(), *_aliases]:
        _ALIAS_TO_SKILL[_alias] = _canonical

_VERSION_SUFFIX_RE = re.compile(r"\s+v?\d+(?:\.\d+)*\+?$")


def normalize_skill(raw: str) -> str | None:
    """Return the canonical name for ``raw`` (e.g. ``"sklearn"`` -> ``"scikit-learn"``).

    Trailing versions are ignored (``"Java 17"`` -> ``"Java"``). Returns
    ``None`` when the skill is not in the vocabulary.
    """
    key = " ".join(raw.strip().lower().split())
    if key in _ALIAS_TO_SKILL:
        return _ALIAS_TO_SKILL[key]
    key = _VERSION_SUFFIX_RE.sub("", key)
    return _ALIAS_TO_SKILL.get(key)


@lru_cache(maxsize=1)
def _free_text_patterns() -> tuple[re.Pattern, re.Pattern]:
    insensitive, sensitive = [], []
    for alias, canonical in _ALIAS_TO_SKILL.items():
        if alias in LIST_ONLY:
            continue
        if alias in _CASE_SENSITIVE_BY_ALIAS:
            sensitive.append(_CASE_SENSITIVE_BY_ALIAS[alias])
        else:
            insensitive.append(alias)

    def build(aliases: list[str], flags: int) -> re.Pattern:
        # Longest first so "Spring Boot" wins over "Spring" at the same position; the words of
        # a multi-word alias may be split by any whitespace, a line break in PDF text included.
        alternatives = "|".join(re.escape(a).replace(r"\ ", r"\s+") for a in sorted(aliases, key=len, reverse=True))
        return re.compile(rf"(?<![\w+#.])(?:{alternatives})(?![\w+#]|\.\w)", flags)

    return build(insensitive, re.IGNORECASE), build(sensitive, 0)


def find_skills(text: str) -> list[str]:
    """Find vocabulary skills mentioned anywhere in free text, in order of first mention."""
    found: list[tuple[int, int, str]] = []
    for pattern in _free_text_patterns():
        for match in pattern.finditer(text):
            found.append((match.start(), match.end(), _ALIAS_TO_SKILL[" ".join(match.group(0).lower().split())]))
    # Drop matches nested inside a longer match from the other pattern.
    found.sort(key=lambda m: (m[0], -(m[1] - m[0])))
    skills, covered_until = [], -1
    for start, end, skill in found:
        if start < covered_until:
            continue
        covered_until = end
        if skill not in skills:
            skills.append(skill)
    return skills


# Using a framework implies using its language (years with Django count as years with
# Python), and a managed service implies its cloud.
IMPLIED_SKILLS = {
    "Django": "Python", "Django REST Framework": "Python", "Flask": "Python",
    "FastAPI": "Python", "Celery": "Python", "pytest": "Python", "pandas": "Python",
    "NumPy": "Python", "scikit-learn": "Python", "PyTorch": "Python", "Airflow": "Python",
    "Spring": "Java", "Spring Boot": "Java", "Hibernate": "Java", "JUnit": "Java",
    "Maven": "Java", "React": "JavaScript", "Next.js": "JavaScript", "Redux": "JavaScript",
    "Node.js": "JavaScript", "Express": "JavaScript", "Angular": "TypeScript",
    "Vue.js": "JavaScript", "Jest": "JavaScript", "SwiftUI": "Swift", "XCTest": "Swift",
    "Jetpack Compose": "Kotlin", "Flutter": "Dart",
    # Managed services imply their cloud.
    "AWS SageMaker": "AWS", "AWS Glue": "AWS",
    "ASP.NET": ".NET", "VB.NET": ".NET", "C#": ".NET", "SQL Server": "SQL",
    # Office includes Excel.
    "Microsoft Office": "Excel",
}


def with_implied(skills: list[str]) -> list[str]:
    """``skills`` followed by the languages they imply that are not already listed."""
    result = list(skills)
    for skill in skills:
        implied = IMPLIED_SKILLS.get(skill)
        if implied and implied not in result:
            result.append(implied)
    return result


# --- skills outside the vocabulary ------------------------------------------------------------
#
# The vocabulary cannot list every tool of every profession. A skill it does not know
# ("Windows Server", "GAAP", "QuickBooks") is matched as written: any case, with spaces,
# hyphens and slashes interchangeable ("TCP/IP" = "TCP IP", "month-end" = "month end") and
# an optional plural, while ".", "#" and "+" stay significant (".NET" is not "net").


@lru_cache(maxsize=512)
def term_pattern(term: str) -> re.Pattern:
    """How a skill the vocabulary does not know is found in text."""
    parts = [part for part in re.split(r"[\s\-/]+", term.strip()) if part]
    body = r"[\s\-/]*".join(map(re.escape, parts))
    return re.compile(rf"(?<![\w.#+]){body}(?:e?s)?(?![\w#+])", re.IGNORECASE)


def mentions(text: str, skill: str) -> bool:
    """Whether ``text`` names ``skill``: through the vocabulary when it knows the skill (its
    aliases, or a skill implying it: Django names Python), else as written (``term_pattern``)."""
    canonical = normalize_skill(skill)
    if canonical:
        return canonical in with_implied(find_skills(text))
    return bool(term_pattern(skill).search(text))
