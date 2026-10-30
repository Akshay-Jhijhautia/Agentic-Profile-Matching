"""Turn a job description or a natural-language request into ``Requirements``.

The model proposes the requirements; rules then decide wherever the text is
explicit (``apply_rules``), so the same job description gives the same
knockouts whichever model, or run, read it:

- Skills. The skills vocabulary knows common skills and their aliases; any
  other skill the model proposes counts when the text names it, spelled as the
  text does (``jd_terms``: "Windows Server", "GAAP", "QuickBooks"), so a role
  of any profession gets its own requirements without a word list.
- Must-haves. A skill is a must-have only where the text demands it: not in a
  "preferred" section or a softened phrase ("familiarity with", "a plus"), not
  as one of several options ("Oracle or PostgreSQL"), not only outside the
  requirements section of a JD that has one, and only if resumes can be
  checked for it: a vocabulary skill, or a term the text writes as a name
  (``is_named``: "Active Directory", "TCP/IP"), never a concept resumes
  paraphrase ("relational databases", "month-end close"). A requirement naming
  exactly one skill makes that skill a must-have. Only requirements naming
  several skills ("deploying models with Docker and Kubernetes on AWS") are
  left to the model.
- Years come from the text: a years phrase applies to the must-have skills it
  names ("3+ years with React and Node.js": both), else to the one must-have
  skill of its requirement ("React and 3+ years"), else to total experience
  ("6+ years of software development"). The model's years count only when the
  text states none in figures.
- A degree is required only when a requirement names it without "or
  equivalent", "preferred" and the like; the lowest degree it names counts.
- Nice-to-haves are every other skill and term the text names, concepts
  included.

What rules cannot settle, the model's choice on a requirement naming several
skills, is listed by ``open_questions`` so a recruiter can confirm it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from . import llm, prompts
from .llm_cache import ResponseCache
from .rag import fs_tools, job_matcher, skills_vocab
from .schemas import Requirements, SkillRequirement, is_skill

# Clause boundaries for spotting alternatives: "PyTorch or TensorFlow and
# scikit-learn" offers a choice between the first two only.
_CLAUSE_SPLIT_RE = re.compile(r"[\n,;()]|\band\b", re.IGNORECASE)
_OR_RE = re.compile(r"\bor\b", re.IGNORECASE)
# Softening that keeps a degree from being required.
_SOFTENED_RE = re.compile(
    r"\b(?:equivalent|preferred|a plus|nice to have|bonus|desirable|ideally)\b", re.IGNORECASE
)
# Wording that makes a skill optional, in a heading or in the requirement itself.
_OPTIONAL_RE = re.compile(
    r"\b(?:nice[- ]to[- ]haves?|preferred|bonus|a plus|desirable|good[- ]to[- ]haves?|optional|ideally|"
    r"familiar(?:ity)?|exposure to|would be (?:great|nice)|helpful|an advantage)\b",
    re.IGNORECASE,
)
_REQUIRED_HEADING_RE = re.compile(
    r"\b(?:requirements?|required|qualifications|must[- ]haves?|what you'?ll need|who you are|"
    r"what we'?re looking for|you have|skills)\b",
    re.IGNORECASE,
)
# A line's requirements: its ";"-separated parts and its sentences.
_REQUIREMENT_SPLIT_RE = re.compile(r";|(?<=[.!?])\s+(?=[A-Z])")
# Where the words of a years phrase end ("5 years of experience operating X, and you ..."):
# a full stop ends it, the dot in "Node.js" does not.
_PHRASE_END_RE = re.compile(r"[,;:]|\.(?=\s|$)|\band you\b", re.IGNORECASE)
# File paths and names in a chat message ("screen ~/Downloads/react-team/jane.pdf"): never
# skills. "CI/CD" is not a path: a path starts with "/", "~/", "./" or a drive letter.
_PATH_RE = re.compile(
    r"(?<![\w/])(?:~|\.{1,2}|[A-Za-z]:)?[/\\][^\s,;]+|\b[\w.-]+\.(?:pdf|docx?|txt|md|rtf|odt)\b", re.IGNORECASE
)
# The longest single line read as a request ("React and 3+ years experience"), in words.
REQUEST_WORDS = 30


@dataclass(frozen=True)
class Clause:
    """One requirement of the text and where it stands."""

    text: str
    # "required", "preferred", "other" (another section) or "untitled" (before any heading).
    section: str
    skills: tuple[str, ...]
    optional: bool
    alternatives: frozenset[str]  # lower-cased skills offered as one option among several
    concepts: frozenset[str] = frozenset()  # lower-cased terms resumes paraphrase (see ``is_named``)

    @property
    def demands(self) -> list[str]:
        """The skills this requirement could make must-haves."""
        if self.optional:
            return []
        return [s for s in self.skills if s.lower() not in self.alternatives | self.concepts
                and s not in skills_vocab.NICE_ONLY]


def _logical_lines(text: str) -> list[str]:
    """Lines as written, with wrapped prose joined back into its paragraph (bullets and
    headings always start a new line)."""
    lines: list[str] = []
    joinable = False
    for line in text.splitlines():
        starts_new = not line.strip() or job_matcher.is_jd_header(line) or bool(job_matcher.BULLET_RE.match(line))
        if lines and joinable and not starts_new:
            lines[-1] = f"{lines[-1]} {line.strip()}"
        else:
            lines.append(line)
        joinable = bool(line.strip()) and not job_matcher.is_jd_header(line)
    return lines


def skills_in(text: str, terms: list[str] = ()) -> list[str]:
    """The vocabulary skills ``text`` names, then the ``terms`` it names, in text order."""
    located = sorted((m.start(), term) for term in terms if (m := skills_vocab.term_pattern(term).search(text)))
    return skills_vocab.find_skills(text) + [term for _, term in located]


def clauses(text: str, terms: list[str] = ()) -> list[Clause]:
    """The text's requirements, each with its section, skills (vocabulary skills and the
    ``terms``) and softening."""
    concepts = frozenset(t.lower() for t in terms if not is_named(t, text))
    section, found = "untitled", []
    for line in _logical_lines(_PATH_RE.sub(" ", text)):
        if job_matcher.is_jd_header(line):
            section = ("preferred" if _OPTIONAL_RE.search(line)
                       else "required" if _REQUIRED_HEADING_RE.search(line) else "other")
            continue
        for part in _REQUIREMENT_SPLIT_RE.split(line):
            if part.strip():
                found.append(Clause(
                    part.strip(), section, tuple(skills_in(part, terms)),
                    section == "preferred" or bool(_OPTIONAL_RE.search(part)),
                    frozenset(alternative_skills(part, terms)), concepts,
                ))
    return found


def is_request(text: str) -> bool:
    """A one-line request such as "Find me React developers with 3+ years"."""
    stripped = text.strip()
    return "\n" not in stripped and len(stripped.split()) <= REQUEST_WORDS


def demanding(found: list[Clause]) -> list[Clause]:
    """The clauses that can make must-haves: the requirements section when the text has
    one (not its title, intro or "about" text), else the whole text."""
    section = "required" if any(c.section == "required" for c in found) else "untitled"
    return [c for c in found if c.section == section and not c.optional]


def alternative_skills(text: str, terms: list[str] = ()) -> set[str]:
    """Lower-cased skills (and ``terms``) offered as one option among others ("Oracle or
    PostgreSQL", "Snowflake or BigQuery", "Kubernetes or similar"), whether or not the
    others are skills the vocabulary knows."""
    found = set()
    for clause in _CLAUSE_SPLIT_RE.split(text):
        if _OR_RE.search(clause):
            found.update(skill.lower() for skill in skills_in(clause, terms))
    return found


def degree_required(rules: dict) -> bool:
    """True when a requirement line (per the rule parser) names a degree without softening it."""
    return any(
        req["type"] == "degree" and not _SOFTENED_RE.search(req["source"])
        for req in rules["must_haves"]
    )


def required_degree(rules: dict) -> str | None:
    """The lowest degree an unsoftened requirement line names, if any."""
    degrees = [req["degree"] for req in rules["must_haves"]
               if req["type"] == "degree" and not _SOFTENED_RE.search(req["source"])]
    return min(degrees, key=lambda d: job_matcher.resume_rag.DEGREE_LEVELS[d]) if degrees else None


def must_have_skills(model_must: list[str], found: list[Clause], request: bool) -> list[str]:
    """The model's must-haves the text allows, plus every skill a requirement names alone."""
    clauses_that_demand = demanding(found)
    allowed = {s.lower() for c in clauses_that_demand for s in c.demands}
    chosen = {s.lower() for s in model_must if s.lower() in allowed}
    # A requirement naming one skill demands it (in a prose JD, a lone skill may only be
    # mentioned, so there the model decides).
    if request or any(c.section == "required" for c in found):
        chosen |= {c.demands[0].lower() for c in clauses_that_demand if len(c.demands) == 1}
    return [s for s in _in_text_order(found) if s.lower() in chosen]


def stated_years(
    found: list[Clause], must: list[str], terms: list[str] = ()
) -> tuple[dict[str, float], float | None, bool]:
    """Years per must-have skill and total years, as the text states them (see module
    docstring); the last value says whether the text states any years at all."""
    must_lower = {s.lower(): s for s in must}
    concepts = found[0].concepts if found else frozenset()
    per_skill: dict[str, float] = {}
    total, stated = None, False
    for clause in demanding(found):
        for match in job_matcher.YEARS_RE.finditer(clause.text):
            stated, years = True, float(match["years"])
            phrase = _PHRASE_END_RE.split(clause.text[match.start("rest"):])[0]
            # A concept ("5+ years of accounting experience") names no skill: the years are total.
            named = [s for s in skills_in(phrase, terms) if s.lower() not in concepts]
            own = [s for s in clause.skills if s.lower() in must_lower]
            if named:
                subjects = [s for s in named if s.lower() in must_lower]
            elif not phrase.strip() and len(own) == 1:
                subjects = own
            else:
                subjects = []
                total = max(total or 0.0, years)
            for skill in subjects:
                key = must_lower[skill.lower()]
                per_skill[key] = max(per_skill.get(key, 0.0), years)
    return per_skill, total, stated


def jd_terms(items: list[str], text: str) -> list[str]:
    """The skills among ``items`` that the vocabulary does not know and the text names, spelled
    as the text does (``as_stated``), once each."""
    terms: dict[str, str] = {}
    for item in items:
        if skills_vocab.normalize_skill(item) or skills_vocab.find_skills(item) or not is_skill(item):
            continue
        stated = as_stated(item, text)
        if stated:
            terms.setdefault(stated.lower(), stated)
    return list(terms.values())


def is_named(term: str, text: str) -> bool:
    """Whether the text writes ``term`` as a name (a tool, product, standard or certificate:
    "Windows Server", "GAAP", "TCP/IP", ".NET") rather than a concept ("relational databases",
    "month-end close", "accounting"): always with a capital past the start of a sentence, a
    digit or a symbol, never in plain lower case. Resumes name a tool the same way but
    paraphrase a concept, so only a name is checked as a must-have; a concept is left to the
    deep review."""
    named = False
    for match in skills_vocab.term_pattern(term).finditer(text):
        written = match[0]
        if written.islower() and not re.search(r"[\d.#+/&]", written):
            return False  # "accounting experience": a common word, however a title writes it
        before = text[:match.start()].rsplit("\n", 1)[-1]
        sentence_start = not job_matcher.BULLET_RE.sub("", before).strip() or re.search(r"[.:!?]\s*$", before)
        named = named or bool(re.search(r"[\d.#+/&]", written) or any(ch.isupper() for ch in written[1:])
                              or (written[0].isupper() and not sentence_start))
    return named


# Words around a skill that describe how well or in what form, not which skill: a model adds
# them ("Microsoft Excel", "Excel skills") where resumes write just the skill ("Excel").
_QUALIFIERS = {"microsoft", "ms", "advanced", "strong", "solid", "basic", "proficient", "expert", "excellent",
               "hands-on"}
_GENERIC_ENDINGS = {"skills", "skill", "experience", "knowledge", "proficiency", "expertise", "certification",
                    "certifications"}


def as_stated(preference: str, text: str) -> str | None:
    """``preference`` as the job description writes it ("financial services"), capitalised;
    None when the text does not say it. Models phrase the same preference differently ("Cloud
    Data Warehousing" for "a cloud data warehouse"); only the text's own words give every model
    the same requirement. Spacing, hyphens and a plural may differ, and the words around a
    skill that only qualify it are left out ("Advanced Excel skills" is "Excel")."""
    words = preference.strip().split()
    # A qualifier only goes when a name remains: "Microsoft 365" stays whole.
    while len(words) > 1 and words[0].lower() in _QUALIFIERS and re.search(r"[A-Za-z]", " ".join(words[1:])):
        words = words[1:]
    while len(words) > 1 and words[-1].lower() in _GENERIC_ENDINGS:
        words = words[:-1]
    found = skills_vocab.term_pattern(" ".join(words)).search(text) if words else None
    if not found:
        return None
    stated = " ".join(found[0].split())
    return stated[0].upper() + stated[1:]


def apply_rules(proposed: Requirements, text: str) -> Requirements:
    """The model's requirements with every explicit part of the text decided by rules."""
    terms = jd_terms([*proposed.must_have_skills, *proposed.nice_to_have], text)
    found = clauses(text, terms)
    spelled = {t.lower(): t for t in terms}
    model_must = [spelled.get((as_stated(s, text) or s).lower(), s) for s in proposed.must_have_skills]
    must = must_have_skills(model_must, found, is_request(text))
    per_skill, total, stated = stated_years(found, must, terms)
    if not stated:  # years written in words, or none: the model's reading is all there is
        model_years = {r.skill.lower(): r.min_years for r in proposed.must_have}
        per_skill = {s: model_years[s.lower()] for s in must if model_years.get(s.lower())}
        total = proposed.min_total_years
    must_lower = {s.lower() for s in must}
    in_clauses = {s.lower() for s in _in_text_order(found)}
    elsewhere = [t for t in terms if t.lower() not in in_clauses | must_lower]  # e.g. only in a heading
    return Requirements.model_validate({
        "role_title": proposed.role_title,
        "must_have": [SkillRequirement(skill=s, min_years=per_skill.get(s)).model_dump() for s in must],
        "nice_to_have": [s for s in _in_text_order(found) if s.lower() not in must_lower] + elsewhere,
        "min_total_years": total,
        "min_degree": required_degree(job_matcher.parse_job_description(text)),
    })


@dataclass(frozen=True)
class OpenQuestion:
    """A requirement naming several skills where the model chose which are must-haves."""

    requirement: str
    must: tuple[str, ...]
    nice: tuple[str, ...]

    def line(self) -> str:
        chosen = f"**{', '.join(self.must)}** must-have, " if self.must else ""
        return f'"{self.requirement}": {chosen}**{", ".join(self.nice)}** only nice-to-have'


def open_questions(text: str, requirements: Requirements) -> list[OpenQuestion]:
    """The requirements of ``text`` naming several skills that did not all become must-haves:
    the only must-have decisions the model made on its own (see ``apply_rules``)."""
    must = {s.lower() for s in requirements.must_have_skills}
    terms = jd_terms([*requirements.must_have_skills, *requirements.nice_to_have], text)
    questions: dict[str, OpenQuestion] = {}
    for clause in demanding(clauses(text, terms)):
        demands = clause.demands
        chosen = [s for s in demands if s.lower() in must]
        if len(demands) > 1 and len(chosen) < len(demands):
            requirement = job_matcher.BULLET_RE.sub("", clause.text).strip().rstrip(".")
            questions.setdefault(requirement, OpenQuestion(
                requirement, tuple(chosen), tuple(s for s in demands if s.lower() not in must)
            ))
    return list(questions.values())


def _in_text_order(found: list[Clause]) -> list[str]:
    seen: dict[str, str] = {}
    for skill in (skill for clause in found for skill in clause.skills):
        seen.setdefault(skill.lower(), skill)
    return list(seen.values())


def load_job_description(text_or_path: str) -> str:
    """The text itself, or the contents when it names a readable JD file."""
    candidate = text_or_path.strip()
    if "\n" not in candidate and Path(candidate).suffix.lower() in fs_tools.SUPPORTED_READ_EXTENSIONS:
        result = fs_tools.read_file(candidate)
        if not result["success"]:
            raise FileNotFoundError(f"Could not read job description: {result['error']}")
        return result["content"]
    return text_or_path


def extract_requirements(
    text: str, *, model: BaseChatModel | None = None, cache: ResponseCache | None = None
) -> Requirements:
    """Must-have and nice-to-have requirements for a JD or a short request."""
    if not text or not text.strip():
        raise ValueError("Job description must be a non-empty string")
    proposed = llm.structured(Requirements, prompts.EXTRACT_REQUIREMENTS, text.strip(), model=model, cache=cache)
    return apply_rules(proposed, text)
