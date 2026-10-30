"""RAG pipeline for resumes: load, chunk by section, extract metadata, embed, store.

Resumes are read with the file system tools in ``fs_tools``, split
into their natural sections (Summary, Skills, Experience, Education, ...) and
chunked so that a chunk never mixes two sections and, where possible, never
splits a single job entry. Each chunk is embedded and stored in a local
vector store (``vector_store``: SQLite and NumPy) together with the resume's
metadata (name, skills, experience years, education), so the job matcher can
filter and rank on it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sys
from collections import defaultdict
from datetime import date
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import openai

from . import fs_tools, personal_data, skills_vocab
from .vector_store import Chunk, Collection, VectorStore

logger = logging.getLogger(__name__)

LLM_MAX_INPUT_CHARS = 12000
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 100
DEFAULT_DB_DIR = "index"
COLLECTION_NAME = "resumes"
# Part of every resume's content hash: bump it when chunking or metadata
# extraction changes so the next `index` run refreshes every resume.
PIPELINE_VERSION = "6"

MAX_CHUNK_CHARS = 1500
HEADER_SECTION = "Header"
GENERAL_SECTION = "General"
# A preamble longer than name + contact lines carries real content worth embedding.
MAX_HEADER_LINES_TO_SKIP = 3

# Canonical section name -> header spellings (compared after normalization).
SECTION_ALIASES = {
    "Summary": [
        "summary", "professional summary", "career summary", "profile",
        "professional profile", "objective", "career objective", "about", "about me",
    ],
    "Skills": [
        "skills", "technical skills", "core skills", "key skills", "core competencies",
        "competencies", "technologies", "tech stack", "skills and technologies",
    ],
    "Experience": [
        "experience", "work experience", "professional experience", "employment",
        "employment history", "work history", "career history", "relevant experience",
    ],
    "Education": ["education", "academic background", "education and training", "qualifications"],
    "Projects": ["projects", "personal projects", "key projects", "selected projects"],
    "Certifications": [
        "certifications", "certificates", "licenses and certifications",
        "certifications and licenses",
    ],
    "Publications": ["publications", "research", "papers"],
    "Awards": ["awards", "honors", "awards and honors", "achievements"],
    "Languages": ["languages"],
}
_ALIAS_TO_SECTION = {alias: name for name, aliases in SECTION_ALIASES.items() for alias in aliases}
# Aliases too common as in-entry labels ("Tech stack: Django") to start a section inline.
_NO_INLINE_ALIASES = {"tech stack", "technologies", "languages", "about", "research", "employment"}

# Sections made of independent entries (one job, one project) that a chunk should keep whole.
ENTRY_SECTIONS = {"Experience", "Projects"}

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DATE = rf"(?:{_MONTH}\s+|\d{{1,2}}/)?(?:19|20)\d{{2}}"
DATE_RANGE_RE = re.compile(
    rf"(?P<start>{_DATE})\s*(?:-|–|—|to)\s*(?P<end>{_DATE}|present|current|now)",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*[-•*▪◦·]\s*")
# "Skills: Python, SQL" - a header with its content on the same line.
_INLINE_HEADER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z &/]{1,40}?)\s*:\s*(\S.*)$")


# --- loading -------------------------------------------------------------------


def file_stamp(path: str) -> str:
    """Size and modification time: unchanged files are recognised without reading them."""
    stat = os.stat(path)
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def read_entry(entry: dict) -> tuple[dict | None, dict | None]:
    """Read one ``fs_tools.list_files`` entry: ``(resume, None)`` or ``(None, skipped)``.

    A resume is ``{"path", "file_name", "content", "stamp"}``; a skipped file
    is ``{"file", "reason"}``. Scanned PDFs are skipped: what little text they
    have (a scanner app's watermark) is not the resume.
    """
    result = fs_tools.read_file(entry["path"])
    if not result["success"]:
        skipped = {"file": entry["name"], "reason": result["error"]}
    elif not result["content"].strip() or result["metadata"].get("scanned"):
        skipped = {"file": entry["name"], "reason": result.get("warning", "no extractable text")}
    else:
        return {
            "path": result["filepath"], "file_name": entry["name"], "content": result["content"],
            "stamp": file_stamp(result["filepath"]),
        }, None
    logger.debug("Skipping %s: %s", skipped["file"], skipped["reason"])  # callers report it
    return None, skipped


def read_folder(directory: str) -> tuple[list[dict], list[dict]]:
    """Read every file in ``directory``: ``(resumes, skipped)`` as from ``read_entry``,
    so callers can tell the user exactly what was left out and why."""
    resumes, skipped = [], []
    for entry in fs_tools.list_files(directory):
        resume, skip = read_entry(entry)
        if resume:
            resumes.append(resume)
        else:
            skipped.append(skip)
    return resumes, skipped


def load_resumes(directory: str) -> list[dict]:
    """The readable resumes in ``directory`` (see ``read_folder`` for what was skipped)."""
    return read_folder(directory)[0]


# --- section splitting -----------------------------------------------------------


def _normalize_header(line: str) -> str:
    text = line.strip().strip("#*=_-:|•").strip().lower()
    text = text.replace("&", "and")
    return re.sub(r"\s+", " ", text)


def match_section_header(line: str) -> tuple[str, str] | None:
    """Return ``(section, inline_content)`` if ``line`` is a section header.

    ``inline_content`` is the text after a colon for one-line sections such as
    ``"Skills: Python, SQL"``, otherwise ``""``.
    """
    if not line.strip() or len(line.strip()) > 60:
        return None
    section = _ALIAS_TO_SECTION.get(_normalize_header(line))
    if section:
        return section, ""
    inline = _INLINE_HEADER_RE.match(line)
    if inline:
        alias = _normalize_header(inline.group(1))
        section = _ALIAS_TO_SECTION.get(alias)
        if section and alias not in _NO_INLINE_ALIASES:
            return section, inline.group(2).strip()
    return None


def split_sections(text: str) -> dict[str, str]:
    """Split resume text into ``{section: text}`` in document order.

    Text before the first recognised header is stored under ``"Header"``
    (usually name and contact line). A section that appears twice is merged.
    If no header is recognised the whole text is returned as ``"General"``.
    """
    sections: dict[str, list[str]] = {HEADER_SECTION: []}
    current = HEADER_SECTION
    found_header = False
    for line in text.splitlines():
        header = match_section_header(line)
        if header:
            current, inline = header
            found_header = True
            sections.setdefault(current, [])
            if inline:
                sections[current].append(inline)
            continue
        sections[current].append(line)

    if not found_header:
        body = text.strip()
        return {GENERAL_SECTION: body} if body else {}

    result = {}
    for name, lines in sections.items():
        body = "\n".join(lines).strip()
        if body:
            result[name] = body
    return result


# --- chunking ----------------------------------------------------------------------


def _is_entry_start(line: str) -> bool:
    """A job/project title line: not a bullet and carries a date range."""
    return bool(line.strip()) and not _BULLET_RE.match(line) and bool(DATE_RANGE_RE.search(line))


def _split_entries(text: str) -> list[str]:
    """Group a section's lines into entries, each starting at a dated title line."""
    entries: list[list[str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if _is_entry_start(line) or not entries:
            entries.append([line])
        else:
            entries[-1].append(line)
    return ["\n".join(e) for e in entries]


def _split_by_lines(text: str, limit: int) -> list[str]:
    """Pack lines into pieces of at most ``limit`` chars; hard-wrap a single overlong line."""
    pieces, current = [], ""
    for line in text.splitlines():
        while len(line) > limit:
            if current:
                pieces.append(current)
                current = ""
            cut = line.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            pieces.append(line[:cut])
            line = line[cut:].lstrip()
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            pieces.append(current)
            current = line
        else:
            current = candidate
    if current.strip():
        pieces.append(current)
    return pieces


def _pack(blocks: list[str], limit: int) -> list[str]:
    """Greedily join blocks into chunks of at most ``limit`` chars.

    A block is only broken apart when it alone exceeds the limit.
    """
    chunks, current = [], ""
    for block in blocks:
        if len(block) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_by_lines(block, limit))
            continue
        candidate = f"{current}\n{block}" if current else block
        if len(candidate) > limit:
            chunks.append(current)
            current = block
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _section_pieces(section: str, body: str, limit: int) -> list[str]:
    if len(body) <= limit:
        return [body]
    blocks = _split_entries(body) if section in ENTRY_SECTIONS else body.splitlines()
    return _pack([b for b in blocks if b.strip()], limit)


def chunk_resume(
    text: str,
    candidate_name: str,
    resume_path: str,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[dict]:
    """Split a resume into section-aware chunks.

    Each section becomes one chunk; a section longer than ``max_chars`` is
    split between job/project entries (Experience, Projects) or lines (others).
    A short header (name + contact) is dropped because it carries no
    matchable content; names live in metadata instead.

    Returns ``[{"id", "section", "chunk_index", "text", "document",
    "candidate_name", "resume_path"}]``, where ``text`` is the raw section
    text (used for excerpts) and ``document`` is what is embedded and
    keyword-searched: a ``"Section: ..."`` prefix for context, then the text
    without the candidate's name, contact details or personal fields
    (``personal_data``), so who they are plays no part in the search.
    """
    file_name = Path(resume_path).name
    chunks = []
    index = 0
    for section, body in split_sections(text).items():
        if section == HEADER_SECTION and len(body.splitlines()) <= MAX_HEADER_LINES_TO_SKIP:
            continue
        for part, piece in enumerate(_section_pieces(section, body, max_chars)):
            chunks.append(
                {
                    "id": f"{file_name}::{section.lower()}::{part}",
                    "section": section,
                    "chunk_index": index,
                    "text": piece,
                    "document": f"Section: {section}\n{_impersonal(piece, candidate_name, section)}",
                    "candidate_name": candidate_name,
                    "resume_path": resume_path,
                }
            )
            index += 1
    return chunks


def _impersonal(text: str, name: str | None, section: str) -> str:
    """``text`` without personal details; in the header, its name, contact and location
    lines are dropped whole."""
    if section == HEADER_SECTION:
        text = "\n".join(line for line in text.splitlines() if not personal_data.is_personal_line(line, name))
    return personal_data.redact(text, name)


# --- metadata extraction --------------------------------------------------------------

DEGREE_LEVELS = {"Associate": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
_DEGREE_PATTERNS = [
    ("PhD", re.compile(r"\b(?:ph\.?\s?d\b|doctor(?:ate)?\b|d\.\s?phil\b)", re.IGNORECASE)),
    (
        "Master's",
        re.compile(
            r"\b(?:master'?s?\b|m\.\s?s\.?|m\.?\s?sc\b|ms\b|m\.\s?a\.|mba\b|m\.?\s?tech\b"
            r"|m\.\s?eng\b|meng\b|m\.\s?e\.)",
            re.IGNORECASE,
        ),
    ),
    (
        "Bachelor's",
        re.compile(
            r"\b(?:bachelor'?s?\b|b\.\s?s\.?|b\.?\s?sc\b|bs\b|b\.\s?a\.|ba\b|b\.?\s?tech\b"
            r"|b\.\s?eng\b|beng\b|b\.\s?e\.|bba\b)",
            re.IGNORECASE,
        ),
    ),
    ("Associate", re.compile(r"\bassociate(?:'s)?\s+(?:degree|of)\b", re.IGNORECASE)),
]

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CONTACT_LINE_RE = re.compile(r"@|https?://|www\.|linkedin|github|\d{3}", re.IGNORECASE)
_NOT_A_NAME = {"resume", "curriculum vitae", "cv", "profile"}
# Words of job titles and headings: a line with one is not a name. Many resumes open with a
# title ("INFORMATION TECHNOLOGY MANAGER") or a heading ("CAREER OVERVIEW") instead of a name.
# Common surnames that are also occupations (Cook, Baker, Mason, Taylor) are left out.
_TITLE_WORDS = {
    "access", "accomplishments", "accountant", "accounting", "administrative", "administrator",
    "adviser", "advisor", "advocate", "agent", "analysis", "analyst", "architect", "assistant",
    "associate", "attorney", "auditor", "base", "business", "career", "chef", "chief", "clerk",
    "consultant", "control", "controller", "coordinator", "core", "craftsman", "customer",
    "data", "designer", "developer", "development", "director", "electrician", "engineer",
    "engineering", "executive", "expert", "finance", "financial", "head", "highlights", "hr",
    "information", "instructor", "intern", "it", "junior", "knowledge", "lead", "leader",
    "management", "manager", "marketing", "mechanic", "monitor", "nurse", "objective", "office",
    "officer", "operations", "operator", "overview", "patient", "pharmacist", "planning",
    "president", "professional", "programmer", "project", "qualifications", "receptionist",
    "recruiter", "rep", "representative", "sales", "scientist", "senior", "service", "software",
    "specialist", "staff", "strategic", "strengths", "summary", "supervisor", "support",
    "systems", "teacher", "team", "tech", "technical", "technician", "technologist",
    "technology", "therapist", "trainer", "training", "vice",
}
_NAME_PARTICLES = {"de", "del", "la", "le", "da", "di", "van", "von", "der", "den", "bin", "al"}
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1
)}
_DATE_PARTS_RE = re.compile(r"(?:(?P<mon>[a-z]+)\.?\s+|(?P<num>\d{1,2})/)?(?P<year>\d{4})")
_TOTAL_YEARS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b[^.\n]{0,40}?\bexperience", re.IGNORECASE
)
# "9 years of Python experience", "7 years of experience in Python".
_YEARS_OF_SKILL_RE = re.compile(
    r"(?P<years>\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b(?:\s+of)?"
    r"(?:\s+(?:professional|hands-on|industry|commercial))?(?:\s+experience)?"
    r"(?:\s+(?:in|with|using))?\s+(?P<rest>[^.;\n]{1,80})",
    re.IGNORECASE,
)
# A skill named in a years claim must appear right after it to be credited.
_YEARS_CLAIM_WINDOW = 25
_LIST_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z /&-]{1,30}:\s*")
MAX_RAW_SKILL_CHARS = 40


def _is_name_word(word: str) -> bool:
    letters = word.replace(".", "").replace("'", "").replace("-", "")
    return letters.isalpha() and (word[0].isupper() or word in _NAME_PARTICLES)


def _extract_name(sections: dict[str, str]) -> str | None:
    source = sections.get(HEADER_SECTION) or sections.get(GENERAL_SECTION, "")
    for line in source.splitlines()[:5]:
        line = line.strip()
        if not line or _CONTACT_LINE_RE.search(line) or line.lower() in _NOT_A_NAME:
            continue
        words = line.split()
        if (2 <= len(words) <= 5 and all(_is_name_word(w) for w in words)
                and not {w.strip(".,").lower() for w in words} & _TITLE_WORDS):
            return line
    return None


def _name_from_path(resume_path: str) -> str | None:
    words = [w for w in re.split(r"[_\-\s]+", Path(resume_path).stem) if w]
    words = [w for w in words if w.lower() not in {"resume", "cv"}]
    return " ".join(w.capitalize() for w in words) or None


def _split_list_items(text: str) -> list[str]:
    """Split a skills list on , ; | • and newlines, but not inside parentheses."""
    items, current, depth = [], [], 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if depth == 0 and ch in ",;|•\n":
            items.append("".join(current))
            current = []
        else:
            current.append(ch)
    items.append("".join(current))
    return [i.strip() for i in items if i.strip()]


def _skills_from_list(text: str) -> list[str]:
    """Normalize the items of a Skills section; unknown items are kept as written."""
    skills = []
    for item in _split_list_items(text):
        item = _LIST_LABEL_RE.sub("", _BULLET_RE.sub("", item)).strip().rstrip(".")
        parts = [item]
        paren = re.match(r"^(?P<main>[^(]+)\((?P<inner>[^)]*)\)", item)
        if paren:  # "AWS (EC2, S3, Lambda)" -> AWS, EC2, S3, Lambda
            parts = [paren["main"].strip(), *re.split(r"[,;/]", paren["inner"])]
        for part in (p.strip() for p in parts):
            if not part:
                continue
            canonical = skills_vocab.normalize_skill(part)
            if canonical:
                skills.append(canonical)
            elif found := skills_vocab.find_skills(part):
                skills.extend(found)
            elif len(part) <= MAX_RAW_SKILL_CHARS:
                skills.append(part)
    return skills


def _dedupe(items: list[str]) -> list[str]:
    seen, result = set(), []
    for item in items:
        if item.lower() not in seen:
            seen.add(item.lower())
            result.append(item)
    return result


def _extract_skills(sections: dict[str, str], candidate_name: str | None) -> list[str]:
    listed = _skills_from_list(sections.get("Skills", ""))
    prose = "\n".join(
        _impersonal(body, candidate_name, section) for section, body in sections.items() if section != "Skills"
    )
    return skills_vocab.with_implied(_dedupe(listed + skills_vocab.find_skills(prose)))


def _month_index(token: str, today: date, is_end: bool) -> int | None:
    """Months since year 0. End dates are exclusive (the month after the last one worked)."""
    token = token.strip().lower()
    if token in {"present", "current", "now"}:
        return today.year * 12 + today.month
    parts = _DATE_PARTS_RE.search(token)
    if not parts:
        return None
    year = int(parts["year"])
    month = _MONTHS.get((parts["mon"] or "")[:3]) or (int(parts["num"]) if parts["num"] else None)
    if month is None or not 1 <= month <= 12:
        return year * 12  # year only: "2018 - 2021" counts as 3 years
    return year * 12 + month - 1 + (1 if is_end else 0)


_ONGOING = {"present", "current", "now"}


@dataclass(frozen=True)
class DatedRole:
    """One role of the Experience section with its dates, in months since year 0."""

    start: int
    end: int  # exclusive; today for a current role
    entry: str  # the role's text; its first line is the heading with the dates
    start_text: str  # the dates as written: "Mar 2020", "2019", "Present"
    end_text: str
    start_month_known: bool  # False for a year alone
    end_month_known: bool

    @property
    def current(self) -> bool:
        return self.end_text.strip().lower() in _ONGOING

    @property
    def title(self) -> str:
        """The heading without its dates: "Backend Developer, Contoso"."""
        return DATE_RANGE_RE.sub("", self.entry.splitlines()[0]).strip(" \t()[]-–—,|")


def _month_known(token: str) -> bool:
    token = token.strip().lower()
    parts = _DATE_PARTS_RE.search(token)
    return token in _ONGOING or bool(parts and (parts["num"] or (parts["mon"] or "")[:3] in _MONTHS))


def dated_roles(experience: str, today: date) -> list[DatedRole]:
    """Every role in ``experience`` whose heading has a date range; undated entries are ignored."""
    roles = []
    now = today.year * 12 + today.month
    for entry in _split_entries(experience):
        dates = DATE_RANGE_RE.search(entry.splitlines()[0])
        if not dates:
            continue
        start = _month_index(dates["start"], today, is_end=False)
        end = _month_index(dates["end"], today, is_end=True)
        if start is None or end is None:
            continue
        end = min(end, now)
        if end > start:
            roles.append(DatedRole(start, end, entry, dates["start"], dates["end"],
                                   _month_known(dates["start"]), _month_known(dates["end"])))
    return roles


def _role_intervals(experience: str, today: date) -> list[tuple[int, int, str]]:
    """``(start, end, entry_text)`` per dated role, in months."""
    return [(role.start, role.end, role.entry) for role in dated_roles(experience, today)]


def _merged_years(intervals: list[tuple[int, int]]) -> float:
    """Total years covered by month intervals, counting overlaps once."""
    total, current_start, current_end = 0, None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return round(total / 12, 1)


def _extract_experience(
    sections: dict[str, str], today: date
) -> tuple[float | None, dict[str, float]]:
    """Return ``(total_years, {skill: years})``.

    Total years come from merged role date ranges, or failing that from a
    "N years of experience" phrase. A skill earns the years of every role whose
    text mentions it, raised to any explicit claim like "9 years of Python".
    """
    experience = sections.get("Experience") or sections.get(GENERAL_SECTION, "")
    roles = _role_intervals(experience, today)

    total = _merged_years([(s, e) for s, e, _ in roles]) if roles else None
    if total is None:
        claim = _TOTAL_YEARS_RE.search("\n".join(sections.values()))
        total = float(claim.group(1)) if claim else None

    per_skill: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for start, end, entry in roles:
        for skill in skills_vocab.with_implied(skills_vocab.find_skills(entry)):
            per_skill[skill].append((start, end))
    skill_years = {skill: _merged_years(spans) for skill, spans in per_skill.items()}

    for claim in _YEARS_OF_SKILL_RE.finditer("\n".join(sections.values())):
        named = skills_vocab.find_skills(claim["rest"][:_YEARS_CLAIM_WINDOW])
        if named:
            skill_years[named[0]] = max(skill_years.get(named[0], 0.0), float(claim["years"]))
    return total, skill_years


def degrees_mentioned(text: str) -> list[str]:
    """Every degree named in ``text``, highest first (keys of ``DEGREE_LEVELS``)."""
    return [degree for degree, pattern in _DEGREE_PATTERNS if pattern.search(text)]


def degree_level(text: str) -> str | None:
    """Highest degree named in ``text``: ``"PhD"``, ``"Master's"``, ``"Bachelor's"``, ``"Associate"``."""
    found = degrees_mentioned(text)
    return found[0] if found else None


def _extract_education(sections: dict[str, str]) -> tuple[list[str], str | None]:
    lines = [_BULLET_RE.sub("", l).strip() for l in sections.get("Education", "").splitlines()]
    lines = _dedupe([l for l in lines if l])
    return lines, degree_level("\n".join(lines))


# --- LLM fallback -------------------------------------------------------------------------

_LLM_FIELD_SCHEMAS = {
    "experience_years": {
        "type": ["number", "null"],
        "description": "Total years of professional work experience.",
    },
    "education": {
        "type": "array",
        "items": {"type": "string"},
        "description": "One entry per degree, e.g. 'B.S. Computer Science, MIT, 2018'.",
    },
    "highest_degree": {"type": ["string", "null"], "enum": [*DEGREE_LEVELS, None]},
}


_LLM_SYSTEM = (
    "Extract the requested fields from the resume. "
    "Use null (or an empty list) when the resume does not state a value."
)

# A chat model asked for JSON: ``(json_schema, system_prompt, resume_text) -> values``.
# It raises ``MetadataLLMError`` when the call fails.
MetadataLLM = Callable[[dict, str, str], dict]


class MetadataLLMError(RuntimeError):
    """The metadata fallback's model call failed (the rule-based values are kept)."""


def metadata_schema(fields: list[str]) -> dict:
    """JSON schema asking for ``fields`` only."""
    properties = {f: _LLM_FIELD_SCHEMAS[f] for f in fields}
    return {
        "title": "resume_metadata",
        "description": "Resume fields the rule-based parser could not find.",
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _apply_llm_values(metadata: dict, values: dict) -> list[str]:
    """Copy valid LLM values into ``metadata`` and return the names of the fields filled."""
    filled = []
    years = values.get("experience_years")
    if isinstance(years, (int, float)) and 0 <= years <= 60:
        metadata["experience_years"] = round(float(years), 1)
        filled.append("experience_years")
    education = values.get("education")
    if isinstance(education, list) and any(isinstance(e, str) and e.strip() for e in education):
        metadata["education"] = [e.strip() for e in education if isinstance(e, str) and e.strip()]
        filled.append("education")
        degree = values.get("highest_degree")
        if degree not in DEGREE_LEVELS:
            degree = degree_level("\n".join(metadata["education"]))
        metadata["highest_degree"] = degree
    return filled


# --- public entry point --------------------------------------------------------------------


def extract_metadata(
    text: str,
    resume_path: str = "",
    llm: MetadataLLM | None = None,
    today: date | None = None,
) -> dict:
    """Extract searchable metadata from resume text.

    Fields are parsed with rules first. If ``experience_years`` or ``education``
    is still missing and ``llm`` is given, one structured-output call fills only
    the missing fields, from the resume without its name and contact details.
    A missing name is never asked for (a model cannot find a name the resume
    does not have, and asking for each anonymous resume made indexing slow): it
    falls back to the file name.

    Returns ``{"name", "email", "skills", "experience_years", "skill_years",
    "education", "highest_degree", "llm_fields"}`` where ``llm_fields`` lists
    the fields the LLM supplied.
    """
    today = today or date.today()
    sections = split_sections(text)
    experience_years, skill_years = _extract_experience(sections, today)
    education, highest_degree = _extract_education(sections)
    email = _EMAIL_RE.search(text)
    name = _extract_name(sections)
    metadata = {
        "name": name,
        "email": email.group(0) if email else None,
        # Never from the name or contact lines: "Ruby Swift" is not a Ruby or Swift skill.
        "skills": _extract_skills(sections, name or (_name_from_path(resume_path) if resume_path else None)),
        "experience_years": experience_years,
        "skill_years": skill_years,
        "education": education,
        "highest_degree": highest_degree,
        "llm_fields": [],
    }

    if not metadata["name"]:
        metadata["name"] = _name_from_path(resume_path) if resume_path else None
    missing = [f for f in ("experience_years", "education") if not metadata[f]]
    if metadata["experience_years"] == 0:
        missing.remove("experience_years")  # parsed, just no experience yet
    if missing and llm is not None:
        fields = missing + (["highest_degree"] if "education" in missing else [])
        impersonal = personal_data.redact(text, metadata["name"])  # dates and degrees need no identity
        try:
            values = llm(metadata_schema(fields), _LLM_SYSTEM, impersonal[:LLM_MAX_INPUT_CHARS])
            metadata["llm_fields"] = _apply_llm_values(metadata, values)
        except MetadataLLMError as exc:
            logger.warning("LLM metadata fallback failed for %s: %s", resume_path or "resume", exc)
    return metadata


# --- embeddings -------------------------------------------------------------------------------


class Embedder(Protocol):
    name: str  # recorded on the collection so queries use the embedder that built it

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Vectors for resume chunks."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """The vector for a search (some models embed queries and documents differently)."""
        ...


class OpenAIEmbedder:
    """OpenAI embeddings (``text-embedding-3-small`` by default), sent in batches."""

    def __init__(self, model: str | None = None, client=None, batch_size: int = EMBED_BATCH_SIZE):
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        self._client = client
        self.batch_size = batch_size
        self.name = f"openai:{self.model}"

    @property
    def client(self):
        # Created on first use, so a missing key is reported by ``Services.check_ready``.
        if self._client is None:
            self._client = openai.OpenAI()
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            response = self.client.embeddings.create(
                model=self.model, input=texts[i : i + self.batch_size]
            )
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


# Instruction some retrieval models were trained with for queries (documents get none).
_RETRIEVAL_QUERY = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class LocalModel:
    """An open-source embedding model run on this machine, and how it expects its input."""

    size_mb: int
    query_prefix: str = ""
    document_prefix: str = ""


# Only models whose scores were measured and calibrated (``job_matcher.SEMANTIC_RANGES``,
# fitted with ``scripts/ranking_eval.py``) are offered: an uncalibrated model would put
# match scores, and so interview recommendations, on a different scale. Of ten models compared on
# the sample resumes, these two ranked as well as OpenAI's text-embedding-3-small.
LOCAL_MODELS = {
    "ibm-granite/granite-embedding-small-english-r2": LocalModel(180),
    "snowflake/snowflake-arctic-embed-m": LocalModel(430, _RETRIEVAL_QUERY),
}
DEFAULT_LOCAL_MODEL = "ibm-granite/granite-embedding-small-english-r2"


class LocalEmbedder:
    """Embeddings computed on this machine with an open-source model (fastembed, ONNX).

    No API key and nothing leaves the machine. The model is downloaded once
    into ``cache_dir`` on first use and loaded lazily, so opening an index
    costs nothing until something is embedded.
    """

    def __init__(self, model: str | None = None, cache_dir: str | None = None, batch_size: int = 32):
        self.model = model or os.getenv("LOCAL_EMBEDDING_MODEL") or DEFAULT_LOCAL_MODEL
        if self.model not in LOCAL_MODELS:
            raise ValueError(
                f"Unsupported local embedding model '{self.model}'. Use one of: {', '.join(LOCAL_MODELS)}."
            )
        self.spec = LOCAL_MODELS[self.model]
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.name = f"local:{self.model}"
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = load_local_model(self.model, self.cache_dir)
        return self._engine

    def _vectors(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.engine.embed(texts, batch_size=self.batch_size)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._vectors([self.spec.document_prefix + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._vectors([self.spec.query_prefix + text])[0]


def load_local_model(model: str, cache_dir: str | None):
    """The fastembed model from ``cache_dir``, downloaded there the first time."""
    try:
        from fastembed import TextEmbedding
        from huggingface_hub.utils import logging as hub_logging
        from loguru import logger as fastembed_logger
    except ImportError as exc:
        raise ValueError("Local embeddings need fastembed: pip install fastembed") from exc
    # Their log lines (an expected cache miss logged as ERROR, rate-limit advice) are noise
    # here: the notice below says what happens, and failures are raised with a fix.
    fastembed_logger.disable("fastembed")
    hub_logging.set_verbosity_error()
    folder = str(Path(cache_dir).expanduser()) if cache_dir else None
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
    try:
        return TextEmbedding(model, cache_dir=folder, local_files_only=True)
    except Exception:  # not downloaded yet (or incomplete): fetch it below
        pass
    size = LOCAL_MODELS[model].size_mb
    print(f"Downloading the embedding model {model} ({size} MB, once)…", file=sys.stderr)
    try:
        return TextEmbedding(model, cache_dir=folder)
    except Exception as exc:  # network, disk or a corrupt download
        raise ValueError(
            f"Could not download the embedding model {model} ({size} MB, from Hugging Face): {exc}. "
            "Check your connection, or set EMBEDDER=openai."
        ) from exc


STOPWORDS = frozenset(
    "a an and are as at be by for from has have in into is it of on or our over the to "
    "using we will with you your experience years year building built strong".split()
)


class HashingEmbedder:
    """Offline bag-of-words embedder: hashed token counts, L2-normalized.

    It only captures word overlap, not meaning. It exists for tests and for
    trying the pipeline without an API key. Stopwords (and filler such as
    "experience") are dropped so shared boilerplate does not dominate.
    """

    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions
        self.name = f"hashing:{dimensions}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9+#]+", text.lower()):
            if token in STOPWORDS:
                continue
            digest = hashlib.md5(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


def embedder_from_name(name: str, cache_dir: str | None = None) -> Embedder:
    """Rebuild an embedder from the ``name`` stored on a collection (``cache_dir``: local models)."""
    kind, _, arg = name.partition(":")
    if kind == "openai":
        return OpenAIEmbedder(model=arg or None)
    if kind == "hashing":
        return HashingEmbedder(int(arg) if arg else 512)
    if kind == "local":
        return LocalEmbedder(model=arg or None, cache_dir=cache_dir)
    raise ValueError(f"Unknown embedder '{name}'")


def default_embedder_name() -> str:
    return f"openai:{os.getenv('OPENAI_EMBEDDING_MODEL') or DEFAULT_EMBEDDING_MODEL}"


# --- vector store ---------------------------------------------------------------------------------


class EmbedderMismatchError(ValueError):
    """The index was built with a different embedder than the one requested."""

    def __init__(self, message: str, stored: str):
        super().__init__(message)
        self.stored = stored


def _content_hash(content: str) -> str:
    return hashlib.sha256(f"{PIPELINE_VERSION}\n{content}".encode("utf-8")).hexdigest()


def _chunk_record(chunk: dict, meta: dict, file_name: str, content_hash: str, file_stamp: str = "") -> dict:
    """The stored record of one chunk: chunk fields plus the resume's metadata.

    ``skill_years`` is stored as a JSON string and empty values are left out,
    which keeps records flat and small.
    """
    record = {
        "candidate_name": chunk["candidate_name"],
        "resume_path": chunk["resume_path"],
        "file_name": file_name,
        "section": chunk["section"],
        "chunk_index": chunk["chunk_index"],
        "content_hash": content_hash,
        "file_stamp": file_stamp,
        "email": meta["email"],
        "skills": meta["skills"],
        "skill_years": json.dumps(meta["skill_years"]),
        "experience_years": meta["experience_years"],
        "education": meta["education"],
        "highest_degree": meta["highest_degree"],
        "degree_level": DEGREE_LEVELS.get(meta["highest_degree"], 0),
        "llm_fields": meta["llm_fields"],
    }
    return {key: value for key, value in record.items() if value not in (None, [], "")}


def parse_record(record: dict) -> dict:
    """Resume-level metadata from a stored chunk record (inverse of ``_chunk_record``)."""
    return {
        "candidate_name": record["candidate_name"],
        "resume_path": record["resume_path"],
        "file_name": record.get("file_name", Path(record["resume_path"]).name),
        "email": record.get("email"),
        "skills": list(record.get("skills", [])),
        "skill_years": json.loads(record.get("skill_years", "{}")),
        "experience_years": record.get("experience_years"),
        "education": list(record.get("education", [])),
        "highest_degree": record.get("highest_degree"),
        "degree_level": record.get("degree_level", 0),
    }


class ResumeRAG:
    """Indexes resumes into a persistent vector store collection (cosine similarity).

    The collection records which embedder built it; opening it with a
    different one raises ``EmbedderMismatchError`` (use ``reset=True`` to
    rebuild). With no ``embedder`` given, the recorded one is reused, so the
    job matcher always embeds queries the same way the resumes were embedded.
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_DB_DIR,
        embedder: Embedder | None = None,
        metadata_llm: MetadataLLM | None = None,
        collection_name: str = COLLECTION_NAME,
        reset: bool = False,
        collection_metadata: dict | None = None,
    ):
        self.persist_dir = persist_dir
        self.metadata_llm = metadata_llm
        self._embedder = embedder
        self._collection_metadata = collection_metadata or {}
        self._store = VectorStore(persist_dir)
        if reset:
            self._store.delete_collection(collection_name)
        self.collection = self._open_collection(collection_name)

    def _open_collection(self, name: str) -> Collection:
        collection = self._store.get_collection(name)
        if collection is None:
            embedder_name = self._embedder.name if self._embedder else default_embedder_name()
            return self._store.create_collection(name, {"embedder": embedder_name, **self._collection_metadata})
        stored = collection.metadata.get("embedder")
        if self._embedder and stored and stored != self._embedder.name:
            raise EmbedderMismatchError(
                f"Index at '{self.persist_dir}' was built with '{stored}', not "
                f"'{self._embedder.name}'. Re-index with --reset to switch embedders.",
                stored,
            )
        return collection

    @property
    def embedder_name(self) -> str:
        return self.collection.metadata.get("embedder", default_embedder_name())

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = embedder_from_name(self.embedder_name)
        return self._embedder

    # --- indexing ---

    def index_directory(self, directory: str, force: bool = False) -> dict:
        """Index every resume in ``directory``; skip unchanged ones unless ``force``.

        A file whose size and modification time match the index is not read
        again; one whose content hash matches is kept and its stamp refreshed.
        Resumes previously indexed from this directory whose files are gone
        (or no longer readable) are removed from the index.

        Returns ``{"indexed", "unchanged", "removed"}`` (lists of resume
        paths), ``"skipped"``
        (``{"file", "reason"}`` per unusable file) and ``"chunks"`` (number of
        chunks written).
        """
        summary = {"indexed": [], "unchanged": [], "removed": [], "chunks": 0, "skipped": []}
        seen = set()
        for entry in fs_tools.list_files(directory):
            path = str(Path(entry["path"]).resolve())
            stored = None if force else self._stored_record(path)
            if stored and stored.get("file_stamp") == file_stamp(path):
                seen.add(path)
                summary["unchanged"].append(path)
                continue
            resume, skipped = read_entry(entry)
            if skipped:
                summary["skipped"].append(skipped)
                continue
            seen.add(resume["path"])
            content_hash = _content_hash(resume["content"])
            if stored and stored.get("content_hash") == content_hash:
                self._set_stamp(resume["path"], resume["stamp"])
                summary["unchanged"].append(resume["path"])
                continue
            summary["chunks"] += self.index_resume(resume, content_hash)
            summary["indexed"].append(resume["path"])

        directory_path = Path(directory).expanduser().resolve()
        for path in sorted(self.indexed_paths() - seen):
            if Path(path).parent == directory_path:
                self.remove_resume(path)
                summary["removed"].append(path)
        return summary

    def index_resume(self, resume: dict, content_hash: str | None = None) -> int:
        """Chunk, extract metadata, embed and store one resume from ``load_resumes``.

        Replaces any chunks already stored for the same path. Returns the
        number of chunks written.
        """
        path, content = resume["path"], resume["content"]
        meta = extract_metadata(content, path, llm=self.metadata_llm)
        chunks = chunk_resume(content, meta["name"], path)
        if not chunks:
            logger.warning("No content to index in %s", path)
            return 0
        # Embed before deleting so a failed API call leaves the old version intact.
        vectors = self.embedder.embed([c["document"] for c in chunks])
        file_name = resume.get("file_name") or Path(path).name
        content_hash = content_hash or _content_hash(content)
        path_key = hashlib.sha1(path.encode("utf-8")).hexdigest()[:8]

        self.remove_resume(path)
        self.collection.add(
            [
                Chunk(f"{c['id']}@{path_key}", c["document"],
                      _chunk_record(c, meta, file_name, content_hash, resume.get("stamp", "")))
                for c in chunks
            ],
            vectors,
        )
        return len(chunks)

    def remove_resume(self, resume_path: str) -> None:
        self.collection.delete(resume_path)

    def _stored_record(self, resume_path: str) -> dict | None:
        """One stored chunk record of ``resume_path`` (they share the resume-level fields)."""
        found = self.collection.chunks(resume_path, limit=1)
        return found[0].metadata if found else None

    def _set_stamp(self, resume_path: str, stamp: str) -> None:
        self.collection.update_metadata(resume_path, {"file_stamp": stamp})

    # --- reading ---

    def _all_records(self) -> list[dict]:
        return [chunk.metadata for chunk in self.collection.chunks()]

    def all_chunks(self) -> list[dict]:
        """Every stored chunk as ``{"id", "document", "metadata"}`` (for keyword search)."""
        return [
            {"id": chunk.id, "document": chunk.document, "metadata": chunk.metadata}
            for chunk in self.collection.chunks()
        ]

    def indexed_paths(self) -> set[str]:
        return {record["resume_path"] for record in self._all_records()}

    def list_resumes(self) -> list[dict]:
        """One metadata dict per indexed resume, sorted by candidate name."""
        resumes = {}
        for record in self._all_records():
            resumes.setdefault(record["resume_path"], parse_record(record))
        return sorted(resumes.values(), key=lambda r: r["candidate_name"].lower())

    def stats(self) -> dict:
        return {
            "persist_dir": self.persist_dir,
            "collection": self.collection.name,
            "embedder": self.embedder_name,
            "resumes": len(self.indexed_paths()),
            "chunks": self.collection.count(),
        }


# --- CLI --------------------------------------------------------------------------------------------


def display_path(path: str) -> str:
    """``path`` relative to the working directory when inside it, else ``~/...`` under
    the home directory, else as given."""
    for base, prefix in ((Path.cwd(), ""), (Path.home(), "~/")):
        try:
            return prefix + str(Path(path).relative_to(base))
        except ValueError:
            continue
    return path


def missing_api_key(embedder_name: str) -> bool:
    """True when ``embedder_name`` needs OpenAI but no ``OPENAI_API_KEY`` is set."""
    return embedder_name.startswith("openai") and not os.getenv("OPENAI_API_KEY")
