"""Personal details in resume text, found and replaced, so who a candidate is plays no
part in judging them.

Used for blind screening (the model never sees a candidate's name, contact details or
personal fields) and in indexing, so a name is neither embedded nor read as a skill
("Ruby Swift" knows neither Ruby nor Swift unless her resume says so).

Replaced: the candidate's name, emails, phone numbers, profile and website links, and
the values of personal fields (date of birth, age, gender, marital status, nationality,
citizenship, religion, address). Not replaced: school names, dates (graduation years
included) and locations inside the experience, which are evidence as much as they are
personal; ``is_personal_line`` lets the header's name, contact and location lines be
dropped whole.
"""

from __future__ import annotations

import re

from . import skills_vocab

NAME = "[candidate]"
EMAIL = "[email]"
PHONE = "[phone]"
LINK = "[link]"
REMOVED = "[removed]"

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# A web address with a scheme or www., or a profile path on a site people link to.
LINK_RE = re.compile(
    r"\b(?:https?://|www\.)[^\s|,;()]+"
    r"|\b(?:[\w-]+\.)*(?:linkedin|github|gitlab|bitbucket|behance|dribbble|medium|kaggle|stackoverflow)"
    r"\.(?:com|io|org)/[^\s|,;()]*",
    re.IGNORECASE,
)
# Phone-like: digits with spaces, dots, dashes or parentheses (counted in ``_phone``).
_PHONE_CANDIDATE_RE = re.compile(r"(?<![\w/])\+?\(?\d[\d\s().-]{6,}\d(?![\w/])")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
PERSONAL_FIELD_RE = re.compile(
    r"^(?P<label>[ \t]*[-•*]?[ \t]*(?:date of birth|d\.?o\.?b\.?|born|age|gender|sex|marital status"
    r"|nationality|citizenship|religion|address|photo)[ \t]*[:\-–][ \t]*)\S.*$",
    re.IGNORECASE | re.MULTILINE,
)
# "Chicago, IL", "London, United Kingdom": a header line that is only a place.
_LOCATION_RE = re.compile(r"^[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3},\s*[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3}$")
_MONTHS = {
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july", "august", "september", "october",
    "november", "december",
}


def _phone(match: re.Match) -> str:
    """A phone number has 9 to 15 digits and is not a date range ("01.2019 - 12.2023")."""
    text = match.group(0)
    digits = sum(ch.isdigit() for ch in text)
    if not 9 <= digits <= 15 or len(_YEAR_RE.findall(text)) >= 2:
        return text
    return PHONE


def _name_patterns(name: str) -> list[re.Pattern]:
    """The full name in any case, then each part of it as written (or in capitals).

    A part that is also a skill ("Swift", "Ruby", "Go") or a month ("May") is only
    replaced inside the full name, so "built apps in Swift" keeps its evidence.
    """
    parts = [p for p in re.split(r"[\s,]+", name.strip()) if p]
    if not parts:
        return []
    full = re.compile(r"\b" + r"\s+".join(map(re.escape, parts)) + r"\b", re.IGNORECASE)
    own = [
        p for p in parts
        if len(p.strip(".")) >= 2 and p.lower() not in _MONTHS and not skills_vocab.normalize_skill(p)
    ]
    variants = sorted({v for p in own for v in (p, p.upper())}, key=len, reverse=True)
    return [full] + ([re.compile(r"\b(?:" + "|".join(map(re.escape, variants)) + r")\b")] if variants else [])


def redact(text: str, name: str | None) -> str:
    """``text`` with the candidate's name, contact details and personal fields replaced."""
    text = EMAIL_RE.sub(EMAIL, text)
    text = LINK_RE.sub(LINK, text)
    text = _PHONE_CANDIDATE_RE.sub(_phone, text)
    text = PERSONAL_FIELD_RE.sub(lambda m: m["label"] + REMOVED, text)
    for pattern in _name_patterns(name or ""):
        text = pattern.sub(NAME, text)
    return text


def is_personal_line(line: str, name: str | None) -> bool:
    """A header line that only identifies the candidate: their name, a line with contact
    details (which usually carries the location too), or a place on its own."""
    stripped = line.strip()
    if not stripped:
        return False
    if name and stripped.lower() == " ".join(name.split()).lower():
        return True
    if EMAIL_RE.search(stripped) or LINK_RE.search(stripped) or _phone_in(stripped):
        return True
    return bool(_LOCATION_RE.match(stripped))


def _phone_in(text: str) -> bool:
    return any(_phone(m) == PHONE for m in _PHONE_CANDIDATE_RE.finditer(text))
