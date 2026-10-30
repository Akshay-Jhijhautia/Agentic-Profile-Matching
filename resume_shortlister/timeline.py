"""Gaps between roles and short completed roles, read from a resume's dates.

They are shown to the recruiter as things worth asking about and never scored: they have
many legitimate causes (caregiving, illness, study, layoffs, contracts), and penalising
them can disadvantage people for reasons unrelated to the role. Code computes them, so
every run says the same.
"""

from __future__ import annotations

import re
from datetime import date

from .rag import resume_rag

# A shorter pause between roles is ordinary job changing.
MIN_GAP_MONTHS = 6
# A completed role shorter than this is noted, unless it is short by nature.
SHORT_ROLE_MONTHS = 12
_SHORT_BY_NATURE_RE = re.compile(
    r"\b(?:intern(?:ship)?|contract(?:or)?|freelance|temporary|temp|seasonal|part[- ]time|placement)\b", re.IGNORECASE
)


def _duration(months: int) -> str:
    years, rest = divmod(months, 12)
    parts = [f"{years} year{'s' * (years != 1)}"] * bool(years) + [f"{rest} month{'s' * (rest != 1)}"] * bool(rest)
    return " ".join(parts)


def notes(resume_text: str, today: date) -> list[str]:
    """The resume's gaps of ``MIN_GAP_MONTHS`` or more, then its completed roles shorter than
    ``SHORT_ROLE_MONTHS``. Dates given as a year alone count as the whole year, so a role
    ending "2022" and the next starting "2023" leave no gap."""
    sections = resume_rag.split_sections(resume_text)
    experience = sections.get("Experience") or sections.get(resume_rag.GENERAL_SECTION, "")
    roles = sorted(resume_rag.dated_roles(experience, today), key=lambda role: role.start)
    found = []
    covered_until, latest = None, None  # the end of the roles so far, and the role reaching it
    for role in roles:
        if latest is not None and role.start - covered_until >= MIN_GAP_MONTHS:
            found.append(
                f"{_duration(role.start - covered_until)} between {latest.title} (until {latest.end_text}) "
                f"and {role.title} (from {role.start_text})"
            )
        end = role.end if role.end_month_known else role.end + 12  # "2022" may run to its end
        if covered_until is None or end > covered_until:
            covered_until, latest = end, role
    for role in roles:
        length = role.end - role.start
        if (not role.current and role.start_month_known and role.end_month_known and length < SHORT_ROLE_MONTHS
                and not _SHORT_BY_NATURE_RE.search(role.title)):
            found.append(f"{role.title}: {_duration(length)} ({role.start_text} to {role.end_text})")
    return found
