"""Explainability for one candidate: strengths and gaps, how the score adds up,
the detailed match report, and saving a screening's reports to disk.

Everything is rendered from results already computed by the screening rounds,
so a report can never disagree with the ranking.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .markdown import join, or_dash, table
from .rag import fs_tools
from .schemas import COMPONENT_LABELS, DEEP_WEIGHT, CandidateMatch, Requirements

CANDIDATE_DIR = "candidates"
SUMMARY_FILE = "summary.md"


# --- one-line cells ---------------------------------------------------------------------------------


def must_have_cell(match: CandidateMatch) -> str:
    if match.failed_requirements:
        return f"✗ fails {len(match.failed_requirements)}"
    if match.unverified:
        return f"⚠ {len(match.unverified)} unverified"
    return "✓ all met" if match.must_have_checks else "–"


def decision_cell(match: CandidateMatch) -> str:
    """"✅ hire (high)", "❌ no hire (low) · borderline", or "–" before round 3."""
    rec = match.recommendation
    text = "–" if rec is None else f"{'✅ hire' if rec.decision == 'hire' else '❌ no hire'} ({rec.confidence})"
    return f"{text} · borderline" if match.borderline else text


def recommendation_line(match: CandidateMatch) -> str | None:
    rec = match.recommendation
    if rec is None:
        return None
    verdict = "Hire" if rec.decision == "hire" else "No hire"
    return f"**Recommendation:** {verdict} ({rec.confidence} confidence). {rec.rationale}"


# --- strengths and gaps ----------------------------------------------------------------------------


def strengths_and_gaps(match: CandidateMatch, requirements: Requirements) -> tuple[list[str], list[str]]:
    """From the deep analysis when it ran, otherwise from the round-1 checks."""
    if match.assessment is not None:
        return list(match.assessment.strengths), list(match.assessment.gaps)
    must = {skill.lower() for skill in requirements.must_have_skills}
    matched = {skill.lower() for skill in match.matched_skills}
    strengths = [f"Meets {c.detail}" for c in match.must_have_checks if c.status == "met"]
    strengths += [f"Has nice-to-have {s}" for s in match.matched_skills if s.lower() not in must]
    gaps = [f"Fails {reason}" for reason in match.failed_requirements]
    gaps += [f"Unverified: {detail}" for detail in match.unverified]
    gaps += [f"No {s} (nice to have)" for s in requirements.nice_to_have if s.lower() not in matched]
    return strengths, gaps


# --- score breakdown ---------------------------------------------------------------------------------


def score_rows(match: CandidateMatch) -> list[list[object]]:
    """How the final score adds up: component, value, weight, points."""
    retrieval_share = 1 - DEEP_WEIGHT if match.assessment else 1
    weights = {name: retrieval_share * w for name, w in match.retrieval_weights().items()}
    rows = []
    for name, points in match.contributions().items():
        if name == "deep":
            value, weight = match.assessment.deep_score, DEEP_WEIGHT
        else:
            value, weight = f"{getattr(match.score_breakdown, name):.0%}", weights[name]
        rows.append([COMPONENT_LABELS[name].capitalize(), value, f"{weight:.2f}", f"{points:.1f}"])
    rows.append(["**Final score**", "", "", f"**{match.final_score}**"])
    return rows


# --- reports ----------------------------------------------------------------------------------------


def summary_section(match: CandidateMatch, requirements: Requirements) -> list[str]:
    """The candidate's block in the shortlist report: decision, analysis, strengths, gaps."""
    lines = ["", f"## {match.rank}. {match.name} ({match.final_score})", ""]
    if (line := recommendation_line(match)) is not None:
        lines += [line, ""]
    if match.assessment is not None:
        lines += [f"**Deep analysis ({match.assessment.deep_score}):** {match.assessment.summary}", ""]
    strengths, gaps = strengths_and_gaps(match, requirements)
    lines += [f"**Strengths:** {join(strengths[:3])}", "", f"**Gaps:** {join(gaps[:3])}", ""]
    if match.improvements:
        lines += ["**To become a clear hire:**", ""]
        lines += [f"- {i.suggestion} (verify: {i.how_to_verify})" for i in match.improvements]
        lines.append("")
    lines.append(f"**Retrieval ({match.match_score}):** {match.reasoning}")
    if match.excerpts:
        lines += ["", f"> {match.excerpts[0]}"]
    return lines


def candidate_report(match: CandidateMatch, requirements: Requirements) -> str:
    """The detailed match report for one candidate."""
    lines = [f"# {match.name}: {requirements.role_title}", ""]
    status = f"Rank {match.rank} · final score {match.final_score} · {decision_cell(match)}"
    lines += [f"**{status}**", ""]
    if (line := recommendation_line(match)) is not None:
        lines += [line, ""]
    if match.failed_requirements:
        lines += [f"**Knocked out:** fails {join(match.failed_requirements)}", ""]
    years = f"{match.experience_years:g} years" if match.experience_years is not None else "unknown"
    lines += [f"Resume `{match.resume}` · experience {years} · degree {or_dash(match.highest_degree)}", ""]
    if match.assessment is not None:
        lines += [match.assessment.summary, ""]

    lines += ["## How the score adds up", "", *table(["Component", "Value", "Weight", "Points"], score_rows(match))]
    if match.must_have_checks:
        rows = [[c.requirement, c.status.replace("_", " "), c.detail] for c in match.must_have_checks]
        lines += ["", "## Must-have checks", "", *table(["Requirement", "Status", "Detail"], rows)]
    if match.assessment is not None:
        rows = [[e.requirement, e.verdict, e.evidence or "–"] for e in match.assessment.requirement_evidence]
        lines += ["", "## Evidence per requirement", "", *table(["Requirement", "Verdict", "Evidence"], rows)]

    strengths, gaps = strengths_and_gaps(match, requirements)
    for title, items in (("Strengths", strengths), ("Gaps", gaps)):
        lines += ["", f"## {title}", "", *([f"- {item}" for item in items] or ["- none found"])]
    if match.assessment is not None and match.assessment.risk_flags:
        lines += ["", "## Risk flags", "", *[f"- {flag}" for flag in match.assessment.risk_flags]]
    if match.improvements:
        rows = [[i.gap, i.suggestion, i.how_to_verify] for i in match.improvements]
        lines += ["", "## Improvement suggestions", "", *table(["Gap", "Suggestion", "How to verify"], rows)]

    lines += ["", "## Retrieval reasoning", "", match.reasoning]
    lines += [f"\n> {excerpt}" for excerpt in match.excerpts]
    return "\n".join(lines) + "\n"


# --- saving ------------------------------------------------------------------------------------------


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "screening"


def _write(path: Path, content: str) -> None:
    result = fs_tools.write_file(str(path), content)
    if not result["success"]:
        raise OSError(result["error"])


def save_reports(
    reports_dir: str, requirements: Requirements, shortlist: list[CandidateMatch], summary: str
) -> Path:
    """Write ``summary.md`` and one detailed report per shortlisted candidate into a new
    timestamped folder, so every screening (and refinement) keeps its own record."""
    base = Path(reports_dir) / f"{_slug(requirements.role_title)}-{datetime.now():%Y%m%d-%H%M%S}"
    folder, suffix = base, 2
    while folder.exists():
        folder, suffix = base.with_name(f"{base.name}-{suffix}"), suffix + 1
    _write(folder / SUMMARY_FILE, summary)
    for match in shortlist:
        _write(folder / CANDIDATE_DIR / f"{match.rank:02d}-{match.id}.md", candidate_report(match, requirements))
    return folder
