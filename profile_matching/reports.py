"""Cohort-level reports and chat text built from screening results: the
shortlist report, what changed after a refinement, the chat summary and the
assistant's context. Per-candidate rendering lives in ``explainability``.

Everything here is deterministic: reports restate the scores, checks and
reasoning already computed, so they never disagree with the ranking.
"""

from __future__ import annotations

import re
from .explainability import decision_cell, must_have_cell, summary_section
from .markdown import join, or_dash, table
from .schemas import CandidateMatch, RankChange, RankingDiff, Requirements

# Knocked-out candidates listed so a strict search still shows who came close.
NEAR_MISSES = 3
SUMMARY_TOP = 3


def requirements_lines(requirements: Requirements) -> list[str]:
    lines = [
        f"**Must have:** {join([r.label for r in requirements.must_have])}",
        f"**Nice to have:** {join(requirements.nice_to_have)}",
    ]
    if requirements.min_total_years:
        lines.append(f"**Experience:** {requirements.min_total_years:g}+ years")
    if requirements.min_degree:
        lines.append(f"**Education:** {requirements.min_degree} degree or higher")
    return lines


def _passing(candidates: list[CandidateMatch]) -> int:
    return sum(m.passes_must_haves for m in candidates)


def _near_misses(candidates: list[CandidateMatch]) -> list[CandidateMatch]:
    return [m for m in candidates if not m.passes_must_haves][:NEAR_MISSES]


# --- what changed --------------------------------------------------------------------------------


def _movement(change: RankChange) -> str:
    """"▲ 5 → 2", "▲ new (was #14)", "▼ left (now #12)", "= 3"."""
    if change.kind == "new":
        return f"▲ new (was #{change.old_rank})" if change.old_rank else "▲ new"
    if change.kind == "dropped":
        return f"▼ left (now #{change.new_rank})" if change.new_rank else "▼ left"
    arrow = {"up": "▲", "down": "▼", "same": "="}[change.kind]
    return f"= {change.new_rank}" if change.kind == "same" else f"{arrow} {change.old_rank} → {change.new_rank}"


def changes_section(diff: RankingDiff) -> list[str]:
    """Markdown lines: the requirement edits and how each affected candidate moved, with reasons."""
    lines = ["## What changed", ""]
    lines += [f"- {change}" for change in diff.requirement_changes] or ["- The requirements did not change."]
    moved = diff.of_kind("new", "dropped", "up", "down")
    if moved:
        rows = [
            [c.name, _movement(c), f"{or_dash(c.old_score)} → {or_dash(c.new_score)}", join(c.reasons, "–")]
            for c in moved
        ]
        lines += ["", *table(["Candidate", "Change", "Score", "Why"], rows)]
    else:
        lines += ["", "The shortlist order is unchanged."]
    return lines + [""]


# --- shortlist report and chat summary -----------------------------------------------------------------


def confirmation_message(questions: list[str]) -> str:
    """Asks the recruiter to settle the requirements the job description leaves open
    (``questions``: ``OpenQuestion.line()`` texts)."""
    example = re.search(r"\*\*([^*,]+)[^*]*\*\* only nice-to-have", questions[0])
    hint = f' (for example "make {example.group(1)} a must-have")' if example else ""
    return "\n".join([
        "Before screening, please check what the job description leaves open:",
        *(f"- {question}." for question in questions),
        "",
        f"Reply **yes** to screen with these, or say what to change{hint}.",
    ])


def shortlist_report(
    requirements: Requirements,
    candidates: list[CandidateMatch],
    shortlist: list[CandidateMatch],
    diff: RankingDiff | None = None,
    assumptions: list[str] | None = None,
) -> str:
    """The shortlist report: requirements (and any open ones the screening assumed), what
    changed (after a refinement), the ranking table with every round's scores and the
    recommendation, a block per candidate (strengths, gaps, improvements) and the closest
    knocked-out candidates."""
    passing = _passing(candidates)
    lines = [f"# Shortlist: {requirements.role_title}", ""]
    lines += [f"- {line}" for line in requirements_lines(requirements)] + [""]
    if assumptions:
        lines += ["**Please check:** the job description leaves these open, so the screening assumed:",
                  *(f"- {assumption}" for assumption in assumptions), ""]
    if diff is not None:
        lines += changes_section(diff)
    lines.append(
        f"Screened {len(candidates)} resumes: {passing} meet every must-have, "
        f"{len(candidates) - passing} were knocked out."
    )
    if shortlist:
        rows = [
            [m.rank, f"{m.name} (`{m.id}`)", m.final_score, m.match_score,
             or_dash(m.assessment.deep_score if m.assessment else None), must_have_cell(m), decision_cell(m)]
            for m in shortlist
        ]
        header = ["#", "Candidate", "Score", "Retrieval", "Deep", "Must-haves", "Recommendation"]
        lines += ["", *table(header, rows)]
        for m in shortlist:
            lines += summary_section(m, requirements)
    else:
        lines += ["", "**No candidate meets every must-have.**"]

    misses = _near_misses(candidates)
    if misses:
        lines += ["", "## Closest knocked-out candidates", ""]
        lines += [f"- {m.name} (`{m.id}`, {m.match_score}): fails {join(m.failed_requirements)}" for m in misses]
    return "\n".join(lines) + "\n"


def _recommendation_summary(shortlist: list[CandidateMatch]) -> list[str]:
    """Sentences on who to advance and who is borderline (empty before round 3)."""
    decided = [m for m in shortlist if m.recommendation is not None]
    if not decided:
        return []
    advance = [m.name for m in decided if m.recommendation.decision == "interview"]
    sentences = [f"Recommended for interview: {join(advance)}." if advance
                 else "No candidate is recommended for interview."]
    borderline = [f"{m.name} ({m.final_score})" for m in shortlist if m.borderline]
    if borderline:
        sentences.append(f"Borderline: {join(borderline)}.")
    return sentences


def shortlist_summary(
    requirements: Requirements, candidates: list[CandidateMatch], shortlist: list[CandidateMatch]
) -> str:
    """A short chat message: how many passed and who leads, or who came closest."""
    passing = _passing(candidates)
    head = (
        f"Screened {len(candidates)} resumes for {requirements.role_title}: "
        f"{passing} meet every must-have."
    )
    if shortlist:
        top = ", ".join(f"{m.name} ({m.final_score})" for m in shortlist[:SUMMARY_TOP])
        return " ".join([f"{head} Top matches: {top}.", *_recommendation_summary(shortlist)])
    misses = _near_misses(candidates)
    if not misses:
        return f"{head} No resumes are indexed yet."
    closest = misses[0]
    return (
        f"{head} The closest is {closest.name} ({closest.match_score}), who fails "
        f"{join(closest.failed_requirements)}. Consider relaxing that requirement."
    )


def conversation_context(
    requirements: Requirements | None,
    candidates: list[CandidateMatch],
    shortlist: list[CandidateMatch],
    diff: RankingDiff | None = None,
) -> str:
    """The current search (and the latest refinement's changes) for the assistant's system prompt."""
    if requirements is None:
        return "No role is set yet. The user can describe one to start a search."
    lines = [f"Current role: {requirements.role_title}", *requirements_lines(requirements)]
    lines.append(f"Screened {len(candidates)} resumes; {_passing(candidates)} pass every must-have.")
    if shortlist:
        lines.append("Shortlist (rank. name (id): final score, must-haves, recommendation):")
        lines += [f"{m.rank}. {m.name} ({m.id}): {m.final_score}, {must_have_cell(m)}, {decision_cell(m)}"
                  for m in shortlist]
    if diff is not None:
        lines.append(f"Latest refinement: {join(diff.requirement_changes, 'no requirement changes')}")
        lines += [f"- {c.name}: {_movement(c)}, {join(c.reasons, 'no reason recorded')}"
                  for c in diff.of_kind("new", "dropped", "up", "down")]
    return "\n".join(lines)
