"""Markdown reports and chat summaries built from screening results.

Everything here is deterministic: reports restate the scores, checks and
reasoning already computed, so they never disagree with the ranking.
"""

from __future__ import annotations

from .schemas import CandidateMatch, RankChange, RankingDiff, Requirements

# Knocked-out candidates listed so a strict search still shows who came close.
NEAR_MISSES = 3
SUMMARY_TOP = 3


def _join(items: list[str], empty: str = "none") -> str:
    return ", ".join(items) if items else empty


def requirements_lines(requirements: Requirements) -> list[str]:
    lines = [
        f"**Must have:** {_join([r.label for r in requirements.must_have])}",
        f"**Nice to have:** {_join(requirements.nice_to_have)}",
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


def _must_have_cell(match: CandidateMatch) -> str:
    if match.unverified:
        return f"⚠ {len(match.unverified)} unverified"
    return "✓ all met" if match.must_have_checks else "–"


def _percent(value: float | None) -> str:
    return "–" if value is None else f"{value:.0%}"


def _or_dash(value: int | None) -> str:
    return "–" if value is None else str(value)


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
        lines += ["", "| Candidate | Change | Score | Why |", "|---|---|---|---|"]
        for change in moved:
            score = f"{_or_dash(change.old_score)} → {_or_dash(change.new_score)}"
            lines.append(f"| {change.name} | {_movement(change)} | {score} | {_join(change.reasons, '–')} |")
    else:
        lines += ["", "The shortlist order is unchanged."]
    return lines + [""]


def shortlist_report(
    requirements: Requirements,
    candidates: list[CandidateMatch],
    shortlist: list[CandidateMatch],
    diff: RankingDiff | None = None,
) -> str:
    """The full shortlist report: requirements, what changed (after a refinement),
    ranking table, reasoning and near misses."""
    passing = _passing(candidates)
    lines = [f"# Shortlist: {requirements.role_title}", ""]
    lines += [f"- {line}" for line in requirements_lines(requirements)] + [""]
    if diff is not None:
        lines += changes_section(diff)
    lines.append(
        f"Screened {len(candidates)} resumes: {passing} meet every must-have, "
        f"{len(candidates) - passing} were knocked out."
    )
    if shortlist:
        lines += [
            "",
            "| # | Candidate | Score | Semantic | Keyword | Skill coverage | Must-haves |",
            "|---|---|---|---|---|---|---|",
        ]
        for m in shortlist:
            b = m.score_breakdown
            lines.append(
                f"| {m.rank} | {m.name} (`{m.id}`) | {m.match_score} | {_percent(b.semantic)} | "
                f"{_percent(b.keyword)} | {_percent(b.skill_coverage)} | {_must_have_cell(m)} |"
            )
        for m in shortlist:
            lines += ["", f"## {m.rank}. {m.name} ({m.match_score})", "", m.reasoning]
            if m.excerpts:
                lines += ["", f"> {m.excerpts[0]}"]
    else:
        lines += ["", "**No candidate meets every must-have.**"]

    misses = _near_misses(candidates)
    if misses:
        lines += ["", "## Closest knocked-out candidates", ""]
        lines += [f"- {m.name} (`{m.id}`, {m.match_score}): fails {_join(m.failed_requirements)}" for m in misses]
    return "\n".join(lines) + "\n"


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
        top = ", ".join(f"{m.name} ({m.match_score})" for m in shortlist[:SUMMARY_TOP])
        return f"{head} Top matches: {top}."
    misses = _near_misses(candidates)
    if not misses:
        return f"{head} No resumes are indexed yet."
    closest = misses[0]
    return (
        f"{head} The closest is {closest.name} ({closest.match_score}), who fails "
        f"{_join(closest.failed_requirements)}. Consider relaxing that requirement."
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
        lines.append("Shortlist (rank. name (id): score, must-haves):")
        lines += [f"{m.rank}. {m.name} ({m.id}): {m.match_score}, {_must_have_cell(m)}" for m in shortlist]
    if diff is not None:
        lines.append(f"Latest refinement: {_join(diff.requirement_changes, 'no requirement changes')}")
        lines += [f"- {c.name}: {_movement(c)}, {_join(c.reasons, 'no reason recorded')}"
                  for c in diff.of_kind("new", "dropped", "up", "down")]
    return "\n".join(lines)
