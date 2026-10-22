"""What changed after a refinement, and why: computed, not guessed.

``requirement_changes`` lists the edits between two ``Requirements``.
``ranking_diff`` compares two screenings and gives every candidate who is in
the new shortlist or left the old one a reason for their move: a must-have
they now fail or no longer fail, the score component whose weighted
contribution changed most, or, when nothing about them changed, the others
who were knocked out above or overtook them. The LLM only narrates this diff.
"""

from __future__ import annotations

from .rag import job_matcher
from .schemas import CandidateMatch, ChangeKind, RankChange, RankingDiff, Requirements

# Weights of the hybrid score, so component changes are compared by their effect on it.
COMPONENT_WEIGHTS = {
    "semantic": job_matcher.SEMANTIC_WEIGHT,
    "keyword": job_matcher.KEYWORD_WEIGHT,
    "skill_coverage": job_matcher.SKILL_WEIGHT,
}
COMPONENT_LABELS = {
    "semantic": "semantic fit",
    "keyword": "keyword match",
    "skill_coverage": "must-have skill coverage",
}
# Smallest weighted change (on the 0..1 hybrid scale) worth naming as a reason.
MIN_COMPONENT_EFFECT = 0.005


def _years(value: float | None, unit: str = "yrs") -> str:
    return f"{value:g}+ {unit}" if value else "none"


def requirement_changes(old: Requirements, new: Requirements) -> list[str]:
    """Human-readable edits from ``old`` to ``new``."""
    changes = []
    old_must = {r.skill.lower(): r for r in old.must_have}
    new_must = {r.skill.lower(): r for r in new.must_have}
    new_nice = {s.lower() for s in new.nice_to_have}
    for key, req in new_must.items():
        if key not in old_must:
            changes.append(f"{req.label} is now a must-have")
        elif old_must[key].min_years != req.min_years:
            before = _years(old_must[key].min_years)
            changes.append(f"{req.skill}: {before} → {_years(req.min_years)}")
    for key, req in old_must.items():
        if key not in new_must:
            where = "moved to nice-to-have" if key in new_nice else "dropped"
            changes.append(f"{req.skill} is no longer a must-have ({where})")
    old_nice = {s.lower() for s in old.nice_to_have}
    added = [s for s in new.nice_to_have if s.lower() not in old_nice and s.lower() not in old_must]
    removed = [s for s in old.nice_to_have if s.lower() not in new_nice and s.lower() not in new_must]
    if added:
        changes.append(f"Nice to have added: {', '.join(added)}")
    if removed:
        changes.append(f"Nice to have removed: {', '.join(removed)}")
    if old.min_total_years != new.min_total_years:
        before, after = _years(old.min_total_years, "years"), _years(new.min_total_years, "years")
        changes.append(f"Total experience: {before} → {after}")
    if old.min_degree != new.min_degree:
        changes.append(f"Degree: {old.min_degree or 'none'} → {new.min_degree or 'none'}")
    return changes


def _component_reason(old: CandidateMatch, new: CandidateMatch) -> str | None:
    """The score component whose weighted change moved this candidate most."""

    def value(match: CandidateMatch, name: str) -> float:
        return getattr(match.score_breakdown, name) or 0.0

    effects = {name: weight * (value(new, name) - value(old, name)) for name, weight in COMPONENT_WEIGHTS.items()}
    name = max(effects, key=lambda n: abs(effects[n]))
    if abs(effects[name]) < MIN_COMPONENT_EFFECT:
        return None
    return f"{COMPONENT_LABELS[name]} {value(old, name):.0%} → {value(new, name):.0%}"


def reasons_for(old: CandidateMatch | None, new: CandidateMatch | None) -> list[str]:
    """Why a candidate's standing changed between two screenings."""
    if old is None:
        return ["not screened before"]
    if new is None:
        return ["no longer in the index"]
    reasons = []
    if old.failed_requirements and not new.failed_requirements:
        reasons.append(f"no longer fails {'; '.join(old.failed_requirements)}")
    if new.failed_requirements:
        newly = [r for r in new.failed_requirements if r not in old.failed_requirements]
        if newly:
            reasons.append(f"now fails {'; '.join(newly)}")
    if old.match_score != new.match_score:
        component = _component_reason(old, new)
        score = f"score {old.match_score} → {new.match_score}"
        reasons.append(f"{score} ({component})" if component else score)
    if new.unverified and not old.unverified:
        reasons.append(f"now unverified: {'; '.join(new.unverified)}")
    elif old.unverified and not new.unverified and not new.failed_requirements:
        reasons.append("no longer unverified")  # resolved in their favour, not turned into a failure
    return reasons


def _relative_reason(
    kind: ChangeKind, match: CandidateMatch, old_by_id: dict, new_by_id: dict
) -> str | None:
    """Why a candidate moved although nothing about them changed: others did."""
    before = old_by_id[match.id]
    if kind == "up":
        knocked = [m for m in old_by_id.values()
                   if m.rank < before.rank and m.id in new_by_id and not new_by_id[m.id].passes_must_haves]
        if knocked:
            return f"{len(knocked)} candidate(s) ranked above now fail a must-have"
    if kind == "down":
        overtaking = [m.name for m in new_by_id.values()
                      if m.rank < match.rank and m.id in old_by_id and old_by_id[m.id].rank > before.rank]
        if overtaking:
            return f"overtaken by {', '.join(overtaking[:3])}"
    return None


def _kind(old_rank: int | None, new_rank: int) -> ChangeKind:
    if old_rank is None:
        return "new"
    return "up" if new_rank < old_rank else "down" if new_rank > old_rank else "same"


def ranking_diff(
    old_requirements: Requirements,
    new_requirements: Requirements,
    old_candidates: list[CandidateMatch],
    new_candidates: list[CandidateMatch],
    old_shortlist: list[CandidateMatch],
    new_shortlist: list[CandidateMatch],
) -> RankingDiff:
    """Every candidate in the new shortlist, then everyone who left it, with reasons."""
    old_by_id = {m.id: m for m in old_candidates}
    new_by_id = {m.id: m for m in new_candidates}
    old_ranks = {m.id: m.rank for m in old_shortlist}
    new_ids = {m.id for m in new_shortlist}

    changes = []
    for match in new_shortlist:
        before = old_by_id.get(match.id)
        kind = _kind(old_ranks.get(match.id), match.rank)
        reasons = reasons_for(before, match)
        if not reasons and kind in ("up", "down"):
            relative = _relative_reason(kind, match, old_by_id, new_by_id)
            reasons = [relative] if relative else []
        changes.append(RankChange(
            id=match.id, name=match.name, kind=kind,
            old_rank=before.rank if before else None, new_rank=match.rank,
            old_score=before.match_score if before else None, new_score=match.match_score,
            reasons=reasons,
        ))
    for match in old_shortlist:
        if match.id in new_ids:
            continue
        now = new_by_id.get(match.id)
        reasons = reasons_for(match, now)
        if now is not None and now.passes_must_haves:
            reasons.append(f"outranked by other candidates (now #{now.rank})")
        changes.append(RankChange(
            id=match.id, name=match.name, kind="dropped",
            old_rank=match.rank, new_rank=now.rank if now else None,
            old_score=match.match_score, new_score=now.match_score if now else None,
            reasons=reasons,
        ))
    return RankingDiff(
        requirement_changes=requirement_changes(old_requirements, new_requirements), changes=changes
    )
