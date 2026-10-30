"""Guardrails between the LLM and the ranking: the model judges, code decides.

The deep review (round 2) only returns categorical verdicts with verbatim
quotes. Code then:

1. checks every quote against the resume text: a verdict whose quote is not
   in the resume is downgraded (no credit for invented evidence);
2. sets a skill's verdict by where its evidence is: named in a role or project
   (code finds the line) is strong, only listed or claimed is partial; then bounds verdicts by the
   facts round 1 measured: a must-have measured as met counts as strong, an
   unverified one (listed, years unclear) as exactly partial;
3. keeps the model's risk flags only when their quote is in the resume, one per
   kind, at the severity code gives that kind (``RISK_SEVERITY``), and adds
   code-detected risks, such as text that tries to instruct the model;
4. computes the deep score from a fixed rubric;

and round 3 decides whom to recommend for interview with a fixed rule, leaving the model only
the rationale. The same verdicts therefore always give the same score and
decision, whichever model produced them, and every adjustment is recorded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .rag import resume_rag, skills_vocab
from .rag.job_matcher import requirement_label
from .schemas import (
    Assessment,
    CandidateMatch,
    DeepReview,
    MustHaveCheck,
    Recommendation,
    RequirementEvidence,
    Requirements,
    ReviewRisk,
    RiskFlag,
    RiskKind,
    Verdict,
)

# --- rubric ------------------------------------------------------------------------------------------

VERDICT_VALUE: dict[Verdict, float] = {"strong": 1.0, "partial": 0.5, "missing": 0.0}
# Share of the deep score from must-haves; nice-to-haves get the rest (all of it goes to
# must-haves when the role has no nice-to-haves, and the other way round).
MUST_SHARE = 0.7
RISK_PENALTY = {"low": 2, "medium": 5, "high": 10}
MAX_RISK_PENALTY = 20
# Each kind of risk the model may raise, at the severity code gives it. The model only says
# whether a kind applies; only code-detected risks can be high (and block an interview).
RISK_SEVERITY: dict[RiskKind, str] = {"claim_mismatch": "medium", "contradiction": "medium", "repeated_content": "low"}

# --- decision rule -------------------------------------------------------------------------------------

INTERVIEW_THRESHOLD = 75
# Distance from the threshold that makes a score-based decision clear-cut, or a close call.
CLEAR_MARGIN, CLOSE_MARGIN = 10, 5

# --- evidence checks -------------------------------------------------------------------------------------

# Share of a quote that must appear verbatim (after normalising) in the resume.
MIN_QUOTE_MATCH = 0.8
MIN_QUOTE_CHARS = 8
DOWNGRADE: dict[Verdict, Verdict] = {"strong": "partial", "partial": "missing", "missing": "missing"}
# Must-have kinds whose status round 1 measured (rather than just found the skill listed).
MEASURED_KINDS = {"skill_years", "total_years", "degree"}

INJECTION_RE = re.compile(
    r"ignore (?:all |any |the )?(?:previous|prior|above) (?:instructions|prompts?)|disregard (?:all |the )?"
    r"(?:previous |above )?instructions|you are (?:now )?(?:an? )?(?:ai|assistant|language model)|"
    r"system prompt|(?:rank|rate|score) (?:me|this candidate) (?:first|highest|as)|"
    r"recommend (?:me|this candidate) for hire",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReviewItem:
    """One requirement the deep review must judge, with a stable id for the model to cite."""

    id: str
    label: str
    kind: str  # "must" or "nice"
    check_type: str | None  # the must-have's ``to_must_haves`` type; None for nice-to-haves
    skill: str | None
    check_label: str | None  # the round-1 ``MustHaveCheck.requirement`` it pairs with


def review_items(requirements: Requirements) -> list[ReviewItem]:
    """Must-haves (M1, M2, ...) then nice-to-haves (N1, ...), each with a stable id."""
    items = []
    for i, req in enumerate(requirements.to_must_haves(), 1):
        label = {
            "skill_years": f"{req.get('skill')} ({req.get('min_years', 0):g}+ yrs)",
            "skill": req.get("skill"),
            "total_years": f"{req.get('min_years', 0):g}+ years of total experience",
            "degree": f"{req.get('degree')} degree or higher",
        }[req["type"]]
        items.append(ReviewItem(f"M{i}", label, "must", req["type"], req.get("skill"), requirement_label(req)))
    for i, skill in enumerate(requirements.nice_to_have, 1):
        items.append(ReviewItem(f"N{i}", skill, "nice", None, skill, None))
    return items


def items_for_prompt(items: list[ReviewItem]) -> list[dict]:
    return [{"id": item.id, "requirement": item.label, "kind": f"{item.kind}-have"} for item in items]


# --- 1. evidence grounding ---------------------------------------------------------------------------------


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\"'“”‘’…•▪◦·]|\.{3}", " ", text.lower())).strip()


def _found(quote: str, text: str) -> float:
    """How much of ``quote`` (normalised) appears in one piece in ``text`` (normalised), 0..1."""
    if quote in text:
        return 1.0
    match = SequenceMatcher(None, quote, text, autojunk=False).find_longest_match(0, len(quote), 0, len(text))
    return match.size / len(quote)


def grounded(quote: str, resume_text: str) -> bool:
    """Whether ``quote`` appears in the resume, allowing small differences in punctuation."""
    quote = _normalise(quote)
    return len(quote) >= MIN_QUOTE_CHARS and _found(quote, _normalise(resume_text)) >= MIN_QUOTE_MATCH


# --- 2. what the evidence shows -----------------------------------------------------------------------------

# Sections that describe work done. Evidence anywhere else (a skills list, a summary, a
# certificate) shows a skill is claimed, not used.
WORK_SECTIONS = {"Experience", "Projects"}


def quote_section(quote: str, resume_text: str) -> str | None:
    """The section a grounded quote comes from; None when the resume has no sections."""
    sections = resume_rag.split_sections(resume_text)
    if set(sections) <= {resume_rag.GENERAL_SECTION}:
        return None
    quote = _normalise(quote)
    return max(sections, key=lambda name: _found(quote, _normalise(sections[name])))


def names_skill(quote: str, skill: str) -> bool:
    """Whether ``quote`` names ``skill``, by any of its spellings or a skill implying it
    (Django: Python)."""
    found = {s.lower() for s in skills_vocab.with_implied(skills_vocab.find_skills(quote))}
    return skill.lower() in found or bool(re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", quote, re.IGNORECASE))


def work_line(skill: str, resume_text: str) -> str | None:
    """The first line of a role or project that names ``skill``, if any."""
    for name, body in resume_rag.split_sections(resume_text).items():
        if name in WORK_SECTIONS:
            for line in body.splitlines():
                if names_skill(line, skill):
                    return line.strip(" \t-•*▪◦·")
    return None


def verdict_from_evidence(
    verdict: Verdict, item: ReviewItem, quote: str, resume_text: str
) -> tuple[Verdict, str, str]:
    """A skill's verdict and evidence by where the evidence is: named in a role or project is
    strong (code finds the line, whichever line the model quoted), only listed or claimed is
    partial. ``(verdict, quote, note)``; the note says why when the verdict changed.

    The model may call the same evidence strong in one run and partial in the next, or quote
    the skills list in one run and a role in the next; this gives every run the same answer.
    """
    if item.skill is None:
        return verdict, quote, ""
    used = work_line(item.skill, resume_text)
    if used:
        quoted_work = quote and quote_section(quote, resume_text) in WORK_SECTIONS and names_skill(quote, item.skill)
        return "strong", quote if quoted_work else used, "" if verdict == "strong" else "used in a role or project"
    section = quote_section(quote, resume_text) if quote else None
    if verdict == "missing" or section is None:
        return verdict, quote, ""
    reason = "the quoted work does not name it" if section in WORK_SECTIONS else f"only listed or claimed ({section.lower()})"
    return "partial", quote, "" if verdict == "partial" else reason


# --- bounds from round 1 -----------------------------------------------------------------------------------


def bound_by_check(verdict: Verdict, item: ReviewItem, check: MustHaveCheck | None, listed: bool) -> tuple[Verdict, str]:
    """The verdict allowed by what round 1 measured, and a note when it changes."""
    if item.kind == "nice":
        if verdict == "missing" and listed:
            return "partial", "listed in the resume's skills"
        return verdict, ""
    if check is None:
        return verdict, ""
    if check.status == "met" and item.check_type in MEASURED_KINDS and verdict != "strong":
        return "strong", f"measured as met: {check.detail}"
    if check.status == "met" and verdict == "missing":
        return "partial", f"listed on the resume: {check.detail}"
    if check.status == "unverified" and verdict != "partial":
        # Listed on the resume (so not missing), years not evidenced (so not strong).
        return "partial", f"listed but years not evidenced by dated roles: {check.detail}"
    return verdict, ""


# --- 3. code-detected risks -----------------------------------------------------------------------------


def injection_flags(resume_text: str) -> list[RiskFlag]:
    found = sorted({m.group(0).lower() for m in INJECTION_RE.finditer(resume_text)})
    if not found:
        return []
    flag = f"Resume contains text aimed at the screening model: {', '.join(found)}"
    return [RiskFlag(flag=flag, severity="high", kind="prompt_injection")]


def review_risks(risks: list[ReviewRisk], resume_text: str) -> list[RiskFlag]:
    """The model's risks that count: quoted from the resume, the first of each kind, at the
    kind's severity. How many ways the model words one concern no longer changes the score."""
    kept: dict[str, RiskFlag] = {}
    for risk in risks:
        if risk.kind not in kept and grounded(risk.evidence_quote, resume_text):
            kept[risk.kind] = RiskFlag(flag=risk.flag, severity=RISK_SEVERITY[risk.kind], kind=risk.kind)
    return list(kept.values())


# --- 4. rubric ---------------------------------------------------------------------------------------------


def rubric_score(evidence: list[RequirementEvidence], risk_flags: list[RiskFlag]) -> int:
    """Deep score 0-100: weighted verdict values minus capped risk penalties."""
    groups = {kind: [VERDICT_VALUE[e.verdict] for e in evidence if e.kind == kind] for kind in ("must", "nice")}
    if groups["must"] and groups["nice"]:
        shares = {"must": MUST_SHARE, "nice": 1 - MUST_SHARE}
    else:
        shares = {kind: 1.0 if values else 0.0 for kind, values in groups.items()}
    score = 100 * sum(shares[kind] * sum(values) / len(values) for kind, values in groups.items() if values)
    if not evidence:
        score = 100.0  # no requirements to miss
    penalty = min(MAX_RISK_PENALTY, sum(RISK_PENALTY[flag.severity] for flag in risk_flags))
    return max(0, min(100, round(score - penalty)))


def build_assessment(
    review: DeepReview, items: list[ReviewItem], match: CandidateMatch, resume_text: str
) -> Assessment:
    """Apply guardrails 1-4 to the model's review of ``match``."""
    reviews = {r.requirement_id.strip().upper(): r for r in review.requirements}
    checks = {check.requirement: check for check in match.must_have_checks}
    skills = {s.lower() for s in match.matched_skills}

    evidence = []
    for item in items:
        reviewed = reviews.get(item.id)
        notes = []
        if reviewed is None:
            verdict, quote = "missing", ""
            notes.append("not reviewed by the model")
        else:
            verdict, quote = reviewed.verdict, reviewed.evidence_quote.strip()
            if verdict != "missing" and not grounded(quote, resume_text):
                verdict, quote = DOWNGRADE[verdict], ""
                notes.append("quoted evidence not found in the resume")
        verdict, quote, note = verdict_from_evidence(verdict, item, quote, resume_text)
        if note:
            notes.append(note)
        listed = bool(item.skill and item.skill.lower() in skills)
        check = checks.get(item.check_label)
        verdict, note = bound_by_check(verdict, item, check, listed)
        if note:
            notes.append(note)
        evidence.append(RequirementEvidence(
            requirement=item.label, kind=item.kind, verdict=verdict,
            evidence=quote or (check.detail if check else ""),
            note="; ".join(notes),
        ))
    risk_flags = [*review_risks(review.risk_flags, resume_text), *injection_flags(resume_text)]
    return Assessment(
        requirement_evidence=evidence, strengths=review.strengths, gaps=review.gaps,
        risk_flags=risk_flags, deep_score=rubric_score(evidence, risk_flags), summary=review.summary,
    )


# --- round 3 decision rule ------------------------------------------------------------------------------


def decide(match: CandidateMatch) -> Recommendation:
    """Recommend an interview when the final score clears the threshold, no must-have is
    missing and no risk is high. Confidence reflects how clear-cut that is. ``rationale`` is left
    as the rule's outcome until the model explains it."""
    assessment = match.assessment
    missing = [e.requirement for e in assessment.requirement_evidence if e.kind == "must" and e.verdict == "missing"]
    high_risks = [f.flag for f in assessment.risk_flags if f.severity == "high"]
    margin = match.final_score - INTERVIEW_THRESHOLD

    blockers = []
    if missing:
        blockers.append(f"must-have missing: {', '.join(missing)}")
    if high_risks:
        blockers.append(f"high risk: {'; '.join(high_risks)}")
    if blockers:
        rule = "blocked by " + " and ".join(blockers)
        return Recommendation(decision="not_now", confidence="high", rule=rule, rationale=rule)

    advance = margin >= 0
    shaky = any(e.kind == "must" and e.verdict != "strong" for e in assessment.requirement_evidence) or any(
        f.severity == "medium" for f in assessment.risk_flags
    )
    if abs(margin) < CLOSE_MARGIN:
        confidence = "low"
    elif abs(margin) >= CLEAR_MARGIN and not (advance and shaky):
        confidence = "high"
    else:
        confidence = "medium"
    comparison = "≥" if advance else "<"
    rule = f"final score {match.final_score} {comparison} {INTERVIEW_THRESHOLD}; no must-have missing; no high risk"
    return Recommendation(decision="interview" if advance else "not_now", confidence=confidence, rule=rule,
                          rationale=rule)
