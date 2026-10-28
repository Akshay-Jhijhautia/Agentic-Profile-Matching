"""Match a job description against the resumes indexed by ``resume_rag``.

The job description is parsed for required skills and must-have requirements
("5+ years of Python", "Master's degree"). Candidates are then found with a
hybrid search:

- semantic: the JD is embedded with the same embedder that built the index
  and compared against every resume chunk; hits are grouped per candidate;
- keyword: BM25 over each resume's text, with canonical skill tokens added so
  "ML" in a JD matches "machine learning" in a resume;
- critical skills: the JD's required skills checked against each candidate's
  extracted skills.

The candidate pool is the union of the semantic and keyword hits, so a resume
with an exact critical skill is not missed when its embedding ranks low.
Candidates failing a must-have are filtered out; the rest get a 0-100
match score, matched skills, relevant excerpts and a written reasoning.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from . import resume_rag, skills_vocab

TOP_K = 10
# Resumes taken from the keyword ranking into the hybrid pool.
KEYWORD_POOL = 20
# Hybrid score weights (sum to 1). Without required skills in the JD, the skill
# weight is spread over the other two in proportion.
SEMANTIC_WEIGHT = 0.5
KEYWORD_WEIGHT = 0.2
SKILL_WEIGHT = 0.3
# Query repeats per skill token: required skills count double in BM25.
REQUIRED_SKILL_BOOST = 2
# Cosine similarity range mapped onto 0..1 before blending, per embedder (full
# name first, then kind). Raw similarity would compress every score: with
# text-embedding-3-small the sample JD/resume pairs measured 0.34 (unrelated) to
# 0.62 (near-perfect fit). Local models' ranges were fitted so the candidates that
# reach the shortlist (each query's top 5) score as they do with OpenAI, keeping the
# interview threshold meaningful; see ``scripts/ranking_eval.py``.
SEMANTIC_RANGES = {
    "openai": (0.30, 0.65),
    "hashing": (0.0, 0.50),
    "local:ibm-granite/granite-embedding-small-english-r2": (0.661, 0.953),
    "local:snowflake/snowflake-arctic-embed-m": (-0.123, 0.509),
}
DEFAULT_SEMANTIC_RANGE = (0.0, 1.0)
# Fields ``JobMatcher.evaluate`` adds on top of a ``match`` entry.
EVALUATION_ONLY_KEYS = {"failed_requirements"}
# Chunks retrieved per query before grouping by candidate (about 4-5 per resume).
CHUNK_POOL = 50
# Candidate semantic score = blend of the best chunk and the mean of the top 3,
# so one strong section counts most but broad relevance still helps.
BEST_CHUNK_WEIGHT = 0.6
TOP_CHUNKS_FOR_MEAN = 3

BULLET_RE = re.compile(r"^\s*[-•*▪◦·]\s*")
_PREFERRED_RE = re.compile(
    r"\b(?:nice[- ]to[- ]haves?|preferred|bonus|a plus|desirable|good[- ]to[- ]haves?|optional)\b",
    re.IGNORECASE,
)
_SECTION_HEADING_RE = re.compile(
    r"\b(?:requirements?|required|qualifications|responsibilities|about|what you'?ll|"
    r"who you are|what we'?re looking for|must[- ]haves?|the role|benefits|perks|we offer)\b",
    re.IGNORECASE,
)
# "5+ years", "5 + yrs", "3-5 years", "at least 4 years of hands-on experience with"
YEARS_RE = re.compile(
    r"(?P<years>\d+(?:\.\d+)?)\s*(?:\+|(?:-|–|to)\s*\d+(?:\.\d+)?)?\s*\+?\s*(?:years?|yrs?)\b"
    r"(?:\s+of)?(?:\s+(?:professional|hands-on|industry|commercial|relevant|production))?"
    r"(?:\s+experience)?(?:\s+(?:in|with|using|building|developing|of))?\s*(?P<rest>[^.;\n]{0,80})",
    re.IGNORECASE,
)
# A skill must follow a years phrase this closely to be the subject of it.
_YEARS_SKILL_WINDOW = 25


def is_jd_header(line: str) -> bool:
    """A heading such as "Requirements:" or "Nice to have" (not a short requirement line)."""
    stripped = line.strip()
    if not stripped or BULLET_RE.match(line) or len(stripped) > 60:
        return False
    if stripped.endswith(":"):
        return True
    return len(stripped.split()) <= 5 and bool(
        _SECTION_HEADING_RE.search(stripped) or _PREFERRED_RE.search(stripped)
    )


def _years_requirements(line: str) -> list[dict]:
    requirements = []
    for match in YEARS_RE.finditer(line):
        years = float(match["years"])
        named = skills_vocab.find_skills(match["rest"][:_YEARS_SKILL_WINDOW])
        source = match.group(0).strip()
        if named:
            requirements.append(
                {"type": "skill_years", "skill": named[0], "min_years": years, "source": source}
            )
        elif "experience" in line.lower():
            requirements.append({"type": "total_years", "min_years": years, "source": source})
    return requirements


def parse_job_description(job_description: str) -> dict:
    """Pull skills and must-have requirements out of a job description.

    Lines under a "Nice to have" / "Preferred" heading, and clauses (split on
    ";") that say "preferred" / "a plus" themselves, are optional; everything
    else is core.
    Must-haves come from core lines only:

    - ``{"type": "skill_years", "skill", "min_years"}`` from "5+ years of Python"
    - ``{"type": "total_years", "min_years"}`` from "3+ years of experience"
    - ``{"type": "degree", "degree"}`` from "Master's degree in CS" (the lowest
      degree named on the line, so "Bachelor's or Master's" means Bachelor's)

    Each must-have also carries the ``source`` text it was parsed from.
    Returns ``{"text", "required_skills", "preferred_skills", "must_haves"}``.
    """
    core_lines, optional_lines = [], []
    in_optional_section = False
    for line in job_description.splitlines():
        if is_jd_header(line):
            in_optional_section = bool(_PREFERRED_RE.search(line))
        if in_optional_section:
            optional_lines.append(line)
            continue
        for clause in line.split(";"):
            (optional_lines if _PREFERRED_RE.search(clause) else core_lines).append(clause)

    required = skills_vocab.find_skills("\n".join(core_lines))
    preferred = [s for s in skills_vocab.find_skills("\n".join(optional_lines)) if s not in required]

    must_haves = []
    for line in core_lines:
        must_haves.extend(_years_requirements(line))
        degrees = resume_rag.degrees_mentioned(line)
        if degrees:
            must_haves.append({"type": "degree", "degree": degrees[-1], "source": line.strip()})

    return {
        "text": job_description.strip(),
        "required_skills": required,
        "preferred_skills": preferred,
        "must_haves": _dedupe_must_haves(must_haves),
    }


def _dedupe_must_haves(must_haves: list[dict]) -> list[dict]:
    """Keep the strictest requirement of each kind (per skill for skill years)."""
    strictest: dict[tuple, dict] = {}
    for req in must_haves:
        key = (req["type"], req.get("skill"))
        current = strictest.get(key)
        if current is None:
            strictest[key] = req
        elif req["type"] == "degree":
            if resume_rag.DEGREE_LEVELS[req["degree"]] > resume_rag.DEGREE_LEVELS[current["degree"]]:
                strictest[key] = req
        elif req["min_years"] > current["min_years"]:
            strictest[key] = req
    return list(strictest.values())


def _excerpt(document: str) -> str:
    """Chunk text without the ``Candidate: ... | Section: ...`` embedding prefix."""
    return document.split("\n", 1)[1] if "\n" in document else document


# --- keyword search ------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.]*[a-z0-9+#]|[a-z0-9]")


def _skill_token(skill: str) -> str:
    return "skill:" + skill.lower()


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens without stopwords, keeping "c++", "c#" and "node.js" whole."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in resume_rag.STOPWORDS]


def _resume_tokens(texts: list[str], skills: list[str]) -> list[str]:
    """BM25 document for one resume: its words plus one token per extracted skill."""
    tokens = [t for text in texts for t in tokenize(text)]
    return tokens + [_skill_token(s) for s in skills]


def _query_tokens(parsed: dict) -> list[str]:
    tokens = tokenize(parsed["text"])
    tokens += [_skill_token(s) for s in parsed["required_skills"]] * REQUIRED_SKILL_BOOST
    tokens += [_skill_token(s) for s in parsed["preferred_skills"]]
    return tokens


def _skill_match(required: list[str], preferred: list[str], candidate_skills: list[str]) -> dict:
    have = {s.lower() for s in candidate_skills}
    matched = [s for s in required if s.lower() in have]
    return {
        "matched_skills": matched,
        "missing_skills": [s for s in required if s.lower() not in have],
        "matched_preferred_skills": [s for s in preferred if s.lower() in have],
        "skill_coverage": round(len(matched) / len(required), 4) if required else None,
    }


def calibrate_semantic(similarity: float, embedder_name: str) -> float:
    """Map a raw cosine similarity onto 0..1 using the embedder's ``SEMANTIC_RANGES``."""
    low, high = SEMANTIC_RANGES.get(embedder_name) or SEMANTIC_RANGES.get(
        embedder_name.partition(":")[0], DEFAULT_SEMANTIC_RANGE
    )
    return min(1.0, max(0.0, (similarity - low) / (high - low)))


def hybrid_score(semantic: float, keyword: float, skill_coverage: float | None) -> float:
    """Weighted blend of the three signals, each in 0..1."""
    if skill_coverage is None:
        total = SEMANTIC_WEIGHT + KEYWORD_WEIGHT
        return (SEMANTIC_WEIGHT * semantic + KEYWORD_WEIGHT * keyword) / total
    return SEMANTIC_WEIGHT * semantic + KEYWORD_WEIGHT * keyword + SKILL_WEIGHT * skill_coverage


class JobMatcher:
    """Finds the resumes in a ``ResumeRAG`` index that best match a job description."""

    def __init__(self, rag: resume_rag.ResumeRAG):
        self.rag = rag

    # --- semantic ---

    def _embed_query(self, job_description: str) -> list[float]:
        if not job_description or not job_description.strip():
            raise ValueError("Job description must be a non-empty string")
        return self.rag.embedder.embed_query(job_description)

    def _semantic_candidates(
        self, query: list[float], resume_paths: list[str] | None = None
    ) -> dict[str, dict]:
        """Nearest chunks grouped by resume, optionally restricted to ``resume_paths``.

        Unrestricted, the ``CHUNK_POOL`` nearest chunks are taken; restricted,
        every chunk of those resumes is scored. Chunk similarity is the cosine
        similarity; a candidate's ``semantic_score`` is
        ``BEST_CHUNK_WEIGHT * best + (1 - BEST_CHUNK_WEIGHT) * mean(top 3)``.
        """
        total_chunks = self.rag.collection.count()
        if total_chunks == 0 or resume_paths == []:
            return {}
        hits = self.rag.collection.query(
            query, n_results=total_chunks if resume_paths else CHUNK_POOL, resume_paths=resume_paths or None
        )

        candidates: dict[str, dict] = {}
        for hit in hits:
            chunk_id, document, record = hit.chunk.id, hit.chunk.document, hit.chunk.metadata
            path = record["resume_path"]
            if path not in candidates:
                candidates[path] = {
                    "resume_path": path,
                    "candidate_name": record["candidate_name"],
                    "metadata": resume_rag.parse_record(record),
                    "chunks": [],
                }
            candidates[path]["chunks"].append(
                {
                    "id": chunk_id,
                    "section": record["section"],
                    "text": _excerpt(document),
                    "similarity": round(hit.similarity, 4),
                }
            )

        for candidate in candidates.values():
            candidate["chunks"].sort(key=lambda c: c["similarity"], reverse=True)
            sims = [c["similarity"] for c in candidate["chunks"]]
            top = sims[:TOP_CHUNKS_FOR_MEAN]
            score = BEST_CHUNK_WEIGHT * sims[0] + (1 - BEST_CHUNK_WEIGHT) * sum(top) / len(top)
            candidate["semantic_score"] = round(score, 4)
        return candidates

    def semantic_search(self, job_description: str, k: int = TOP_K) -> list[dict]:
        """Top-``k`` candidates by embedding similarity alone.

        Returns ``[{"resume_path", "candidate_name", "metadata",
        "semantic_score", "chunks"}]`` best first, where ``chunks`` lists the
        candidate's matched chunks (``id``, ``section``, ``text``, ``similarity``)
        best first.
        """
        query = self._embed_query(job_description)
        candidates = self._semantic_candidates(query)
        ranked = sorted(candidates.values(), key=lambda c: c["semantic_score"], reverse=True)
        return ranked[:k]

    # --- keyword ---

    def keyword_scores(self, parsed: dict) -> dict[str, float]:
        """BM25 score per resume path for a parsed JD, scaled so the best resume is 1.0."""
        texts: dict[str, list[str]] = {}
        skills: dict[str, list[str]] = {}
        for chunk in self.rag.all_chunks():
            path = chunk["metadata"]["resume_path"]
            texts.setdefault(path, []).append(_excerpt(chunk["document"]))
            skills.setdefault(path, list(chunk["metadata"].get("skills", [])))
        if not texts:
            return {}

        paths = list(texts)
        bm25 = BM25Okapi([_resume_tokens(texts[p], skills[p]) for p in paths])
        raw = bm25.get_scores(_query_tokens(parsed))
        best = float(max(raw))  # plain floats, not numpy scalars, so results serialise
        if best <= 0:
            return {p: 0.0 for p in paths}
        return {p: round(max(0.0, float(score)) / best, 4) for p, score in zip(paths, raw)}

    # --- hybrid ---

    def hybrid_search(
        self,
        job_description: str,
        k: int | None = TOP_K,
        parsed: dict | None = None,
        resume_paths: list[str] | None = None,
    ) -> list[dict]:
        """Top-``k`` candidates (all of the pool if ``k`` is None) by the hybrid of
        semantic, keyword and skill signals.

        The pool is every candidate in the semantic chunk pool plus the
        ``KEYWORD_POOL`` best keyword matches; keyword-only candidates get
        their semantic score from a query restricted to their own chunks.
        With ``resume_paths``, the pool is exactly those resumes instead.

        Each result extends the ``semantic_search`` fields with
        ``semantic_norm`` (the semantic score calibrated to 0..1),
        ``keyword_score``, ``matched_skills``, ``missing_skills``,
        ``matched_preferred_skills``, ``skill_coverage`` (``None`` when the JD
        names no skills), ``hybrid_score`` (0..1) and, per chunk,
        ``skills`` (the JD skills that chunk mentions).
        """
        parsed = parsed or parse_job_description(job_description)
        query = self._embed_query(job_description)
        keyword = self.keyword_scores(parsed)
        if resume_paths is not None:
            candidates = self._semantic_candidates(query, list(resume_paths))
        else:
            candidates = self._semantic_candidates(query)
            by_keyword = sorted((p for p in keyword if keyword[p] > 0), key=keyword.get, reverse=True)
            keyword_only = [p for p in by_keyword[:KEYWORD_POOL] if p not in candidates]
            candidates.update(self._semantic_candidates(query, keyword_only))

        jd_skills = parsed["required_skills"] + parsed["preferred_skills"]
        for candidate in candidates.values():
            candidate["keyword_score"] = keyword.get(candidate["resume_path"], 0.0)
            candidate.update(
                _skill_match(
                    parsed["required_skills"],
                    parsed["preferred_skills"],
                    candidate["metadata"]["skills"],
                )
            )
            for chunk in candidate["chunks"]:
                mentioned = set(skills_vocab.with_implied(skills_vocab.find_skills(chunk["text"])))
                chunk["skills"] = [s for s in jd_skills if s in mentioned]
            candidate["semantic_norm"] = round(
                calibrate_semantic(candidate["semantic_score"], self.rag.embedder_name), 4
            )
            candidate["hybrid_score"] = round(
                hybrid_score(
                    candidate["semantic_norm"],
                    candidate["keyword_score"],
                    candidate["skill_coverage"],
                ),
                4,
            )

        ranked = sorted(
            candidates.values(),
            key=lambda c: (c["hybrid_score"], c["semantic_score"]),
            reverse=True,
        )
        return ranked[:k]

    # --- ranking and output ---

    def match(
        self,
        job_description: str,
        k: int = TOP_K,
        must_haves: list[dict] | None = None,
        use_jd_must_haves: bool = True,
        include_filtered: bool = False,
        details: bool = False,
    ) -> dict:
        """Rank candidates for a job description and build the output document.

        ``must_haves`` (e.g. from ``parse_must_have_spec``) are added to the
        ones parsed from the JD, or replace them when ``use_jd_must_haves`` is
        False. Candidates with a requirement ``not_met`` are filtered out
        before taking the top ``k``; ``unverified`` ones (the resume does not
        say) are kept and flagged in the reasoning.

        Returns ``{"job_description", "top_matches": [{"candidate_name",
        "resume_path", "match_score", "matched_skills", "relevant_excerpts",
        "reasoning"}]}``. ``include_filtered`` adds ``"filtered_out"``;
        ``details`` adds the parsed requirements and a score breakdown per match.
        """
        parsed = parse_job_description(job_description)
        requirements = (parsed["must_haves"] if use_jd_must_haves else []) + (must_haves or [])
        parsed["must_haves"] = _dedupe_must_haves(requirements)
        return self.match_parsed(parsed, k=k, include_filtered=include_filtered, details=details)

    def match_parsed(
        self, parsed: dict, k: int = TOP_K, include_filtered: bool = False, details: bool = False
    ) -> dict:
        """``match`` for requirements that are already parsed.

        ``parsed`` has the ``parse_job_description`` shape; its ``text`` is the
        search query and its ``must_haves`` are applied as given. Returns the
        same document as ``match``.
        """
        matches, filtered = [], []
        for result in self.evaluate(parsed):
            if result["failed_requirements"]:
                filtered.append(
                    {key: result[key] for key in
                     ("candidate_name", "resume_path", "match_score", "failed_requirements")}
                )
            elif len(matches) < k:
                matches.append(
                    {key: value for key, value in result.items()
                     if key not in EVALUATION_ONLY_KEYS and (details or key != "details")}
                )

        output = {"job_description": parsed["text"], "top_matches": matches}
        if details:
            output["requirements"] = {
                "required_skills": parsed["required_skills"],
                "preferred_skills": parsed["preferred_skills"],
                "must_haves": parsed["must_haves"],
            }
        if include_filtered or details:
            output["filtered_out"] = filtered
        return output

    def evaluate(self, parsed: dict, resume_paths: list[str] | None = None) -> list[dict]:
        """Score and check every candidate in the pool (or exactly ``resume_paths``).

        Unlike ``match_parsed`` nothing is filtered out: each result is a
        ``match`` entry with ``details``, plus ``"failed_requirements"`` (empty
        when the candidate passes every must-have). Best score first.
        """
        results = []
        for candidate in self.hybrid_search(
            parsed["text"], k=None, parsed=parsed, resume_paths=resume_paths
        ):
            checks = check_must_haves(candidate["metadata"], parsed["must_haves"])
            result = self._format_match(candidate, round(100 * candidate["hybrid_score"]), checks, True)
            result["failed_requirements"] = [c["detail"] for c in checks if c["status"] == "not_met"]
            results.append(result)
        return results

    def _format_match(self, candidate: dict, score: int, checks: list[dict], details: bool) -> dict:
        match = {
            "candidate_name": candidate["candidate_name"],
            "resume_path": resume_rag.display_path(candidate["resume_path"]),
            "match_score": score,
            "matched_skills": candidate["matched_skills"] + candidate["matched_preferred_skills"],
            "relevant_excerpts": select_excerpts(candidate["chunks"]),
            "reasoning": build_reasoning(candidate, score, checks),
        }
        if details:
            meta = candidate["metadata"]
            match["details"] = {
                "semantic_similarity": candidate["semantic_score"],
                "semantic_score": candidate["semantic_norm"],
                "keyword_score": candidate["keyword_score"],
                "skill_coverage": candidate["skill_coverage"],
                "missing_skills": candidate["missing_skills"],
                "must_have_checks": checks,
                "experience_years": meta["experience_years"],
                "highest_degree": meta["highest_degree"],
            }
        return match


# --- must-have requirements ----------------------------------------------------------------------


def parse_must_have_spec(spec: str) -> dict:
    """Parse a CLI must-have: ``"Python:5"`` (5+ years of Python), ``"Python"``
    (has the skill), ``"experience:5"`` (5+ years total), ``"degree:masters"``.
    """
    name, _, value = (part.strip() for part in spec.partition(":"))
    if not name:
        raise ValueError(f"Invalid must-have '{spec}'")
    key = name.lower()
    if key == "degree":
        degree = resume_rag.degree_level(value) or resume_rag.degree_level(value + " degree")
        if not degree:
            raise ValueError(f"Unknown degree '{value}' (use bachelors, masters or phd)")
        return {"type": "degree", "degree": degree, "source": spec}
    years = None
    if value:
        try:
            years = float(value.rstrip("+"))
        except ValueError:
            raise ValueError(f"Invalid years in must-have '{spec}'") from None
    if key in {"experience", "years", "total"}:
        if years is None:
            raise ValueError(f"Must-have '{spec}' needs a number of years")
        return {"type": "total_years", "min_years": years, "source": spec}
    skill = skills_vocab.normalize_skill(name) or name
    if years is None:
        return {"type": "skill", "skill": skill, "source": spec}
    return {"type": "skill_years", "skill": skill, "min_years": years, "source": spec}


def requirement_label(req: dict) -> str:
    if req["type"] == "skill_years":
        return f"{req['min_years']:g}+ yrs {req['skill']}"
    if req["type"] == "total_years":
        return f"{req['min_years']:g}+ yrs experience"
    if req["type"] == "degree":
        return f"{req['degree']} degree"
    return req["skill"]


def check_must_haves(metadata: dict, must_haves: list[dict]) -> list[dict]:
    """Check a candidate against each must-have.

    Each result is ``{"requirement", "status", "detail"}`` with ``status``
    ``"met"``, ``"not_met"``, or ``"unverified"`` when the resume lists the
    skill (or has no experience/education data) but gives no way to check it.
    """
    skills = {s.lower() for s in metadata["skills"]}
    skill_years = {s.lower(): y for s, y in metadata["skill_years"].items()}
    checks = []
    for req in must_haves:
        label = requirement_label(req)
        if req["type"] in {"skill", "skill_years"}:
            skill = req["skill"].lower()
            has = skill in skills or skill in skill_years
            if not has:
                status, detail = "not_met", f"no {req['skill']} on resume (needs {label})"
            elif req["type"] == "skill":
                status, detail = "met", f"has {req['skill']}"
            elif skill not in skill_years:
                # Undated skill: unverified, unless the whole career is too short for it.
                total = metadata["experience_years"]
                if total is not None and total < req["min_years"]:
                    status = "not_met"
                    detail = f"lists {req['skill']} but has {total:g} yrs total experience (needs {label})"
                else:
                    status, detail = "unverified", f"lists {req['skill']} but years unclear (needs {label})"
            else:
                years = skill_years[skill]
                status = "met" if years >= req["min_years"] else "not_met"
                detail = f"{years:g} yrs {req['skill']} (needs {req['min_years']:g}+)"
        elif req["type"] == "total_years":
            years = metadata["experience_years"]
            if years is None:
                status, detail = "unverified", f"experience unclear (needs {label})"
            else:
                status = "met" if years >= req["min_years"] else "not_met"
                detail = f"{years:g} yrs experience (needs {req['min_years']:g}+)"
        else:
            needed = resume_rag.DEGREE_LEVELS[req["degree"]]
            have = metadata.get("degree_level") or 0
            if have == 0 and not metadata["education"]:
                status, detail = "unverified", f"education unclear (needs {label})"
            else:
                status = "met" if have >= needed else "not_met"
                have_text = metadata["highest_degree"] or "no recognised degree"
                detail = f"{have_text} (needs {req['degree']} or higher)"
        checks.append({"requirement": label, "status": status, "detail": detail})
    return checks


# --- excerpts and reasoning ------------------------------------------------------------------------

MAX_EXCERPTS = 3
MAX_EXCERPT_CHARS = 300
# Similarity bonus per JD skill a chunk mentions, so skill-bearing sections are quoted first.
EXCERPT_SKILL_BONUS = 0.02


def _trim(text: str, limit: int = MAX_EXCERPT_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > 0 else limit] + "…"


def _rank_chunks(chunks: list[dict]) -> list[dict]:
    """Chunks by similarity, nudged up for each JD skill they mention."""
    return sorted(
        chunks,
        key=lambda c: c["similarity"] + EXCERPT_SKILL_BONUS * len(c.get("skills", [])),
        reverse=True,
    )


def select_excerpts(chunks: list[dict]) -> list[str]:
    """Up to ``MAX_EXCERPTS`` of the most relevant chunks as ``"[Section] text"``."""
    return [f"[{c['section']}] {_trim(c['text'])}" for c in _rank_chunks(chunks)[:MAX_EXCERPTS]]


def _fit_label(score: int) -> str:
    if score >= 75:
        return "Strong match"
    if score >= 55:
        return "Good match"
    if score >= 35:
        return "Partial match"
    return "Weak match"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def build_reasoning(candidate: dict, score: int, checks: list[dict]) -> str:
    """Explain a match: fit, skill coverage, which sections matched, must-haves."""
    sentences = []
    ranked = _rank_chunks(candidate["chunks"])
    sections = []
    for chunk in ranked:
        if chunk["section"] not in sections:
            sections.append(chunk["section"])
    top_sections = sections[:2]
    sentences.append(
        f"{_fit_label(score)}: closest resume sections are {_join(top_sections)}"
        if top_sections
        else _fit_label(score)
    )

    matched, missing = candidate["matched_skills"], candidate["missing_skills"]
    if candidate["skill_coverage"] is not None:
        text = f"Has {len(matched)}/{len(matched) + len(missing)} required skills"
        if missing:
            text += f" (missing {_join(missing)})"
        if candidate["matched_preferred_skills"]:
            text += f", plus preferred {_join(candidate['matched_preferred_skills'])}"
        sentences.append(text)

    by_section: dict[str, list[str]] = {}
    for chunk in ranked:
        for skill in chunk.get("skills", []):
            found = by_section.setdefault(chunk["section"], [])
            if skill not in found:
                found.append(skill)
    if by_section:
        parts = [f"{section} mentions {_join(skills[:5])}" for section, skills in by_section.items()]
        sentences.append("; ".join(parts[:2]))

    if checks:
        met = [c["detail"] for c in checks if c["status"] == "met"]
        unverified = [c["detail"] for c in checks if c["status"] == "unverified"]
        if met:
            sentences.append(f"Meets must-haves: {'; '.join(met)}")
        if unverified:
            sentences.append(f"Unverified: {'; '.join(unverified)}")

    years = candidate["metadata"]["experience_years"]
    if years is not None:
        sentences.append(f"{years:g} years of total experience")
    return ". ".join(sentences) + "."
