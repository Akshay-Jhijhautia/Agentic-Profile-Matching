# Agentic Profile Matching

Milestone 3 of the resume assistant: a LangGraph agent that screens resumes
against a job description, explains its rankings and refines them through
conversation. It builds on the Milestone 1 file tools and the Milestone 2
RAG matcher, which live in `profile_matching/rag/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your OPENAI_API_KEY in .env
```

| Variable | Default | Used for |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | Chat model and embeddings |
| `OPENAI_MODEL` | `gpt-6-luna` | Requirement extraction, agent reasoning and the indexer's metadata fallback |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Resume and query embeddings |
| `OPENAI_TEMPERATURE` | provider default | Chat temperature; only set it for models that accept one |
| `OPENAI_TIMEOUT` / `OPENAI_MAX_RETRIES` | `60` / `2` | Chat request limits |
| `CHROMA_DB_DIR` | `chroma_db` | Vector index location |

## Usage

```bash
python main.py index resumes                    # build or refresh the vector index
python main.py index resumes --embedder hashing # offline index (keyword-overlap embeddings)
python main.py stats | list                     # inspect the index
python main.py match --jd-file job_descriptions/frontend_react.txt   # non-agent ranking

# The matching agent (one-shot screening of the whole index)
python matching_agent.py --jd-file job_descriptions/frontend_react.txt
python main.py screen "Find me candidates with React and 3+ years experience" --out reports/react.md
```

## Project layout

| Path | Purpose |
| --- | --- |
| `profile_matching/rag/` | Retrieval engine from Milestones 1–2: `fs_tools`, `resume_rag`, `job_matcher`, `skills_vocab` |
| `profile_matching/config.py` | Settings read once from the environment |
| `profile_matching/llm.py` | The single path to the chat model: `structured(schema, system, user)` |
| `profile_matching/schemas.py` | Pydantic models shared by the LLM layer, tools and graph (`Requirements`, ...) |
| `profile_matching/prompts.py` | All prompt templates |
| `profile_matching/requirements.py` | `extract_requirements`: LLM extraction + rule floor + knockout policy |
| `profile_matching/services.py` | `Services`: the index, matcher, candidate directory, settings and chat model shared by tools and graph |
| `profile_matching/candidates.py` | `CandidateDirectory`: resolve candidates by id, name or part of a name |
| `profile_matching/analysis.py` | Head-to-head comparison and interview questions (facts + LLM reasoning) |
| `profile_matching/tools/` | The agent's tools, built per `Services` by factories (`make_tools`) |
| `matching_agent.py` | The LangGraph agent: graph wiring (`build_graph`) and one-shot CLI |
| `profile_matching/agent/` | Agent state (`AgentState`, checkpointer) and graph nodes |
| `profile_matching/screening.py` | Screening rounds: score every resume, rank, shortlist |
| `profile_matching/reports.py` | Markdown shortlist report and chat summary |
| `main.py` | Single CLI entry point |
| `resumes/` | Demo dataset: 100 resumes (12 hand-written + 88 synthetic) in PDF, DOCX and TXT |
| `job_descriptions/` | Sample job descriptions |
| `scripts/generate_sample_resumes.py` | Regenerates `resumes/` |
| `scripts/synthetic_resumes.py` | Seeded, template-driven resume builder used by the generator |
| `tests/` | Offline pytest suite; `tests/fixtures/resumes/` is a frozen copy of the 12 Milestone 2 samples so expected rankings stay stable as `resumes/` grows |

## Requirements extraction

`extract_requirements(text)` turns a job description or a short request
("Find me candidates with React and 3+ years experience") into `Requirements`:
must-have skills (optionally with minimum years), nice-to-have skills, minimum
total experience, minimum degree and an ideal-candidate profile.

Must-haves are **knockout filters**: a candidate without one is rejected. So
the LLM's answer passes through two deterministic steps:

1. **Recall floor.** The Milestone 2 rule parser adds vocabulary skills the
   LLM skipped (as nice-to-have only) and years it found for a must-have the
   LLM left without them.
2. **Knockout policy.** Skills offered as alternatives ("PyTorch or
   TensorFlow", "Oracle or PostgreSQL") and broad concepts the resume parser
   cannot check ("Relational Databases", "Model Deployment") are demoted to
   nice-to-have, and a degree is dropped unless a requirement line demands it
   without "or equivalent", "preferred" and the like.

Skills are normalised through the shared vocabulary (`react.js` → `React`),
and durations or degrees listed as skills are discarded. Ranking reuses the
Milestone 2 matcher through `JobMatcher.match_parsed(requirements.to_parsed())`.
Its search text is built from the requirements, so a short request searches
like a full job description.

## The matching agent

```
START → parse_jd → extract_requirements → search_resumes → rank_candidates → generate_report → END
```

| Node | What it does |
| --- | --- |
| `parse_jd` | Takes the latest user message: JD text, a short request or a JD file path |
| `extract_requirements` | Must-have vs nice-to-have (see below); every version is kept in `requirements_history` |
| `search_resumes` | Round 1: scores and checks **every** indexed resume (not a retrieval pool) |
| `rank_candidates` | Candidates passing every must-have, best score first; keeps the top `SHORTLIST_SIZE` |
| `generate_report` | Markdown report (requirements, ranking table, reasoning, closest knocked-out candidates) and a chat summary |

The state (`profile_matching/agent/state.py`) holds the conversation
(`messages`), the job understanding (`job_description`, `requirements`,
`requirements_history`) and the results (`candidates` with every resume
ranked, the `shortlist` with each candidate's reasoning, `previous_shortlist`,
`report`). `make_checkpointer()` registers the state's Pydantic types with
LangGraph's serializer, so checkpoints restore without deserialization
warnings.

A skill listed without dated experience is `unverified` (kept, flagged), unless
the candidate's whole career is shorter than the years required, in which case
it is `not_met`.

## Agent tools

`make_tools(services)` builds every tool the agent can call. Each tool takes
its description and argument docs from its docstring. Expected failures, such
as an unknown candidate or an unreadable file, are returned to the model as
`{"success": false, "error": ...}` so it can recover.

| Tool | Purpose |
| --- | --- |
| `read_file`, `list_files`, `search_in_file` | Milestone 1 file tools (read anywhere) |
| `write_file` | Save Markdown/text, only inside `REPORTS_DIR` |
| `rag_search(query, k)` | Milestone 2 hybrid search for open questions |
| `extract_requirements(jd)` | Must-have vs nice-to-have from JD text, a request or a JD file |
| `compare_candidates(candidate_ids)` | 2–5 candidates head to head: scores, must-have checks, LLM verdict |
| `generate_interview_questions(candidate_id, count)` | Questions aimed at the candidate's gaps, unverified claims and strengths |

Candidates are referred to by id (the resume file stem without `resume_`, e.g.
`maria_garcia`), full name or a unique part of a name. An ambiguous reference
("Rafael") lists the matching ids instead of guessing. The comparison and
interview tools read the current role from the agent's state, so the model
never passes requirements around. Their deterministic facts (scores,
must-have checks) are returned with the LLM's reasoning, so every claim can be
traced back.

## Dataset

`resumes/` holds 100 fictional resumes: the 12 hand-written Milestone 2 samples
plus 88 synthetic ones from `scripts/synthetic_resumes.py`. Synthetic resumes
are built from 11 role families (frontend, full-stack, Python/Java backend,
data science, ML, data engineering, DevOps, iOS, Android, QA) at junior to
lead seniority, with overlapping skills, varied header and date formats, and
some edge cases (career changers, no degree, certifications). The same seed
always produces the same resumes.

```bash
python -m scripts.generate_sample_resumes                  # 100 resumes into resumes/
python -m scripts.generate_sample_resumes out/ --count 250 --seed 7
```

## Tests

```bash
pytest -q
```
