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
| `OPENAI_MODEL` | `gpt-4o-mini` | Requirement extraction and agent reasoning |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Resume and query embeddings |
| `OPENAI_TEMPERATURE` | `0` | Chat temperature (empty = provider default) |
| `OPENAI_TIMEOUT` / `OPENAI_MAX_RETRIES` | `60` / `2` | Chat request limits |
| `CHROMA_DB_DIR` | `chroma_db` | Vector index location |

## Usage

```bash
python main.py index resumes                    # build or refresh the vector index
python main.py index resumes --embedder hashing # offline index (keyword-overlap embeddings)
python main.py stats | list                     # inspect the index
python main.py match --jd-file job_descriptions/frontend_react.txt   # non-agent ranking
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
   TensorFlow", "Oracle or PostgreSQL") are demoted to nice-to-have, and a
   degree is dropped unless a requirement line demands it without "or
   equivalent", "preferred" and the like.

Skills are normalised through the shared vocabulary (`react.js` → `React`),
and durations or degrees listed as skills are discarded. Ranking reuses the
Milestone 2 matcher through `JobMatcher.match_parsed(requirements.to_parsed())`.
Its search text is built from the requirements, so a short request searches
like a full job description.

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
