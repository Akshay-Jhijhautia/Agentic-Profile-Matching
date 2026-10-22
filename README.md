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
| `main.py` | Single CLI entry point |
| `resumes/` | Demo resume dataset |
| `job_descriptions/` | Sample job descriptions |
| `scripts/generate_sample_resumes.py` | Regenerates `resumes/` |
| `tests/` | Offline pytest suite; `tests/fixtures/resumes/` is a frozen copy of the 12 Milestone 2 samples so expected rankings stay stable as `resumes/` grows |

## Tests

```bash
pytest -q
```
