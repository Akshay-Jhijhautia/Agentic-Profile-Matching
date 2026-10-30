# Development

Working on the code: setup, where things are, sample data, and how the tests and measurements run.

## Setup

```bash
git clone --depth 1 https://github.com/Akshay-Jhijhautia/resume-shortlister.git
cd resume-shortlister
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # editable install with dev tools and every provider
```

From a source checkout, `python main.py ...` runs the same commands as
`resume-shortlister ...`. The browser interface's source is in `frontend/`
([frontend/README.md](../frontend/README.md)); its build is committed, so
working on the Python code needs no Node.

## Project layout

| Path | Purpose |
| --- | --- |
| `pyproject.toml` | Package metadata, dependencies (provider and `dev` extras) and the `resume-shortlister` command |
| `resume_shortlister/app.py` | The `resume-shortlister` command line |
| `resume_shortlister/agent/graph.py` | The LangGraph graph (`build_graph`, `screen`) |
| `matching_agent.py`, `main.py` | Entry points for a source checkout (re-export the graph; run the CLI) |
| `resume_shortlister/sessions.py` | Saved chat sessions in SQLite |
| `resume_shortlister/maintenance.py` | Finding and removing old uploads, orphaned indexes and old sessions |
| `Dockerfile` | Container image (non-root, data in a volume) |
| `resume_shortlister/pools.py` | One resume pool per folder: indexing, incremental sync, intake report |
| `resume_shortlister/agent/` | `graph.py`, `state.py` (state, checkpointer), `nodes.py`, `routing.py`, `text.py` |
| `resume_shortlister/screening.py` | The three screening rounds |
| `resume_shortlister/guardrails.py` | Review items, evidence grounding, verdict bounds, rubric, decision rule, injection checks |
| `resume_shortlister/requirements.py` | Requirement extraction: the model proposes, rules decide where the text is explicit |
| `resume_shortlister/changes.py` | Requirement and ranking diffs with a reason for every move |
| `resume_shortlister/explainability.py` | Per-candidate score breakdown, strengths and gaps, detailed report, saving |
| `resume_shortlister/reports.py` | Shortlist report, chat summary, assistant context |
| `resume_shortlister/analysis.py` | Comparison, interview questions and report lookup behind the tools |
| `resume_shortlister/tools/` | Agent tools, built per `Services` by factories |
| `resume_shortlister/conversation.py` | A conversation with the agent as a stream of events, shared by the terminal and the web app |
| `resume_shortlister/cli.py` | Rich terminal chat: prints a conversation's events |
| `resume_shortlister/web/` | The local web app: FastAPI API (`app.py`), open sessions (`workspace.py`), response shapes (`models.py`), the built UI (`static/`) |
| `frontend/` | The browser UI's source (React, TypeScript, Vite); see `frontend/README.md` |
| `resume_shortlister/llm.py` | The single path to the chat model: `structured(...)`, `with_tools(...)` |
| `resume_shortlister/llm_cache.py` | Stored model answers: the same question always gets the same answer, with an audit trail |
| `resume_shortlister/providers.py` | Chat model providers: how each is built, its key, its quirks, its error messages |
| `resume_shortlister/schemas.py` | Pydantic models shared across the agent |
| `resume_shortlister/prompts.py` | All prompt templates |
| `resume_shortlister/services.py`, `candidates.py`, `config.py`, `markdown.py` | Shared dependencies, candidate lookup, settings, Markdown helpers |
| `resume_shortlister/rag/` | Retrieval engine: `fs_tools` (read, list, search files), `resume_rag` (section chunking, metadata extraction, indexing), `vector_store` (SQLite records, exact cosine search with NumPy), `job_matcher` (JD parsing, hybrid search, must-have checks), `skills_vocab` (skill aliases) |
| `scripts/` | Sample resume generator, stability checks (screening, requirements), ranking evaluation (development tools, not part of the product) |
| `tests/` | Offline pytest suite; `tests/fixtures/` holds 12 sample resumes and 7 sample JDs |

## Sample data

The repository ships no resumes. For demos and evaluation,
`scripts/generate_sample_resumes.py` writes fictional resumes to any folder:
- 12 hand-written samples (also the test fixtures);
- seeded synthetic ones from `scripts/synthetic_resumes.py`, up to `--count`.

The synthetic resumes cover:
- **11 role families**: frontend, full-stack, Python and Java backend, data science, ML, data engineering, DevOps, iOS, Android and QA;
- **seniority** from junior to lead, with overlapping skills;
- **varied formats**: different header and date styles;
- **edge cases**: career changers, missing degrees, certifications.

```bash
python -m scripts.generate_sample_resumes /tmp/sample_resumes             # 100 resumes
python -m scripts.generate_sample_resumes /tmp/more --count 250 --seed 7
resume-shortlister chat --resumes /tmp/sample_resumes
```

The same seed always produces the same resumes. The measurements and the
sample session in these docs use the default 100.

## Testing

```bash
pytest -q
```

The suite (over 550 tests) runs offline. A fake chat model answers structured
calls and scripts tool-calling turns, and the index is a temporary one built
with a hashing embedder. It covers:
- the retrieval engine (file reading, chunking, metadata extraction, indexing, hybrid search);
- the vector store (exact nearest neighbours against brute force, persistence, concurrent use);
- requirement extraction and the rules (different model readings of the same JD must agree);
- every guardrail;
- the three rounds;
- refinement diffs;
- reports;
- the tools;
- full multi-turn conversations through the real graph, including interrupts, refinements and error recovery;
- the web API (sessions, streamed turns, uploads, candidate details, the token and Host checks), and that the committed browser build matches its source.

The browser interface has its own tests (Vitest and Testing Library):
`npm run typecheck && npm test` in `frontend/`.

Ranking quality is measured separately, on the 100 sample resumes, whose role
families give relevance labels. Requirements are fixed (no LLM), so only the
retrieval stack varies:

```bash
python -m scripts.ranking_eval run --embedder openai:text-embedding-3-small --out a.json --cache emb.json
python -m scripts.ranking_eval compare a.json b.json
```

It reports P@10, nDCG@10 and AUC (how often a relevant candidate outscores an
irrelevant one) for 16 queries, and compares two runs candidate by candidate.
Replacing ChromaDB with the current store changed nothing: identical
knockouts, rankings and scores for all 100 candidates in every query.
