# Configuration

Settings come from the environment and `.env` files ([all of them](#settings)); the browser app's Settings panel shows the ones in effect.

## Chat model providers

One chat model runs every LLM step: requirement extraction, deep review,
rationales, the assistant and the indexer's metadata fallback. Pick it with
`LLM_PROVIDER` and `LLM_MODEL`:

| `LLM_PROVIDER` | Install | Key | Default `LLM_MODEL` |
| --- | --- | --- | --- |
| `openai` (default) | included | `OPENAI_API_KEY` | `gpt-5.6-terra` |
| `anthropic` | `pip install 'resume-shortlister[anthropic]'` | `ANTHROPIC_API_KEY` | `claude-opus-5-5` |
| `google` | `pip install 'resume-shortlister[google]'` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `gemini-3.1-pro` |
| `ollama` | `pip install 'resume-shortlister[ollama]'` | none | none: set it, e.g. `qwen3:32b` |

Any OpenAI-compatible server (vLLM, LM Studio, OpenRouter, xAI, ...) works
through `openai` with `LLM_BASE_URL`; put its key, if it needs one, in
`LLM_API_KEY`, so your `OPENAI_API_KEY` is never sent to it:

```bash
LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=llama-3.3-70b resume-shortlister chat
```

Provider differences are handled in `resume_shortlister/providers.py`: Claude
gets no temperature and uses its native structured outputs, OpenAI's reasoning
models use the Responses API, and errors name the provider and the setting to
fix. Embeddings do not depend on the chat provider: they are computed locally
(see [Embeddings](#embeddings)), so Anthropic, Gemini or Ollama need no OpenAI
key. Only OpenAI has been tested end to end with live calls; the other
providers are covered by offline tests.

## Embeddings

Resumes and searches are embedded by an open-source model running on your
machine: `ibm-granite/granite-embedding-small-english-r2` (Apache-2.0, 180 MB,
downloaded once into `~/.resume-shortlister/models`). No API key, nothing leaves the
machine, and the same resumes get the same vectors on every run. Indexing 100
resumes takes about 5 seconds on a laptop CPU.

It was chosen by measurement, not benchmark tables. Ten local models were
compared with OpenAI's `text-embedding-3-small` on the 100 sample resumes and 16
queries (`scripts/ranking_eval.py`):

| Embeddings | P@10 | nDCG@10 | AUC | semantic AUC |
| --- | --- | --- | --- | --- |
| OpenAI `text-embedding-3-small` | 0.908 | 0.998 | 0.995 | 0.995 |
| **granite-embedding-small-english-r2** (default) | **0.921** | **0.998** | **0.995** | 0.992 |
| snowflake-arctic-embed-m (`LOCAL_EMBEDDING_MODEL`) | 0.915 | 0.998 | 0.994 | 0.993 |
| bge-small-en-v1.5, bge-base, gte-base, nomic, mxbai-large, MiniLM, potion | 0.896–0.908 | ≤ 0.997 | ≤ 0.991 | 0.966–0.989 |

Each model's similarities are mapped onto OpenAI's score scale
(`job_matcher.SEMANTIC_RANGES`, fitted on the candidates who reach the
shortlist), so the interview threshold keeps its meaning. Only calibrated models can
be selected. In a live comparison on 17 job descriptions, with requirements and
deep reviews held fixed, 67 of 69 recommendations were the same; the two that
differed sat on the threshold (final 75–76 with OpenAI, 73–74 locally), the
kind of candidate a second run with the same model can also flip.

`EMBEDDER=openai` uses OpenAI embeddings instead (needs `OPENAI_API_KEY`), and
`EMBEDDER=hashing` a keyword-only embedder for tests. A folder indexed with
other embeddings is re-indexed automatically the next time it is used.

## Settings

| Variable | Default | Used for |
| --- | --- | --- |
| `LLM_PROVIDER` / `LLM_MODEL` | `openai` / provider default | The chat model (see above) |
| `OPENAI_API_KEY` | — | OpenAI chat model (and OpenAI embeddings, if chosen) |
| `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` | — | The other providers' chat models |
| `LLM_BASE_URL` / `LLM_API_KEY` | — | An OpenAI-compatible server or a remote Ollama server, and its key |
| `LLM_TEMPERATURE` | provider default | Only for models that accept one (not Claude) |
| `LLM_TIMEOUT` / `LLM_MAX_RETRIES` | `60` / `2` | Request limits |
| `OPENAI_RESPONSES_API` | `true` (`false` with `LLM_BASE_URL`) | OpenAI's Responses API, required for tool calling with reasoning models |
| `EMBEDDER` | `local` | Resume embeddings: `local`, `openai`, or `hashing` (keyword overlap only, for tests) |
| `LOCAL_EMBEDDING_MODEL` | `ibm-granite/granite-embedding-small-english-r2` | The local model; `snowflake/snowflake-arctic-embed-m` is the other calibrated choice |
| `MODELS_DIR` | `~/.resume-shortlister/models` | Where the local model is downloaded |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | With `EMBEDDER=openai` |
| `RESUME_SHORTLISTER_HOME` | `~/.resume-shortlister` | Local data: indexes, uploads, sessions, `.env` |
| `INDEX_DIR` / `UPLOADS_DIR` / `SESSIONS_DB` | under `RESUME_SHORTLISTER_HOME` | Override one location (`CHROMA_DB_DIR` still works for `INDEX_DIR`) |
| `REPORTS_DIR` | `reports` (the browser app: `~/.resume-shortlister/reports`) | Where reports are saved; the only directory the agent may write to |
| `SHORTLIST_SIZE` | `10` | Candidates kept after round 1 |
| `SCREENING_ROUNDS` | `3` | 1: retrieval only; 2: + deep analysis; 3: + interview recommendations |
| `SAVE_REPORTS` | `true` | Save reports after every screening |
| `LLM_CACHE` / `LLM_CACHE_DB` | `true` / `~/.resume-shortlister/llm_cache.sqlite` | Reuse the model's answer to a question asked before |

Settings come from the environment, then `./.env`, then
`~/.resume-shortlister/.env`. The older `OPENAI_MODEL`, `OPENAI_TEMPERATURE`,
`OPENAI_TIMEOUT` and `OPENAI_MAX_RETRIES` still work for the OpenAI provider.
`resume-shortlister index` and `list` need no API key at all; screening and chat
need a chat model (a local Ollama model works without one).
