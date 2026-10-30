# Changelog

## Unreleased (1.2)

### Added

- **Local web API** (`resume-shortlister web`): sessions, streamed turns
  (Server-Sent Events), folder loading, resume uploads, candidate reports and
  the report download, served on 127.0.0.1 only, with a per-launch token and a
  Host check.
- **Browser interface** (`resume-shortlister web` opens it): drop resume files
  or pick a folder, describe the role, follow the screening as it runs, confirm
  open requirements, and see the ranked shortlist with its recommendations;
  download the report. Built with React and TypeScript into the package, so
  installing needs no Node.
- `resume-shortlister web --host`: the Docker image serves the web app; publish
  its port on this computer only (`-p 127.0.0.1:8765:8765`).
- **Candidate detail in the browser**: select a shortlisted candidate for the
  recommendation and the rule behind it, how the score adds up, the evidence
  per requirement (with the resume quote), must-have checks, strengths, gaps,
  risk flags and improvement suggestions, and their report as a download.
  **Interview questions** for that candidate are written on request (the
  chat's own tool, so asking again gives the stored answer).
- **Conversation in the browser**: ask about the candidates, compare them or
  change the role in a chat below the shortlist, with each turn's steps,
  suggested follow-ups from the shortlist, and the confirmation card in place
  of the agent's question.
- **Sessions and settings in the browser**: reopen, start or delete a
  screening; see the chat model, whether its key is set, the embeddings, the
  screening rules, where your data is and how many model answers are stored,
  with how to change them (in `.env`: the page never handles API keys).

### Changed

- **Renamed from Profile Match to Resume Shortlister**, a name that says what it
  does. The package is `resume-shortlister` (`resume_shortlister` in Python),
  the command `resume-shortlister`, the data home `~/.resume-shortlister` and
  the settings `RESUME_SHORTLISTER_*`. Upgrading needs no changes:
  - `profile-match` still runs, with a note; it will be removed in a later
    version.
  - `PROFILE_MATCH_*` settings are still read; the new names win when both are
    set.
  - An existing `~/.profile-match` keeps being used (its sessions and indexes
    record absolute paths, so it is not moved). Creating
    `~/.resume-shortlister` starts fresh instead.
  - Sessions saved before the rename reopen with their requirements and
    results.
- The terminal chat and the web app share one `Conversation` that turns the
  agent's steps into events.
- The agent's answers name candidates by name instead of by id.
- A shorter README (what it is, getting started, how it works in brief,
  responsible use, limitations). The details moved, unchanged, into guides:
  `docs/usage.md`, `docs/configuration.md`, `docs/how-it-works.md` and
  `docs/development.md`; a test checks that every link between them resolves.
- The browser app saves reports in `~/.resume-shortlister/reports` (unless
  `REPORTS_DIR` is set) instead of the folder the server was started from, and
  `resume-shortlister web` stops at start with the problem when a setting is
  invalid.

### Fixed

- Progress notes named the deep-reviewed candidate with part of an internal
  key ("Deep analysis of 3f2a…:alex_lee") instead of their id.

## 1.1.0 (2026-10-25)

Provider-agnostic, local-first and reproducible.

### Added

- **Any chat model provider**: OpenAI, Anthropic, Google Gemini, Ollama, or
  any OpenAI-compatible server (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`).
  Provider quirks and error messages are handled in one place.
- **Local embeddings by default**: `ibm-granite/granite-embedding-small-english-r2`
  runs on your machine, chosen by comparing ten models with OpenAI's
  embeddings on the sample resumes (`scripts/ranking_eval.py`) and calibrated to
  the same score scale. No API key is needed to index resumes.
- **Rules decide requirements wherever the job description is explicit**:
  optional and alternative skills, years, total experience and degrees no
  longer depend on how the model read the text (`scripts/requirements_stability.py`).
- **Recruiter confirmation**: in chat, when a requirement names several skills
  and the model kept only some as must-haves, the agent asks before screening.
  One-shot screening lists these under "Please check" in the report.
- **Reproducible screenings**: model answers are stored by everything that
  shapes them, so the same screening gives the same result in every run, with
  an audit trail. `profile-match cache` shows or clears them.
- `LICENSE` (MIT), `SECURITY.md`, a Responsible use section in the README.

### Changed

- Round 3 **recommends interviews** ("Recommend interview" / "Not now")
  instead of making hire / no-hire decisions; people make hiring decisions.
  Saved sessions from 1.0 still load.
- The vector index is **SQLite + NumPy** with exact search, replacing ChromaDB;
  rankings are identical, and the base install dropped from 527 MB to about
  300 MB. Folders re-index automatically; `profile-match clean` removes the old
  ChromaDB files.
- Settings use `LLM_*` and `INDEX_DIR` names; the older `OPENAI_*` and
  `CHROMA_DB_DIR` still work.

### Removed

- OCR. Scanned PDFs and images are reported with a request for a text PDF or
  DOCX instead of being read.

## 1.0.0

First release: a LangGraph agent that screens a folder of resumes against a job
description in three rounds (hybrid retrieval, parallel deep review, a decision
rule), with conversational refinement, explainable reports, guardrails, saved
sessions, a `profile-match` command line and a Docker image.
