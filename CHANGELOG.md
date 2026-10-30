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
- **Blind screening**: the model judges anonymous candidates. In the deep
  review, rationales, improvement suggestions, comparisons and interview
  questions, it sees "Candidate 1" instead of names and ids, and resumes
  without name, contact details or personal fields; its answers are mapped
  back to names. The search no longer embeds or keyword-matches the name or
  personal details. Folders are re-indexed once, and earlier stored model
  answers are not reused (the questions changed).
- **Stable deep reviews.** Where a skill's evidence is decides its verdict:
  named in a role or project is strong (code finds the line), only listed or
  claimed is partial. Risk flags come in three kinds (a stated figure the dates
  do not support, a contradiction, repeated content), each needing a quote
  found in the resume, counted once, at the severity of its kind; only
  code-detected risks are high. Over 5 fresh runs on three job descriptions,
  the average deep-score spread fell from 9–15 points to under 4, and interview
  recommendations agreed in every run.
- `resume-shortlister clean --all`: removes every piece of candidate data (every
  index, uploaded copy, saved session, stored model answer and the web app's
  reports) to start from scratch, keeping the settings and the embedding model.
  It lists first and deletes with `--yes`, and only the tool's own files.
- **Requirements for any profession**: a skill the vocabulary does not know
  (Workday, TCP/IP, Ceridian Dayforce) becomes a requirement when the job
  description names it, is searched for in resumes as written, and gets its
  years from the dated roles that name it. A name can be a must-have; a
  concept the text writes in lower case ("month-end close", "relational
  databases") stays a nice-to-have, since resumes paraphrase it. The
  vocabulary learns the common tools' other spellings found in real resumes
  (Windows Server versions, vSphere for VMware, ASP.NET for .NET, Microsoft
  Office for Excel, GAAP spelled out). On 684 real resumes, the IT
  administrator and accountant roles, which had no must-haves (over 80% of
  applicants passed), now have their own.
- **The same requirements from every model**: a non-skill preference counts
  only when the job description says it in those words, spelled as it does;
  the skills vocabulary knows BigQuery, Kinesis, Argo CD, CKA and
  Accessibility (WCAG), and practices and certifications are never knockouts.
  Across gpt-5.6-terra and gpt-6-luna, requirements came out identical and
  every run recommended the same candidates on three job descriptions.
  Folders are re-indexed once.
- **Timeline notes, not scored**: gaps of 6 months or more between roles and
  short completed roles (not internships or contracts) are computed from the
  dates and listed in the candidate's report and detail, for the recruiter to
  ask about. They no longer lower a score.
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

Found by screening 684 real resumes (LiveCareer examples) for the first time:

- A job title or heading at the top of a resume ("INFORMATION TECHNOLOGY
  MANAGER", "CAREER OVERVIEW") was read as the candidate's name, and blind
  screening then hid those words from the model. Such lines are no longer
  names; a resume without one is named after its file.
- Indexing asked the model for the name of every resume without one: 212
  calls and 11 minutes for 219 resumes. It no longer asks for names (90
  seconds), and what it does ask (dates, education) goes without the name and
  contact details.
- Ligatures and other compatibility characters in PDF text ("certiﬁed",
  "Microsoft Ofﬁce", non-breaking spaces) are read as plain letters, so skills
  and keywords match (85% of the real PDFs had them).
- Several processes starting at once (the web app and a terminal screening)
  could fail with "database is locked" while creating a new database, or with
  "UNIQUE constraint failed" while indexing the same new folder (9 of 24
  processes in a reproduction; now 0).
- A new index version re-indexed only files whose content had changed: unchanged
  files kept their old skills and metadata. The version is now part of the file
  stamp, so every folder is re-indexed once.
- A skill name broken across a line in PDF text ("generally accepted /
  accounting principles") is now found.
- `Settings(home=...)` kept the index, sessions, uploads, models and stored
  answers in the default data home; every path not set on its own now follows
  `home`.
- `scripts/stability_check.py` and `scripts/requirements_stability.py` stop
  with a one-line error when the provider refuses (no credits, rate limit)
  instead of a traceback.
- A name that is also a skill counted as one: "Ruby Swift" was credited with
  Ruby and Swift from her name line, passing must-haves she does not show.
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
