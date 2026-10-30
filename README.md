# Resume Shortlister

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Status: developer preview](https://img.shields.io/badge/status-developer%20preview-orange.svg)](#limitations)

**An open-source, explainable resume-screening assistant. It ranks and
explains; the recruiter decides.**

*Formerly Profile Match: the old command, settings and data folder still work
(see the [changelog](CHANGELOG.md)).*

Point it at a folder of resumes and describe the role, in one sentence or a
full job description. Resume Shortlister screens every resume in three rounds,
recommends whom to interview, and shows the evidence behind every score. Then
ask about the candidates, compare them or change the role, in your browser or
in the terminal.

![The shortlist and the conversation in the browser](docs/images/web-shortlist.png)

## Why it is different

- **The model judges, code decides.** The language model reads resumes and
  quotes evidence; scores, knockouts and recommendations come from fixed,
  auditable rules.
- **No invented experience.** Every positive verdict needs a quote that is
  actually in the resume, or it is downgraded.
- **Blind to who the candidate is.** The model judges anonymous candidates:
  no name, contact details or personal fields, and no name in the search.
- **Same input, same result.** Rules settle what the job description states,
  you confirm what it leaves open, code turns the evidence into verdicts, and
  stored model answers make a repeated screening identical. Even with fresh
  model answers, repeated runs and two different models recommend the same
  candidates ([measured](docs/how-it-works.md#guardrails)).
- **Every score is explained**: its breakdown, the evidence, strengths, gaps,
  and what would make a borderline candidate a clear interview.
- **Yours to run.** OpenAI, Anthropic, Gemini, Ollama or any OpenAI-compatible
  server. Embeddings, index and data stay on your machine; with Ollama, nothing
  leaves it.
- **Measured, not claimed.** Ranking quality, run-to-run stability and model
  agreement are measured with scripts in this repository, alongside 643 tests.

**Status:** developer preview (v1.1; the browser interface,
`resume-shortlister web`, is new on the main branch and comes with 1.2).
Validated on synthetic resumes only, so read [Responsible use](#responsible-use)
and [Limitations](#limitations) before relying on it.

## Get started

Requires Python 3.11+ and a chat model: OpenAI by default, or Anthropic,
Google Gemini, a local Ollama model or any OpenAI-compatible server
([providers](docs/configuration.md#chat-model-providers)).

```bash
pip install "resume-shortlister @ git+https://github.com/Akshay-Jhijhautia/resume-shortlister.git"
export OPENAI_API_KEY=sk-...        # or put it in ~/.resume-shortlister/.env

resume-shortlister web              # the app in your browser, on this computer only
```

Drop resumes (PDF, DOCX, TXT or MD) on the page or give a folder, describe the
role, and follow the screening as it runs. Select a candidate for the evidence
behind their place and interview questions; ask follow-ups in the conversation.
The first screening downloads the local embedding model once (180 MB).

Or in the terminal:

```bash
resume-shortlister chat --resumes ~/Downloads/applicants
you> Find me candidates with React and 3+ years experience
you> Compare the top 3 matches side by side
you> Make TypeScript a must-have
you> Write 3 interview questions for the top candidate

resume-shortlister screen --jd-file job.pdf --resumes ~/Downloads/applicants   # one shot
```

Every command, your resumes, where your data lives and Docker:
[Usage](docs/usage.md).

## How it works

1. **Initial screen.** Every resume gets a search score (semantic, keyword and
   must-have coverage) and the must-have checks; the best 10 that pass go on.
   No model calls.
2. **Deep review.** The model reads each shortlisted resume and gives a verdict
   per requirement, with a quote from the resume. Code checks the quotes and
   computes the score.
3. **Recommendation.** A fixed rule recommends an interview or not now; the
   model only writes the rationale. Borderline candidates get what would make
   them a clear interview.

Requirements the job description leaves open are confirmed with you before
screening, and stored model answers make a repeated screening identical.
The details, with measurements: [How it works](docs/how-it-works.md).

## Documentation

| Guide | What it covers |
| --- | --- |
| [Usage](docs/usage.md) | The browser app, the terminal chat and every command, your resumes, local data, Docker, a sample session |
| [Configuration](docs/configuration.md) | Chat model providers, embeddings, every setting |
| [How it works](docs/how-it-works.md) | The agent graph, requirements, the three rounds, guardrails, reproducibility, refinement, reports, tools, design decisions |
| [Development](docs/development.md) | Working on the code: setup, project layout, sample data, tests and measurements |
| [Changelog](CHANGELOG.md) | Changes per version |

## Responsible use

Resume Shortlister assists people who screen resumes; it does not make hiring
decisions, and it should not be used to reject candidates without a person
reviewing the result.

- **People decide.** The tool recommends whom to interview ("Recommend
  interview" / "Not now") with its evidence. A recruiter reviews every
  recommendation, especially candidates near the threshold, where small
  differences in wording or model can change the outcome.
- **Explain, don't hide.** Every score comes with its breakdown, the resume
  lines behind each verdict and the rule that produced the recommendation, so
  any result can be questioned and checked.
- **Bias.** Screening is blind: the model never sees a candidate's name,
  contact details or personal fields, and the search ignores them
  ([details](docs/how-it-works.md#blind-screening)). Employment gaps and short
  roles are listed for you to ask about, never scored. Other signals remain,
  such as schools, graduation years and locations in the work history, and the
  scoring has only been validated on synthetic resumes. Do not rely on it for
  decisions until you have checked it on your own, representative data.
- **Regulation.** Software that assists hiring decisions is regulated in some
  places: the EU AI Act treats it as high-risk, and some jurisdictions (for
  example New York City) require bias audits before automated screening tools
  are used. Check what applies to you; this project provides no compliance
  guarantees.
- **Candidate data.** Resumes, the index, saved sessions and stored model
  answers stay on your machine under `~/.resume-shortlister`. Resume text is sent to
  the chat model you configure; use a local model (Ollama) if it must not leave
  your machine. `resume-shortlister clean` and `resume-shortlister cache --clear` delete
  what is no longer needed.
- **Candidates can game resumes.** Instruction-like text is flagged, but treat
  any result as a starting point for a conversation, not a verdict.

## Limitations

- **Embedding models disagree on details.** Calibration aligns score scales,
  but per candidate the local model and OpenAI differ by about 5 match points
  on average (2 points of final score), more in some fields. Rankings and
  recommendations agreed in the
  [comparison](docs/configuration.md#embeddings); candidates on the threshold can
  go either way, as they can between two runs of the same model.
- **Only OpenAI's models have been measured.** gpt-5.6-terra and gpt-6-luna
  agreed on every recommendation for three job descriptions (see the
  [stability check](docs/how-it-works.md#guardrails)); Anthropic, Gemini and
  Ollama models have not been compared. A candidate right at the threshold
  can still go either way, and a requirement naming several skills is the
  model's call until you confirm it.
- **Heuristic skill years.** A skill's years come from the dated roles that
  mention it, so skills listed only in a Skills section are "unverified".
- **A skill written an unusual way can be missed.** Must-haves are checked by
  name; the vocabulary knows the common other spellings, but not every one
  ("Windows 2000, 2003 Server" is not read as Windows Server), so look at the
  closest knocked-out candidates in the report.
- **Blind screening covers identity, not every proxy.** Names, contact details
  and personal fields are removed; schools, graduation years, locations in the
  work history, affiliations and pronouns are not. A name part that is also a
  skill or a month ("Ruby", "May") is hidden only as part of the full name.
  The chat assistant, which talks to you rather than scoring anyone, and the
  file tools read resumes as written.
- **Synthetic data.** The embedding model was chosen, and its score range
  calibrated, on the generated sample resumes; real resumes may need both
  re-checked with `scripts/ranking_eval.py`.
- **Unsupported formats.** Legacy `.doc`, `.rtf`, `.odt`, scanned PDFs and
  images aren't read; they are reported so they can be converted. Subfolders of a resume folder aren't
  read either.
- **Uploads accumulate** until `resume-shortlister clean` removes them.

## License and security

MIT, see [LICENSE](LICENSE). Report vulnerabilities privately as described in
[SECURITY.md](SECURITY.md). Changes per version are in [CHANGELOG.md](CHANGELOG.md).
