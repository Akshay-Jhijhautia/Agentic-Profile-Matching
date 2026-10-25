# Security policy

Profile Match handles candidate data (resumes, contact details) and feeds
untrusted text to a language model, so security reports are welcome.

## Supported versions

| Version | Supported |
| --- | --- |
| 1.1.x | yes |
| < 1.1 | no |

## Reporting a vulnerability

Please **do not open a public issue**. Report it privately through GitHub:
open the repository's **Security** tab and choose **Report a vulnerability**.
Include the version, how to reproduce it and what an attacker gains.

You can expect an acknowledgement within a week. Fixes are released as a new
patch version, and you are credited in the release notes unless you prefer not
to be.

## What counts

Reports are especially useful about:

- **Resume content that steers its own evaluation**: text that changes a
  candidate's verdicts, scores or recommendation without being flagged as a
  prompt-injection risk.
- **Candidate data leaving the machine** other than the resume text sent to
  the configured chat model, or being written somewhere other than the data
  home (`~/.profile-match`) and the reports directory.
- **File access outside its limits**: the agent's `write_file` tool reaching
  outside the reports directory, or file tools reading through crafted paths.
- **Keys** (API keys) appearing in logs, reports, stored answers or errors.
- **Vulnerable dependencies** that are reachable from the tool.

Out of scope: ranking quality and model disagreement on borderline candidates
(see the README's Limitations), and problems that need an attacker who already
controls the machine or the `.env` file.
