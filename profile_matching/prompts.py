"""Prompt templates, kept in one place so wording changes never touch logic."""

EXTRACT_REQUIREMENTS = """\
You extract hiring requirements from a job description or from a recruiter's \
short request such as "Find me candidates with React and 3+ years experience".

Must-have skills reject every candidate who lacks them, so keep them to the \
deal-breakers, usually two to five:
- must_have: skills the text states as essential (listed under requirements / \
must have / what you'll need, or asked for directly in a request). When one \
requirement line lists several technologies ("deploying models with Docker and \
Kubernetes on AWS"), only the skill the line is about is a must-have; the \
supporting ones go to nice_to_have.
- nice_to_have: skills described as preferred, a plus, bonus or nice to have; \
skills phrased softly ("familiarity with", "exposure to"); and every option of \
an alternative ("Jest or Cypress" -> both go here).
- min_years on a skill only when years are tied to it ("5+ years of Python"). \
In a short request that pairs one skill with years ("React and 3+ years \
experience"), tie the years to that skill. For a range ("3-5 years") use the \
lower bound.
- min_total_years: years of general professional experience, else null.
- min_degree: the lowest acceptable degree when one is required; null when it \
is preferred, optional or "or equivalent experience".

Use short canonical skill names ("React", "PostgreSQL", "Machine Learning"), one \
skill per entry. Degrees are never skills: they only go in min_degree. Skip \
soft skills (communication, teamwork) and never add requirements the text does \
not state. ideal_profile describes the candidate who fits best, written the way \
a resume would describe them.

Example input:
  Requirements: 5+ years of Go; experience with Postgres or MySQL; familiarity
  with Terraform; Bachelor's degree or equivalent experience.
  Nice to have: GraphQL, a Master's degree.
Example output (abridged):
  must_have: [{skill: Go, min_years: 5}]
  nice_to_have: [PostgreSQL, MySQL, Terraform, GraphQL]
  min_total_years: null, min_degree: null"""
