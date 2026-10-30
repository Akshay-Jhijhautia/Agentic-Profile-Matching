"""Gaps and short roles, read from the dates: noted for the recruiter, never scored."""

from datetime import date

from resume_shortlister import timeline

TODAY = date(2026, 10, 30)
RESUME = """Jo Doe

EXPERIENCE
Frontend Engineer, Fabrikam (Dec 2022 - Present)
- Built React apps.
Backend Developer, Contoso (Mar 2020 - Mar 2022)
- Built APIs.
Summer Intern, Initech (Jun 2019 - Sep 2019)
- Tested things.
Support Engineer, Litware (Feb 2018 - Jun 2018)
- Answered tickets.
QA Analyst, Willow (2015 - 2017)
- Wrote tests.
"""


def test_gaps_and_short_roles_from_the_dates():
    assert timeline.notes(RESUME, TODAY) == [
        "11 months between Support Engineer, Litware (until Jun 2018) and Summer Intern, Initech (from Jun 2019)",
        "8 months between Backend Developer, Contoso (until Mar 2022) and Frontend Engineer, Fabrikam (from Dec 2022)",
        "Support Engineer, Litware: 5 months (Feb 2018 to Jun 2018)",
    ]
    # Not noted: a pause under 6 months (Willow to Litware, Initech to Contoso), a role short by
    # nature (the internship), and the current role.


def test_a_year_alone_counts_as_the_whole_year():
    years_only = "EXPERIENCE\nAnalyst, Willow (2019 - 2022)\n- QA.\nEngineer, Lumen (2023 - Present)\n- Apps.\n"
    assert timeline.notes(years_only, TODAY) == []
    with_a_year_missing = years_only.replace("2023 - Present", "2024 - Present")
    assert timeline.notes(with_a_year_missing, TODAY) == [
        "1 year between Analyst, Willow (until 2022) and Engineer, Lumen (from 2024)"]

