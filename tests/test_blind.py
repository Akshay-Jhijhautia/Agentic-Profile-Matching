"""Blind screening: who a candidate is plays no part in judging them."""

import pytest

from resume_shortlister.anonymous import Anonymous
from resume_shortlister.candidates import blind_resume
from resume_shortlister.rag import resume_rag
from resume_shortlister.rag.personal_data import is_personal_line, redact

RESUME = """Ruby Swift
ruby.swift@example.com | +1 (555) 010-1111 | Austin, TX
linkedin.com/in/ruby-swift

SUMMARY
Ruby Swift is a backend developer.
Date of birth: 12 March 1990
Nationality: Canadian

SKILLS
Python, Django

EXPERIENCE
Backend Developer, Contoso (May 2019 - Present)
- Built Python APIs with Django; on-call 01.2020 - 12.2023.
"""


def test_names_contact_details_and_personal_fields_are_replaced():
    text = redact("Anika Laurent (anika@example.com, 555-010-9012) · https://anika.dev\nAge: 34\nAnika's API", "Anika Laurent")
    assert text == "[candidate] ([email], [phone]) · [link]\nAge: [removed]\n[candidate]'s API"


def test_evidence_that_only_looks_personal_is_kept():
    # A name part that is a skill or a month stays outside the full name; dates are not phones.
    text = redact("RUBY SWIFT built iOS apps in Swift from May 2019 - 01.2023 (2015 - 2019).", "Ruby Swift")
    assert text == "[candidate] built iOS apps in Swift from May 2019 - 01.2023 (2015 - 2019)."
    assert redact("Socket.io, Node.js and React.js", "May Chen") == "Socket.io, Node.js and React.js"


@pytest.mark.parametrize("line, personal", [
    ("Ruby Swift", True), ("ruby@example.com | Austin, TX", True), ("Austin, TX", True),
    ("+44 20 7946 0958", True), ("github.com/ruby", True), ("Senior Backend Engineer", False), ("", False),
])
def test_header_lines_that_only_identify_the_candidate(line, personal):
    assert is_personal_line(line, "Ruby Swift") is personal


def test_a_name_is_never_read_as_a_skill():
    assert resume_rag.extract_metadata(RESUME)["skills"] == ["Python", "Django"]
    listed = RESUME.replace("Python, Django", "Python, Django, Swift")
    assert "Swift" in resume_rag.extract_metadata(listed)["skills"]  # listed: a skill after all


def test_what_is_embedded_and_searched_is_impersonal():
    for chunk in resume_rag.chunk_resume(RESUME, "Ruby Swift", "ruby.txt", max_chars=200):
        assert "Ruby Swift" not in chunk["document"] and "example.com" not in chunk["document"]
        assert "Date of birth: [removed]" in chunk["document"] or chunk["section"] != "Summary"
    # The excerpts the recruiter reads keep the resume as written.
    assert any("Ruby Swift" in chunk["text"] for chunk in resume_rag.chunk_resume(RESUME, "Ruby Swift", "ruby.txt"))


def test_the_resume_the_model_reads():
    text = blind_resume(RESUME, "Ruby Swift")
    assert text.startswith("SUMMARY\n[candidate] is a backend developer.")
    for gone in ("Ruby Swift", "example.com", "555", "Austin", "linkedin", "1990", "Canadian"):
        assert gone not in text
    assert "Backend Developer, Contoso (May 2019 - Present)" in text and "01.2020 - 12.2023" in text


def test_labels_map_back_to_ids_and_names():
    anonymous = Anonymous([("anika_laurent", "Anika Laurent"), ("wei_chen", "Wei Chen")])
    assert [anonymous.label(i) for i in ("anika_laurent", "wei_chen")] == ["Candidate 1", "Candidate 2"]
    assert [anonymous.id_of(r) for r in ("Candidate 2", "candidate_1", "Candidate 3", "Wei Chen")] == [
        "wei_chen", "anika_laurent", None, None]
    assert anonymous.named("Candidate 1 leads Candidate 2; candidate 12 is unknown.") == (
        "Anika Laurent leads Wei Chen; candidate 12 is unknown.")
