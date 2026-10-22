import pytest

from profile_matching import changes, reports
from profile_matching.schemas import CandidateMatch, Requirements, ScoreBreakdown
from tests.fakes import fake_assessment


def _role(**overrides) -> Requirements:
    fields = {"role_title": "Frontend Engineer", "must_have": [{"skill": "React", "min_years": 3}],
              "nice_to_have": ["GraphQL"], "min_total_years": None, "min_degree": None,
              "ideal_profile": "Frontend engineer."}
    return Requirements.model_validate(fields | overrides)


def _match(id_, rank, score, failed=(), semantic=0.5, keyword=0.5, coverage=1.0, checks=()):
    return CandidateMatch(
        id=id_, name=id_.title(), resume=f"resumes/resume_{id_}.txt", rank=rank, match_score=score,
        failed_requirements=list(failed), must_have_checks=list(checks), matched_skills=[],
        missing_must_haves=[], score_breakdown=ScoreBreakdown(semantic=semantic, keyword=keyword,
                                                             skill_coverage=coverage),
        experience_years=5.0, highest_degree=None, reasoning="r", excerpts=[],
    )


# --- requirement changes ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"must_have": [{"skill": "React", "min_years": 3}, {"skill": "Kubernetes", "min_years": None}]},
         ["Kubernetes is now a must-have"]),
        ({"must_have": [{"skill": "React", "min_years": 5}]}, ["React: 3+ yrs → 5+ yrs"]),
        ({"must_have": [], "nice_to_have": ["GraphQL", "React"]}, ["React is no longer a must-have (moved to nice-to-have)"]),
        ({"must_have": []}, ["React is no longer a must-have (dropped)"]),
        ({"nice_to_have": ["GraphQL", "Jest"]}, ["Nice to have added: Jest"]),
        ({"nice_to_have": []}, ["Nice to have removed: GraphQL"]),
        ({"min_total_years": 4}, ["Total experience: none → 4+ years"]),
        ({"min_degree": "Master's"}, ["Degree: none → Master's"]),
        ({}, []),
    ],
)
def test_requirement_changes(overrides, expected):
    assert changes.requirement_changes(_role(), _role(**overrides)) == expected


def test_promoting_a_nice_to_have_is_one_change():
    new = _role(must_have=[{"skill": "React", "min_years": 3}, {"skill": "GraphQL", "min_years": None}],
                nice_to_have=[])
    assert changes.requirement_changes(_role(), new) == ["GraphQL is now a must-have"]


# --- reasons ------------------------------------------------------------------------------------------


def test_reasons_name_the_component_that_moved_the_score_most():
    old = _match("ann", 3, 70, semantic=0.6, keyword=0.5, coverage=0.5)
    new = _match("ann", 1, 85, semantic=0.62, keyword=0.5, coverage=1.0)
    assert changes.reasons_for(old, new) == ["score 70 → 85 (must-have skill coverage 50% → 100%)"]


def test_reasons_for_knockouts():
    passing = _match("bob", 2, 80)
    failing = _match("bob", 40, 80, failed=["no Kubernetes on resume (needs Kubernetes)"])
    assert changes.reasons_for(passing, failing) == ["now fails no Kubernetes on resume (needs Kubernetes)"]
    assert changes.reasons_for(failing, passing) == ["no longer fails no Kubernetes on resume (needs Kubernetes)"]


def test_reasons_for_unverified_claims():
    unverified = [{"requirement": "5+ yrs React", "status": "unverified", "detail": "lists React but years unclear"}]
    old, new = _match("cy", 1, 80), _match("cy", 1, 80, checks=unverified)
    assert changes.reasons_for(old, new) == ["now unverified: lists React but years unclear"]
    assert changes.reasons_for(new, old) == ["no longer unverified"]
    failed = _match("cy", 9, 80, failed=["lists React but has 3.8 yrs total experience (needs 5+ yrs React)"])
    assert changes.reasons_for(new, failed) == [
        "now fails lists React but has 3.8 yrs total experience (needs 5+ yrs React)"
    ]


def test_reasons_without_history():
    assert changes.reasons_for(None, _match("dee", 1, 80)) == ["not screened before"]
    assert changes.reasons_for(_match("dee", 1, 80), None) == ["no longer in the index"]


# --- ranking diff ---------------------------------------------------------------------------------------


def test_ranking_diff_classifies_every_move():
    old = [_match("ann", 1, 90), _match("bob", 2, 85), _match("cy", 3, 80), _match("dee", 4, 60)]
    new = [
        _match("cy", 1, 88, coverage=1.0),
        _match("ann", 2, 87),
        _match("dee", 3, 75),
        _match("bob", 4, 85, failed=["no Kubernetes on resume (needs Kubernetes)"]),
    ]
    diff = changes.ranking_diff(_role(), _role(min_total_years=4), old, new, old[:3], new[:3])
    kinds = {c.id: c.kind for c in diff.changes}
    assert kinds == {"cy": "up", "ann": "down", "dee": "new", "bob": "dropped"}
    assert [c.id for c in diff.changes] == ["cy", "ann", "dee", "bob"]  # new shortlist, then leavers
    dee = diff.changes[2]
    assert (dee.old_rank, dee.new_rank, dee.old_score, dee.new_score) == (4, 3, 60, 75)
    bob = diff.changes[3]
    assert bob.reasons == ["now fails no Kubernetes on resume (needs Kubernetes)"]
    assert diff.requirement_changes == ["Total experience: none → 4+ years"]


def test_candidate_outranked_out_of_the_shortlist():
    old = [_match("ann", 1, 90), _match("bob", 2, 85)]
    new = [_match("cy", 1, 95), _match("ann", 2, 90), _match("bob", 3, 85)]
    diff = changes.ranking_diff(_role(), _role(), old, new, old, new[:2])
    (bob,) = diff.of_kind("dropped")
    assert bob.reasons == ["outranked by other candidates (now #3)"]
    (cy,) = diff.of_kind("new")
    assert cy.reasons == ["not screened before"] and cy.old_rank is None


# --- report section -----------------------------------------------------------------------------------


def test_changes_section_renders_moves_and_reasons():
    old = [_match("ann", 1, 90), _match("bob", 2, 85)]
    new = [_match("bob", 1, 92, coverage=1.0), _match("ann", 2, 88)]
    diff = changes.ranking_diff(_role(), _role(min_degree="Master's"), old, new, old, new)
    section = "\n".join(reports.changes_section(diff))
    assert "- Degree: none → Master's" in section
    assert "| Bob | ▲ 2 → 1 | 85 → 92 |" in section
    assert "| Ann | ▼ 1 → 2 | 90 → 88 |" in section


def test_changes_section_when_nothing_moved():
    old = [_match("ann", 1, 90)]
    section = reports.changes_section(changes.ranking_diff(_role(), _role(), old, old, old, old))
    assert "- The requirements did not change." in section
    assert "The shortlist order is unchanged." in section


def test_moves_caused_by_others_are_explained():
    old = [_match("ann", 1, 90), _match("bob", 2, 85), _match("cy", 3, 80)]
    new = [
        _match("cy", 1, 80),
        _match("ann", 3, 90, failed=["no Go on resume (needs Go)"]),
        _match("bob", 4, 85, failed=["no Go on resume (needs Go)"]),
    ]
    diff = changes.ranking_diff(_role(), _role(), old, new, old, new[:1])
    (cy,) = diff.of_kind("up")
    assert cy.reasons == ["2 candidate(s) ranked above now fail a must-have"]

    old = [_match("ann", 1, 90), _match("bob", 2, 85)]
    new = [_match("bob", 1, 85), _match("ann", 2, 90)]  # same scores, order flipped by others
    (ann,) = changes.ranking_diff(_role(), _role(), old, new, old, new).of_kind("down")
    assert ann.reasons == ["overtaken by Bob"]


def test_conversation_context_includes_the_latest_refinement():
    old = [_match("ann", 1, 90), _match("bob", 2, 85)]
    new = [_match("bob", 1, 85), _match("ann", 5, 60, failed=["no Go on resume (needs Go)"])]
    diff = changes.ranking_diff(_role(), _role(min_total_years=4), old, new, old, new[:1])
    context = reports.conversation_context(_role(min_total_years=4), new, new[:1], diff)
    assert "Latest refinement: Total experience: none → 4+ years" in context
    assert "- Ann: ▼ left (now #5), now fails no Go on resume (needs Go)" in context
    assert "Latest refinement" not in reports.conversation_context(_role(), new, new[:1])


def test_a_move_against_the_own_score_also_names_the_others():
    old = [_match("ann", 1, 90), _match("bob", 2, 85)]
    new = [_match("bob", 1, 80), _match("ann", 9, 90, failed=["no Go on resume (needs Go)"])]
    (bob,) = changes.ranking_diff(_role(), _role(), old, new, old, new[:1]).of_kind("up")
    assert bob.reasons == ["score 85 → 80", "1 candidate(s) ranked above now fail a must-have"]


def test_scores_with_and_without_deep_analysis_get_no_component_reason():
    assessed = _match("ann", 1, 80).model_copy(update={"assessment": fake_assessment(98)})
    round_one = _match("ann", 12, 60, failed=["no Go on resume (needs Go)"])
    assert changes.reasons_for(assessed, round_one) == [
        "now fails no Go on resume (needs Go)", f"score {assessed.final_score} → 60",
    ]
