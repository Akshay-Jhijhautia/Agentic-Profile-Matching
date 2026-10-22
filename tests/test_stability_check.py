import pytest

from profile_matching import screening
from scripts.stability_check import kendall_tau, summarise
from tests.test_explainability import ROLE, full_record


@pytest.mark.parametrize(
    "first, second, expected",
    [
        (["a", "b", "c"], ["a", "b", "c"], 1.0),
        (["a", "b", "c"], ["c", "b", "a"], -1.0),
        (["a", "b", "c"], ["b", "a", "c"], pytest.approx(1 / 3)),
        (["a", "b", "x"], ["b", "a", "y"], -1.0),  # only common candidates count
        (["a"], ["a"], 1.0),
    ],
)
def test_kendall_tau(first, second, expected):
    assert kendall_tau(first, second) == expected


def test_summarise_identical_runs_is_perfectly_stable(services):
    shortlist = [full_record(m) for m in screening.evaluate(services, ROLE)[:3]]
    run = {"shortlist": shortlist, "requirements": ROLE}
    metrics = summarise([run, run, run])
    assert metrics["top_k_overlap"] == 1.0 and metrics["kendall_tau"] == 1.0
    assert metrics["decision_agreement"] == 1.0 and metrics["deep_spread_max"] == 0
    assert metrics["identical_requirements"] is True
