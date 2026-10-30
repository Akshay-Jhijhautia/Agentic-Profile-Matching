import pytest

from resume_shortlister import screening
from scripts import stability_check
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


def test_a_provider_failure_is_one_line_not_a_traceback(monkeypatch, capsys, services):
    import httpx
    import openai

    def out_of_credits(*_args, **_kwargs):
        raise openai.RateLimitError("You have no credits remaining.", body=None,
                                    response=httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com")))

    monkeypatch.setattr(stability_check.Services, "from_settings", lambda *a, **k: services)
    monkeypatch.setattr(services, "use_folder", lambda folder: type("Report", (), {"summary": lambda self: "ready"})())
    monkeypatch.setattr(services, "check_ready", lambda: None)
    monkeypatch.setattr(stability_check.matching_agent, "screen", out_of_credits)
    assert stability_check.main(["React developer", "--resumes", "x", "--runs", "2"]) == 1
    err = capsys.readouterr().err
    assert err.strip().splitlines()[-1].startswith("Error: OpenAI") and "Traceback" not in err
