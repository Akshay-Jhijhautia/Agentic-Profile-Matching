"""Reproducible model answers: the response cache and its use by the agent."""

import dataclasses
import itertools
import json
import sqlite3
import threading

import pytest
from langchain_openai import ChatOpenAI

import matching_agent
from resume_shortlister import app, llm, providers
from resume_shortlister.config import Settings
from resume_shortlister.llm_cache import ResponseCache
from resume_shortlister.schemas import DeepReview, Requirements, SkillRequirement
from resume_shortlister.services import Services
from tests.fakes import SCREENING_RESPONSES, FakeChatModel, fake_review
from tests.test_agent_pipeline import PYTHON_ROLE

ANSWER = SkillRequirement(skill="Python", min_years=5)


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(tmp_path / "answers.sqlite")


def _model_calls(model: FakeChatModel) -> int:
    return len(model.calls)


def test_the_same_question_gets_the_stored_answer(cache):
    fake = FakeChatModel({SkillRequirement: ANSWER})
    first = llm.structured(SkillRequirement, "system", "user", model=fake, cache=cache)
    fake.responses[SkillRequirement] = SkillRequirement(skill="Go", min_years=None)  # the model changed its mind
    again = llm.structured(SkillRequirement, "system", "user", model=fake, cache=cache)
    assert first == again == ANSWER and _model_calls(fake) == 1


@pytest.mark.parametrize("change", ["system", "user", "schema", "model"])
def test_a_different_question_asks_the_model(cache, change):
    fake = FakeChatModel({SkillRequirement: ANSWER, Requirements: PYTHON_ROLE})
    llm.structured(SkillRequirement, "system", "user", model=fake, cache=cache)
    other = FakeChatModel({SkillRequirement: ANSWER})
    other.model = "another-model"  # a different model identity
    asked = {
        "system": lambda: llm.structured(SkillRequirement, "system v2", "user", model=fake, cache=cache),
        "user": lambda: llm.structured(SkillRequirement, "system", "user 2", model=fake, cache=cache),
        "schema": lambda: llm.structured(Requirements, "system", "user", model=fake, cache=cache),
        "model": lambda: llm.structured(SkillRequirement, "system", "user", model=other, cache=cache),
    }
    asked[change]()
    assert _model_calls(fake) + _model_calls(other) == 2


def test_json_schema_answers_are_cached_too(cache):
    schema = {"title": "resume_metadata", "type": "object", "properties": {"name": {"type": "string"}}}

    class DictModel(FakeChatModel):
        def with_structured_output(self, schema, **kwargs):
            return super().with_structured_output("dict", **kwargs)

    fake = DictModel({"dict": lambda messages: {"name": "Ann Lee"}})
    assert llm.structured(schema, "s", "u", model=fake, cache=cache) == {"name": "Ann Lee"}
    assert llm.structured(schema, "s", "u", model=fake, cache=cache) == {"name": "Ann Lee"}
    assert _model_calls(fake) == 1 and cache.stats().by_kind == {"resume_metadata": 1}


def test_the_first_stored_answer_wins(cache):
    """Two threads asking the same question at once both use one answer."""
    assert cache.put("k", model="m", kind="K", request={}, answer='"first"') == '"first"'
    assert cache.put("k", model="m", kind="K", request={}, answer='"second"') == '"first"'

    answers = itertools.count()
    fake = FakeChatModel({SkillRequirement: lambda messages: SkillRequirement(skill=f"S{next(answers)}", min_years=None)})
    results = []
    threads = [threading.Thread(target=lambda: results.append(
        llm.structured(SkillRequirement, "s", "u", model=fake, cache=cache))) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len({r.skill for r in results}) == 1


def test_an_unreadable_answer_is_asked_again(cache):
    fake = FakeChatModel({SkillRequirement: ANSWER})
    llm.structured(SkillRequirement, "s", "u", model=fake, cache=cache)
    with sqlite3.connect(cache.path) as db:
        db.execute("UPDATE answers SET answer = 'not json'")
    assert llm.structured(SkillRequirement, "s", "u", model=fake, cache=cache) == ANSWER
    assert _model_calls(fake) == 2


def test_every_answer_has_an_audit_trail(cache):
    fake = FakeChatModel({SkillRequirement: ANSWER})
    for _ in range(3):
        llm.structured(SkillRequirement, "Extract one skill.", "5+ years of Python", model=fake, cache=cache)
    with sqlite3.connect(cache.path) as db:
        model, kind, request, answer, uses = db.execute("SELECT model, kind, request, answer, uses FROM answers").fetchone()
    assert (model, kind, uses) == ("FakeChatModel:?", "SkillRequirement", 3)
    assert json.loads(request) == {"system": "Extract one skill.", "user": "5+ years of Python"}
    assert SkillRequirement.model_validate_json(answer) == ANSWER
    stats = cache.stats()
    assert (stats.answers, stats.reused, stats.by_kind) == (1, 2, {"SkillRequirement": 1})

    assert cache.unused_since(30) == 0
    with sqlite3.connect(cache.path) as db:
        db.execute("UPDATE answers SET last_used = '2020-01-01T00:00:00+00:00'")
    assert cache.unused_since(30) == 1 and cache.remove_unused(30) == 1 and cache.stats().answers == 0


def test_model_identity_names_what_changes_answers():
    assert providers.model_identity(ChatOpenAI(model="gpt-test", api_key="sk-test")) == "openai:gpt-test"
    assert providers.model_identity(ChatOpenAI(model="gpt-test", api_key="sk-test", temperature=0.2)) == \
        "openai:gpt-test temperature=0.2"
    local = ChatOpenAI(model="llama", api_key="x", base_url="http://localhost:8000/v1")
    assert providers.model_identity(local) == "openai:llama@http://localhost:8000/v1"


# --- reproducible screenings --------------------------------------------------------------------------


def _changing_reviews():
    """A model whose deep reviews change from run to run: all strong in the first, all
    partial in the second (it is shared by both runs)."""
    verdicts = itertools.chain(itertools.repeat("strong", 3), itertools.repeat("partial"))
    return lambda messages: fake_review(next(verdicts))(messages)


def _screen(services, reviews):
    services.settings = dataclasses.replace(services.settings, screening_rounds=3, shortlist_size=3)
    services.model = FakeChatModel({Requirements: PYTHON_ROLE, **SCREENING_RESPONSES, DeepReview: reviews})
    state = matching_agent.screen(services, "Python backend engineer; Django is a plus")
    return [(m.id, m.final_score, m.recommendation.decision) for m in state["shortlist"]], services.model


def test_with_the_cache_a_screening_repeats_exactly(services, cache):
    services.llm_cache = cache
    reviews = _changing_reviews()
    first, _ = _screen(services, reviews)
    again, model = _screen(services, reviews)
    assert again == first and len(model.calls) == 0  # every answer came from the cache


def test_without_the_cache_the_model_is_asked_every_time(services):
    reviews = _changing_reviews()
    first, _ = _screen(services, reviews)
    again, model = _screen(services, reviews)
    assert again != first and len(model.calls) > 0


# --- settings and commands ------------------------------------------------------------------------------


def test_the_cache_is_on_unless_turned_off(clean_env, tmp_path):
    clean_env.setenv("RESUME_SHORTLISTER_HOME", str(tmp_path))
    services = Services.from_settings(Settings.from_env())
    assert services.llm_cache.path == tmp_path / "llm_cache.sqlite"
    clean_env.setenv("LLM_CACHE", "false")
    assert Services.from_settings(Settings.from_env()).llm_cache is None


def test_cache_command_shows_and_clears(cache, monkeypatch, capsys):
    llm.structured(SkillRequirement, "s", "u", model=FakeChatModel({SkillRequirement: ANSWER}), cache=cache)
    monkeypatch.setattr(app, "get_settings", lambda: Settings(llm_cache_db=str(cache.path)))
    assert app.main(["cache"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("1 stored model answers, reused 0 times") and "SkillRequirement" in out
    assert app.main(["cache", "--clear"]) == 0
    assert "Deleted 1 stored answers." in capsys.readouterr().out and cache.stats().answers == 0


def test_clean_removes_answers_no_screening_used_for_a_while(cache, tmp_path):
    from resume_shortlister import maintenance
    llm.structured(SkillRequirement, "s", "u", model=FakeChatModel({SkillRequirement: ANSWER}), cache=cache)
    settings = Settings(llm_cache_db=str(cache.path), uploads_dir=str(tmp_path / "u"), db_dir=str(tmp_path / "i"),
                        sessions_db=str(tmp_path / "s.sqlite"))
    assert maintenance.find_stale(settings, older_than_days=30) == []
    with sqlite3.connect(cache.path) as db:
        db.execute("UPDATE answers SET last_used = '2020-01-01T00:00:00+00:00'")
    (stale,) = maintenance.find_stale(settings, older_than_days=30)
    assert (stale.kind, stale.name) == ("cache", "1 stored model answers")
    maintenance.remove(settings, [stale])
    assert cache.stats().answers == 0
