import pytest
from langchain_openai import ChatOpenAI

from profile_matching import llm
from profile_matching.config import Settings
from profile_matching.schemas import SkillRequirement
from tests.fakes import FakeChatModel

ENV_VARS = ("OPENAI_MODEL", "OPENAI_TEMPERATURE", "OPENAI_TIMEOUT", "OPENAI_MAX_RETRIES", "CHROMA_DB_DIR",
            "REPORTS_DIR", "SHORTLIST_SIZE", "SCREENING_ROUNDS")


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_settings_defaults(clean_env):
    assert Settings.from_env() == Settings()


def test_settings_from_env(clean_env):
    clean_env.setenv("OPENAI_MODEL", "gpt-test")
    clean_env.setenv("OPENAI_TEMPERATURE", "0.3")
    clean_env.setenv("OPENAI_TIMEOUT", "5")
    clean_env.setenv("OPENAI_MAX_RETRIES", "0")
    clean_env.setenv("CHROMA_DB_DIR", "/tmp/db")
    clean_env.setenv("REPORTS_DIR", "/tmp/reports")
    clean_env.setenv("SHORTLIST_SIZE", "5")
    clean_env.setenv("SCREENING_ROUNDS", "2")
    assert Settings.from_env() == Settings("gpt-test", 0.3, 5.0, 0, "/tmp/db", "/tmp/reports", 5, 2)


@pytest.mark.parametrize("field, value", [("screening_rounds", 0), ("screening_rounds", 4), ("shortlist_size", 0)])
def test_settings_are_validated(field, value):
    with pytest.raises(ValueError, match=field):
        Settings(**{field: value})


def test_empty_temperature_means_provider_default(clean_env):
    clean_env.setenv("OPENAI_TEMPERATURE", "")
    assert Settings.from_env().llm_temperature is None


def test_chat_model_uses_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "get_settings", lambda: Settings(llm_model="gpt-test", llm_max_retries=1))
    llm.get_chat_model.cache_clear()
    try:
        model = llm.get_chat_model()
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "gpt-test" and model.max_retries == 1
    finally:
        llm.get_chat_model.cache_clear()


def test_structured_returns_the_schema_instance():
    answer = SkillRequirement(skill="Python", min_years=5)
    fake = FakeChatModel({SkillRequirement: answer})
    assert llm.structured(SkillRequirement, "system", "user", model=fake) is answer
    ((_, messages),) = fake.calls
    assert [m.content for m in messages] == ["system", "user"]
