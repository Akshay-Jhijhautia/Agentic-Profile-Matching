import pytest
from langchain_openai import ChatOpenAI

from profile_matching import llm
from profile_matching.config import Settings
from profile_matching.schemas import SkillRequirement
from tests.fakes import FakeChatModel

ENV_VARS = ("OPENAI_MODEL", "OPENAI_TEMPERATURE", "OPENAI_TIMEOUT", "OPENAI_MAX_RETRIES", "CHROMA_DB_DIR",
            "REPORTS_DIR", "SHORTLIST_SIZE", "SCREENING_ROUNDS", "SAVE_REPORTS", "OPENAI_RESPONSES_API",
            "UPLOADS_DIR", "EMBEDDER", "PROFILE_MATCH_HOME", "SESSIONS_DB")


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
    clean_env.setenv("SAVE_REPORTS", "off")
    clean_env.setenv("OPENAI_RESPONSES_API", "false")
    clean_env.setenv("UPLOADS_DIR", "/tmp/uploads")
    clean_env.setenv("EMBEDDER", "hashing")
    assert Settings.from_env() == Settings(
        llm_model="gpt-test", llm_temperature=0.3, llm_timeout=5.0, llm_max_retries=0,
        llm_responses_api=False, db_dir="/tmp/db", reports_dir="/tmp/reports", shortlist_size=5,
        screening_rounds=2, save_reports=False, uploads_dir="/tmp/uploads", embedder="hashing",
    )


@pytest.mark.parametrize(
    "field, value",
    [("screening_rounds", 0), ("screening_rounds", 4), ("shortlist_size", 0), ("embedder", "word2vec")],
)
def test_settings_are_validated(field, value):
    with pytest.raises(ValueError, match=field):
        Settings(**{field: value})


def test_data_home_holds_index_uploads_and_sessions(clean_env, tmp_path):
    clean_env.setenv("PROFILE_MATCH_HOME", str(tmp_path / "pm"))
    settings = Settings.from_env()
    assert (settings.db_dir, settings.uploads_dir, settings.sessions_db) == (
        str(tmp_path / "pm" / "index"), str(tmp_path / "pm" / "uploads"), str(tmp_path / "pm" / "sessions.sqlite"),
    )
    clean_env.setenv("CHROMA_DB_DIR", "/elsewhere/index")  # individual settings still win
    assert Settings.from_env().db_dir == "/elsewhere/index"


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
        assert model.model_name == "gpt-test" and model.max_retries == 1 and model.use_responses_api
    finally:
        llm.get_chat_model.cache_clear()


def test_structured_returns_the_schema_instance():
    answer = SkillRequirement(skill="Python", min_years=5)
    fake = FakeChatModel({SkillRequirement: answer})
    assert llm.structured(SkillRequirement, "system", "user", model=fake) is answer
    ((_, messages),) = fake.calls
    assert [m.content for m in messages] == ["system", "user"]
