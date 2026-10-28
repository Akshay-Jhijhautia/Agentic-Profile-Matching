import dataclasses
import sys

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from resume_shortlister import llm, providers
from resume_shortlister.agent.text import last_ai_text
from resume_shortlister.config import Settings
from resume_shortlister.rag import resume_rag
from tests.fakes import FakeChatModel

REQUEST = httpx.Request("POST", "https://example.test")


@pytest.fixture
def no_keys(clean_env):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        clean_env.delenv(name, raising=False)
    return clean_env


def _model(provider: str, model: str = "m", **settings) -> object:
    return providers.create_chat_model(providers.get(provider), model, Settings(llm_provider=provider, llm_model=model, **settings))


# --- settings ---------------------------------------------------------------------------------------


def test_each_provider_has_its_default_model(clean_env):
    assert Settings().llm_model == "gpt-5.6-terra"
    clean_env.setenv("LLM_PROVIDER", "Anthropic")
    settings = Settings.from_env()
    assert (settings.llm_provider, settings.llm_model) == ("anthropic", "claude-opus-5-5")
    assert settings.model_label == "Anthropic claude-opus-5-5"


def test_llm_names_win_and_openai_names_only_configure_openai(clean_env):
    clean_env.setenv("OPENAI_MODEL", "gpt-old")
    clean_env.setenv("OPENAI_TIMEOUT", "5")
    assert Settings.from_env().llm_model == "gpt-old"
    clean_env.setenv("LLM_MODEL", "gpt-new")
    assert Settings.from_env().llm_model == "gpt-new"
    clean_env.setenv("LLM_PROVIDER", "google")
    clean_env.delenv("LLM_MODEL")
    settings = Settings.from_env()
    assert (settings.llm_model, settings.llm_timeout) == ("gemini-3.1-pro", 60.0)


@pytest.mark.parametrize("settings, error", [
    ({"llm_provider": "mistral"}, "Unknown LLM_PROVIDER 'mistral'"),
    ({"llm_provider": "ollama"}, "Set LLM_MODEL"),
    ({"llm_provider": "anthropic", "llm_temperature": 0.0}, "accept no temperature"),
])
def test_provider_settings_are_validated(settings, error):
    with pytest.raises(ValueError, match=error):
        Settings(**settings)


def test_responses_api_is_on_for_openai_and_off_for_compatible_servers():
    assert Settings().responses_api
    assert not Settings(llm_base_url="http://localhost:8000/v1").responses_api
    assert Settings(llm_base_url="http://localhost:8000/v1", llm_responses_api=True).responses_api


# --- building models ----------------------------------------------------------------------------------


def test_model_specs_name_a_provider_or_keep_the_configured_one():
    assert providers.split_model_spec("anthropic:claude-sonnet-5-5", "openai") == (
        providers.PROVIDERS["anthropic"], "claude-sonnet-5-5")
    assert providers.split_model_spec("gpt-test", "openai") == (providers.PROVIDERS["openai"], "gpt-test")
    # An Ollama tag is not a provider prefix.
    assert providers.split_model_spec("qwen3:8b", "ollama") == (providers.PROVIDERS["ollama"], "qwen3:8b")


def test_make_chat_model_accepts_a_provider_prefix(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "get_settings", lambda: Settings(llm_provider="ollama", llm_model="qwen3"))
    pytest.importorskip("langchain_ollama")
    assert type(llm.make_chat_model()).__name__ == "ChatOllama"
    model = llm.make_chat_model("openai:gpt-test")
    assert isinstance(model, ChatOpenAI) and model.model_name == "gpt-test"


def test_anthropic_model_gets_no_temperature_and_native_structured_output(no_keys):
    pytest.importorskip("langchain_anthropic")
    no_keys.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    model = _model("anthropic", "claude-opus-5-5", llm_timeout=30, llm_max_retries=1)
    assert (model.model, model.temperature) == ("claude-opus-5-5", None)
    assert (model.default_request_timeout, model.max_retries) == (30, 1)
    assert providers.structured_options(model) == {"method": "json_schema"}
    with pytest.raises(ValueError, match="accept no temperature"):
        providers.create_chat_model(providers.get("anthropic"), "m", Settings(llm_temperature=0.2))


def test_google_and_ollama_models(no_keys):
    pytest.importorskip("langchain_google_genai")
    pytest.importorskip("langchain_ollama")
    no_keys.setenv("GEMINI_API_KEY", "g-test")
    google = _model("google", "gemini-3.1-pro", llm_temperature=0.0)
    assert (google.model, google.temperature, google.timeout) == ("gemini-3.1-pro", 0.0, 60.0)
    ollama = _model("ollama", "qwen3:8b", llm_base_url="http://gpu-box:11434")
    assert (ollama.base_url, ollama.client_kwargs) == ("http://gpu-box:11434", {"timeout": 60.0})
    assert providers.structured_options(ollama) == {}


def test_compatible_server_never_gets_the_openai_key(no_keys):
    no_keys.setenv("OPENAI_API_KEY", "sk-openai-secret")
    model = _model("openai", "llama-3.3-70b", llm_base_url="http://localhost:8000/v1")
    assert model.openai_api_key.get_secret_value() == "not-needed"
    assert str(model.openai_api_base) == "http://localhost:8000/v1" and not model.use_responses_api
    no_keys.setenv("LLM_API_KEY", "server-key")
    model = _model("openai", "llama-3.3-70b", llm_base_url="http://localhost:8000/v1")
    assert model.openai_api_key.get_secret_value() == "server-key"


def test_an_override_for_another_provider_gets_none_of_the_configured_endpoint(no_keys):
    pytest.importorskip("langchain_anthropic")
    no_keys.setenv("LLM_API_KEY", "compatible-server-key")
    no_keys.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    settings = Settings(llm_base_url="http://localhost:8000/v1", llm_model="llama")
    model = providers.create_chat_model(*providers.split_model_spec("anthropic:claude-opus-5-5", "openai"), settings)
    assert model.anthropic_api_key.get_secret_value() == "sk-ant-test"
    assert model.anthropic_api_url == "https://api.anthropic.com"
    other = providers.create_chat_model(*providers.split_model_spec("gpt-test", "openai"), settings)
    assert other.openai_api_key.get_secret_value() == "compatible-server-key"


def test_missing_integration_names_the_extra(monkeypatch):
    broken = dataclasses.replace(providers.PROVIDERS["anthropic"], model_class="no_such_module:Chat")
    with pytest.raises(ValueError, match=r"pip install 'resume-shortlister\[anthropic\]'"):
        providers.create_chat_model(broken, "m", Settings())


def test_unknown_models_keep_the_default_structured_output():
    assert providers.structured_options(FakeChatModel({})) == {}
    assert providers.structured_options(ChatOpenAI(model="m", api_key="sk-test")) == {}


# --- keys ---------------------------------------------------------------------------------------------


def test_required_keys_follow_the_provider(no_keys):
    assert providers.required_key_vars(Settings(llm_provider="google")) == ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    assert providers.required_key_vars(Settings(llm_provider="ollama", llm_model="qwen3")) == ()
    assert providers.required_key_vars(Settings(llm_base_url="http://localhost:8000/v1")) == ()
    assert providers.missing_key(("GOOGLE_API_KEY", "GEMINI_API_KEY")) == "GOOGLE_API_KEY"
    no_keys.setenv("GEMINI_API_KEY", "g-test")
    assert providers.missing_key(("GOOGLE_API_KEY", "GEMINI_API_KEY")) is None
    no_keys.setenv("LLM_API_KEY", "any")
    assert providers.missing_key(("ANTHROPIC_API_KEY",)) is None


def test_check_ready_names_the_providers_key(services, no_keys):
    services.settings = dataclasses.replace(services.settings, llm_provider="anthropic", llm_model="claude-opus-5-5")
    with pytest.raises(ValueError, match=r"ANTHROPIC_API_KEY is not set \(needed for Anthropic claude-opus-5-5\)"):
        services.check_ready()
    services.settings = dataclasses.replace(services.settings, llm_provider="ollama", llm_model="qwen3")
    services.check_ready()  # a local server needs no key


# --- errors -------------------------------------------------------------------------------------------


def test_error_messages_name_the_provider_and_the_fix():
    from langchain_openai.chat_models.base import OpenAIAuthenticationError

    auth = OpenAIAuthenticationError("bad key", response=httpx.Response(401, request=REQUEST), body=None)
    assert providers.error_message(auth) == "OpenAI authentication failed. Check OPENAI_API_KEY."
    # A raw SDK error, as the OpenAI embedder raises it.
    assert providers.error_message(openai.APITimeoutError(request=REQUEST)) == (
        "could not reach the OpenAI API. Check your connection or LLM_BASE_URL.")
    assert providers.error_message(ConnectionError("Failed to connect to Ollama")).startswith(
        "could not reach the model server (Failed to connect to Ollama)")
    # langchain-ollama lets the raw network error through.
    assert providers.error_message(httpx.ConnectError("[Errno 61] Connection refused")).startswith(
        "could not reach the model server ([Errno 61] Connection refused)")
    assert providers.error_message(ValueError("No resume folder")) == "No resume folder"


def test_error_messages_for_optional_sdks():
    anthropic = pytest.importorskip("langchain_anthropic.chat_models")
    ollama = pytest.importorskip("ollama")
    rate = anthropic.AnthropicRateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None)
    assert providers.error_message(rate).startswith("Anthropic rate limited the request")
    google = pytest.importorskip("google.genai.errors")
    invalid_key = google.ClientError(400, {"error": {"message": "API key not valid", "status": "INVALID_ARGUMENT",
                                                     "details": [{"reason": "API_KEY_INVALID"}]}})
    assert providers.error_message(invalid_key) == "Google Gemini authentication failed. Check GOOGLE_API_KEY."
    missing = ollama.ResponseError("model 'qwen9' not found", 404)
    assert providers.error_message(missing).startswith("Ollama could not find the model; check LLM_MODEL")


def test_cli_errors_cover_imported_sdks_only(monkeypatch):
    assert openai.OpenAIError in providers.cli_errors() and ValueError in providers.cli_errors()
    monkeypatch.delitem(sys.modules, "ollama", raising=False)
    assert not any(cls.__module__.startswith("ollama") for cls in providers.api_errors())


# --- the indexer's metadata fallback --------------------------------------------------------------------


class SchemaModel:
    """A chat model answering any JSON-schema request with ``answer`` (or raising it)."""

    def __init__(self, answer):
        self.answer = answer

    def with_structured_output(self, schema, **_kwargs):
        def respond(_messages):
            if isinstance(self.answer, Exception):
                raise self.answer
            return self.answer
        return RunnableLambda(respond)


def test_metadata_fallback_uses_the_configured_model(monkeypatch, no_keys):
    monkeypatch.setattr(llm, "get_settings", lambda: Settings(llm_provider="anthropic"))
    assert llm.metadata_fallback() is None  # no ANTHROPIC_API_KEY: rules only

    no_keys.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(llm, "get_chat_model", lambda: SchemaModel({"name": "Ann Lee"}))
    extract = llm.metadata_fallback()
    assert extract(resume_rag.metadata_schema(["name"]), "system", "resume") == {"name": "Ann Lee"}

    down = openai.APIConnectionError(request=REQUEST)
    monkeypatch.setattr(llm, "get_chat_model", lambda: SchemaModel(down))
    with pytest.raises(resume_rag.MetadataLLMError, match="could not reach the OpenAI API"):
        llm.metadata_fallback()(resume_rag.metadata_schema(["name"]), "system", "resume")


def test_replies_made_of_content_blocks_are_read_as_text():
    # Claude (with thinking) and OpenAI's Responses API reply with content blocks, not a string.
    blocks = AIMessage(content=[{"type": "thinking", "thinking": "hmm", "signature": "s"},
                                {"type": "text", "text": "Hello"}])
    assert last_ai_text({"messages": [blocks, AIMessage(content=[], tool_calls=[])]}) == "Hello"
