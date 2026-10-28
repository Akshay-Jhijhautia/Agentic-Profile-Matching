"""The chat-model providers the agent can run on, and what each one needs.

Every provider is a LangChain chat-model integration, imported only when it is
chosen: OpenAI's comes with the base install, the others with a pip extra
(``pip install 'resume-shortlister[anthropic]'``). Each entry records how to build
the model and the quirks the rest of the code never sees: Claude rejects
sampling parameters and gives structured output through its native JSON-schema
mode, OpenAI's reasoning models need the Responses API, a local Ollama server
needs no key.

Any OpenAI-compatible server (vLLM, LM Studio, OpenRouter, xAI, ...) runs
through the ``openai`` provider with ``LLM_BASE_URL``; its key then comes only
from ``LLM_API_KEY``, so ``OPENAI_API_KEY`` is never sent to another server.

Errors are provider-agnostic too: the integrations raise ``langchain_core``
``ModelError`` subclasses (authentication, rate limit, connection, ...), and
``error_message`` turns those, and raw SDK errors such as the embedder's, into
one line that names the provider.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import httpx
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
)
from langchain_core.language_models import BaseChatModel

if TYPE_CHECKING:
    from .config import Settings

# Chat model key for any provider; required for an OpenAI-compatible server that needs one.
API_KEY_VAR = "LLM_API_KEY"


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    # "module:Class" of the LangChain chat model.
    model_class: str
    # pip extra that installs the integration; None: part of the base install.
    extra: str | None
    # None: there is no sensible default and LLM_MODEL must be set.
    default_model: str | None
    # Environment variables that can hold the key, in order; () means no key is needed.
    key_vars: tuple[str, ...]
    # Root modules whose exceptions come from this provider (integration and SDK).
    modules: tuple[str, ...]
    # "module:Class" base exception of the provider's SDK.
    sdk_error: str
    # Constructor arguments for timeouts, retries and the endpoint, named as the integration names them.
    options: Callable[[Settings], dict[str, Any]]
    takes_temperature: bool = True
    # ``with_structured_output`` method; None keeps the integration's default.
    structured_method: str | None = None

    @property
    def install_hint(self) -> str:
        return f"pip install 'resume-shortlister[{self.extra}]'" if self.extra else "pip install resume-shortlister"

    def model_type(self) -> type[BaseChatModel]:
        module, _, name = self.model_class.partition(":")
        try:
            return getattr(importlib.import_module(module), name)
        except ImportError as exc:
            raise ValueError(f"LLM_PROVIDER={self.name} needs its integration: {self.install_hint}") from exc


def _limits(settings: Settings) -> dict[str, Any]:
    return {"timeout": settings.llm_timeout, "max_retries": settings.llm_max_retries}


def _endpoint(settings: Settings) -> dict[str, Any]:
    return {"base_url": settings.llm_base_url} if settings.llm_base_url else {}


def _openai_options(settings: Settings) -> dict[str, Any]:
    options = _limits(settings) | _endpoint(settings) | {"use_responses_api": settings.responses_api}
    if settings.llm_base_url:
        # Another server: never send it OPENAI_API_KEY; local servers accept any key.
        options["api_key"] = os.getenv(API_KEY_VAR) or "not-needed"
    return options


PROVIDERS: dict[str, Provider] = {
    provider.name: provider
    for provider in (
        Provider(
            name="openai", label="OpenAI", model_class="langchain_openai:ChatOpenAI", extra=None,
            default_model="gpt-5.6-terra", key_vars=("OPENAI_API_KEY",),
            modules=("langchain_openai", "openai"), sdk_error="openai:OpenAIError", options=_openai_options,
        ),
        Provider(
            name="anthropic", label="Anthropic", model_class="langchain_anthropic:ChatAnthropic",
            extra="anthropic", default_model="claude-opus-5-5", key_vars=("ANTHROPIC_API_KEY",),
            modules=("langchain_anthropic", "anthropic"), sdk_error="anthropic:AnthropicError",
            options=lambda s: _limits(s) | _endpoint(s),
            # Current Claude models reject temperature, and forced tool use (the default
            # structured-output method) under always-on thinking.
            takes_temperature=False, structured_method="json_schema",
        ),
        Provider(
            name="google", label="Google Gemini", model_class="langchain_google_genai:ChatGoogleGenerativeAI",
            extra="google", default_model="gemini-3.1-pro", key_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
            modules=("langchain_google_genai", "google"), sdk_error="google.genai.errors:APIError",
            options=_limits,
        ),
        Provider(
            name="ollama", label="Ollama", model_class="langchain_ollama:ChatOllama", extra="ollama",
            default_model=None, key_vars=(), modules=("langchain_ollama", "ollama"),
            sdk_error="ollama:ResponseError",
            options=lambda s: _endpoint(s) | {"client_kwargs": {"timeout": s.llm_timeout}},
        ),
    )
}


def get(name: str) -> Provider:
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Use one of: {', '.join(PROVIDERS)}.") from None


def split_model_spec(spec: str, default_provider: str) -> tuple[Provider, str]:
    """``"anthropic:claude-sonnet-5-5"`` -> (anthropic, model); a bare name keeps ``default_provider``.

    Only a known provider name counts as a prefix, so Ollama tags such as
    ``qwen3:8b`` stay whole.
    """
    prefix, _, rest = spec.partition(":")
    if rest and prefix in PROVIDERS:
        return PROVIDERS[prefix], rest
    return get(default_provider), spec


def create_chat_model(provider: Provider, model: str, settings: Settings) -> BaseChatModel:
    """``provider``'s chat model ``model`` with the configured limits.

    The endpoint settings (LLM_BASE_URL, LLM_API_KEY, OPENAI_RESPONSES_API)
    belong to the configured provider; another provider (a ``provider:model``
    override) gets its own defaults, so no key or URL leaks across providers.
    """
    configured = provider.name == settings.llm_provider
    if not configured:
        settings = replace(settings, llm_provider=provider.name, llm_model=model, llm_base_url=None,
                           llm_responses_api=None)
    kwargs = provider.options(settings)
    if settings.llm_temperature is not None:
        if not provider.takes_temperature:
            raise ValueError(f"{provider.label} models accept no temperature; unset LLM_TEMPERATURE.")
        kwargs["temperature"] = settings.llm_temperature
    if configured and os.getenv(API_KEY_VAR) and provider.key_vars:
        kwargs["api_key"] = os.environ[API_KEY_VAR]
    return provider.model_type()(model=model, **kwargs)


def model_identity(model: Any) -> str:
    """Which model answers, as far as it changes the answer: provider, model name, endpoint
    and temperature (``"openai:gpt-5.6-terra"``). Part of every cache key."""
    root = type(model).__module__.partition(".")[0]
    provider = next((p.name for p in PROVIDERS.values() if root in p.modules), type(model).__name__)
    name = getattr(model, "model_name", None) or getattr(model, "model", None) or "?"
    identity = f"{provider}:{name}"
    endpoint = getattr(model, "openai_api_base", None) or getattr(model, "base_url", None)
    if endpoint:
        identity += f"@{endpoint}"
    temperature = getattr(model, "temperature", None)
    return identity if temperature is None else f"{identity} temperature={temperature}"


def structured_options(model: Any) -> dict[str, Any]:
    """Extra ``with_structured_output`` arguments for ``model``'s provider (none for unknown models)."""
    root = type(model).__module__.partition(".")[0]
    provider = next((p for p in PROVIDERS.values() if root in p.modules), None)
    return {"method": provider.structured_method} if provider and provider.structured_method else {}


# --- keys ------------------------------------------------------------------------------------------


def required_key_vars(settings: Settings) -> tuple[str, ...]:
    """Variables that can hold the configured chat model's key; () when none is needed."""
    if settings.llm_provider == "openai" and settings.llm_base_url:
        return ()  # an OpenAI-compatible server: LLM_API_KEY only if it asks for one
    return get(settings.llm_provider).key_vars


def missing_key(key_vars: tuple[str, ...]) -> str | None:
    """The variable to set when none of ``key_vars`` (nor LLM_API_KEY) is set, else None."""
    if not key_vars or os.getenv(API_KEY_VAR) or any(os.getenv(var) for var in key_vars):
        return None
    return key_vars[0]


def missing_key_message(var: str, purpose: str) -> str:
    return (
        f"{var} is not set ({purpose}). Export it, or put {var}=... in a .env file "
        "in this folder or in ~/.resume-shortlister/.env."
    )


# --- errors ----------------------------------------------------------------------------------------


def api_errors() -> tuple[type[BaseException], ...]:
    """Exceptions that mean a model provider failed, rather than a bug.

    SDK base errors are looked up only for SDKs already imported: one that was
    never imported cannot have raised anything.
    """
    # httpx.TransportError: a network failure an integration did not wrap (langchain-ollama).
    found: list[type[BaseException]] = [ModelError, ConnectionError, httpx.TransportError]
    for provider in PROVIDERS.values():
        module, _, name = provider.sdk_error.partition(":")
        if module in sys.modules:
            found.append(getattr(sys.modules[module], name))
    return tuple(found)


def cli_errors() -> tuple[type[BaseException], ...]:
    """Expected failures the command line reports in one line instead of a traceback."""
    return (FileNotFoundError, NotADirectoryError, ValueError, *api_errors())


def _provider_of(exc: BaseException) -> Provider | None:
    for cls in type(exc).__mro__:
        root = cls.__module__.partition(".")[0]
        provider = next((p for p in PROVIDERS.values() if root in p.modules), None)
        if provider:
            return provider
    return None


def error_message(exc: BaseException) -> str:
    """A one-line explanation of an expected failure, naming the provider and the fix."""
    if not isinstance(exc, api_errors()):
        return str(exc)
    provider = _provider_of(exc)
    label = provider.label if provider else "The model provider"
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    # Raw SDK errors (the embedder's, Ollama's) only say it in the class name.
    unreachable = any(word in cls.__name__ for cls in type(exc).__mro__ for word in ("Connect", "Network", "Timeout"))
    # Google answers an invalid key with 400 API_KEY_INVALID rather than 401.
    if (isinstance(exc, (ModelAuthenticationError, ModelPermissionDeniedError)) or status in (401, 403)
            or "API_KEY_INVALID" in str(exc)):
        key = f" Check {provider.key_vars[0]}." if provider and provider.key_vars else ""
        return f"{label} authentication failed.{key}"
    if isinstance(exc, ModelRateLimitError) or status == 429:
        return f"{label} rate limited the request or the quota is used up: {exc}"
    if isinstance(exc, ModelNotFoundError) or status == 404:
        return f"{label} could not find the model; check LLM_MODEL: {exc}"
    if unreachable:
        # Without a known provider (Ollama raises a plain ConnectionError) its own text says more.
        target = f"the {provider.label} API" if provider else f"the model server ({exc})"
        return f"could not reach {target}. Check your connection or LLM_BASE_URL."
    return f"{provider.label} API error: {exc}" if provider else f"model provider error: {exc}"
