# llm_providers.py
"""
Modular system for switching between LLM providers (Google, Groq, OpenAI, OpenRouter).

Usage via environment variable (recommended):
    export LLM_PROVIDER=openai # or google / groq / openrouter
    export LLM_MODEL=gpt-4.1 # optional -- overwrites the default model
    export LLM_TEMPERATURE=0.3 # optional -- default: 0.2

Usage via code:
    from llm_providers import get_llm, LLMProvider
    llm = get_llm(provider=LLMProvider.OPENAI, temperature=0.4)
    llm = get_llm(provider=LLMProvider.OPENROUTER, model_name="google/gemini-2.5-flash")

Required API keys in the .env file:
    GOOGLE_API_KEY      -> Google Gemini
    GROQ_API_KEY        -> Groq
    OPENAI_API_KEY      -> OpenAI
    OPENROUTER_API_KEY  -> OpenRouter
"""

import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, TypeVar

from dotenv import load_dotenv
from langchain.agents import create_agent

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from ...config import validate_provider

load_dotenv()


# ============================================================================
# ENUM OF PROVIDERS
# ============================================================================


class LLMProvider(Enum):
    """Identifiers of supported providers."""

    GOOGLE = "google"
    GROQ = "groq"
    OPENAI = "openai"
    OPENROUTER = "openrouter"


# ============================================================================
# BASE CLASS
# ============================================================================


class BaseLLMProvider(ABC):
    """Common interface for all LLM providers."""

    def __init__(self, temperature: float = 0.2, model_name: str | None = None):
        self.temperature = temperature
        self.model_name = model_name or self.get_default_model()
        self._llm = None

    @abstractmethod
    def get_default_model(self) -> str:
        """Default model for the provider."""

    @abstractmethod
    def get_api_key(self) -> str:
        """API key for the provider."""

    @abstractmethod
    def create_llm(self) -> Any:
        """Instantiates and returns the LLM."""

    def get_llm(self) -> Any:
        """Lazy loading — creates the LLM only on the first call."""
        if self._llm is None:
            self._llm = self.create_llm()
        return self._llm

    def create_agent_with_tools(self, tools: list, system_prompt: str, name: str | None) -> Any:
        """Creates a ReAct agent with linked tools.

        Args:
            tools: List of LangChain tools to bind to the LLM.
            system_prompt: The system prompt to use for the agent.
            name: An optional name for the `CompiledStateGraph`

        Returns:
            An agent instance with the LLM and tools ready for use.
        """
        llm = self.get_llm()
        llm_with_tools = llm.bind_tools(tools)
        return create_agent(
            model=llm_with_tools, tools=tools, system_prompt=system_prompt, name=name
        )


# ============================================================================
# CONCRETE PROVIDERS
# ============================================================================


class GoogleProvider(BaseLLMProvider):
    """Google Gemini via langchain-google-genai."""

    def get_default_model(self) -> str:
        """Popular Gemini models:
        gemini-2.5-flash (versatile, good for writing and general tasks)
        gemini-2.5-pro (more capable, ideal for dense scientific writing)"""
        print("default model: gemini-2.5-flash")
        return "gemini-2.5-flash"

    def get_api_key(self) -> str:
        """Returns the Google API key from environment variable."""
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY not found in .env")
        return key

    def create_llm(self) -> ChatGoogleGenerativeAI:
        """Instantiates the ChatGoogleGenerativeAI LLM with the specified model and API key."""
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=self.get_api_key(),
        )


class GroqProvider(BaseLLMProvider):
    """Groq via langchain-groq."""

    def get_default_model(self) -> str:
        """Groq's llama-3.3-70b-versatile is a strong open-source option for writing tasks.
        Popular models:
            llama-3.3-70b-versatile (open-source, strong for writing)
            mixtral-8x7b-32768 (fast, good for structured tasks)
        """
        return os.getenv("LLM_MODEL") or "llama-3.3-70b-versatile"

    def get_api_key(self) -> str:
        """Returns the Groq API key from environment variable."""
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not found in .env")
        return key

    def create_llm(self) -> ChatGroq:
        """Instantiates the ChatGroq LLM with the specified model and API key."""
        return ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            groq_api_key=self.get_api_key(),
        )


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI via langchain-openai.

    Suggested models:
        gtp-5.1          → More advanced, ideal for dense and creative writing.
        gpt-4.1          → more capable, ideal for dense scientific writing
        gpt-4.1-mini     → faster and more economical, good for structured tasks
        gpt-4o           → good cost/quality ratio
        gpt-4o-mini      → economical for simple tasks
        o3               → advanced reasoning (slower)
        o4-mini          → fast reasoning (good for tool use)
        gpt-3.5-turbo     → baseline strong performance, more affordable

    Define the model via the LLM_MODEL environment variable or model_name=...
    """

    def get_default_model(self) -> str:
        """Returns the default OpenAI model, which can be overridden by the LLM_MODEL environment variable.
        Popular models:
            gpt-4.1 (more capable, ideal for dense scientific writing)
            gpt-4.1-mini (faster and more economical, good for structured tasks
        """
        model = os.getenv("LLM_MODEL") or "gpt-4.1"
        print(f"default model: {model}")
        return model

    def get_api_key(self) -> str:
        """Returns the OpenAI API key from environment variable."""
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        return key

    def create_llm(self) -> ChatOpenAI:
        """Instantiates the ChatOpenAI LLM with the specified model and API key."""
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.get_api_key(),
        )


class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter via langchain-openai (OpenAI-compatible API).

    OpenRouter aggregates multiple models. Popular models:
        google/gemini-2.5-flash → fast, versatile
        google/gemini-2.5-pro → more capable
        anthropic/claude-3.5-sonnet → excellent for writing
        anthropic/claude-3-opus → more powerful
        openai/gpt-4-turbo → OpenAI via OpenRouter
        meta-llama/llama-3.3-70b-instruct → powerful open-source

    Define the model via the LLM_MODEL environment variable or model_name=...
    Get your key at: https://openrouter.ai/
    """

    def get_default_model(self) -> str:
        """Returns the default OpenRouter model, which can be overridden by the LLM_MODEL environment variable."""
        model = os.getenv("LLM_MODEL") or "google/gemini-2.5-flash"
        print(f"default model: {model}")
        return model

    def get_api_key(self) -> str:
        """Returns the OpenRouter API key from environment variable."""
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not found in .env")
        return key

    def create_llm(self) -> ChatOpenAI:
        """Instantiates the ChatOpenAI LLM configured to use OpenRouter's API with the specified model and API key."""
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.get_api_key(),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/duartejr/paper_reviwer",
                "X-Title": "Paper Reviewer",
            },
        )


# ============================================================================
# FACTORY
# ============================================================================


class LLMFactory:
    """Creates LLM providers via enum or environment variable."""

    _providers = {
        LLMProvider.GOOGLE: GoogleProvider,
        LLMProvider.GROQ: GroqProvider,
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.OPENROUTER: OpenRouterProvider,
    }

    @classmethod
    def create_provider(
        cls,
        provider: LLMProvider,
        temperature: float = 0.2,
        model_name: str | None = None,
    ) -> BaseLLMProvider:
        """
        Instantiates the chosen provider.

        Args:
            provider    : LLMProvider.GOOGLE | .GROQ | .OPENAI | .OPENROUTER
            temperature : 0.0 – 1.0 (default 0.2)
            model_name  : overrides the provider's default model (optional)

        Returns:
            An instance of the selected LLM provider, ready to create LLMs and agents.
        """
        provider_class = cls._providers.get(provider)
        if not provider_class:
            raise ValueError(
                f"Provider '{provider}' not supported. Options: {[p.value for p in LLMProvider]}"
            )
        return provider_class(temperature=temperature, model_name=model_name)

    @classmethod
    def from_env(cls) -> BaseLLMProvider:
        """
        Reads environment variables and instantiates the corresponding provider.

        Environment variables:
            LLM_PROVIDER    : "google" | "groq" | "openai" | "openrouter"  (default: openai)
            LLM_MODEL       : model name (optional)
            LLM_TEMPERATURE : float 0.0–1.0 (optional, default: 0.2)

        Args:
            None (reads from environment)

        Returns:
            An instance of the LLM provider specified in the environment variables.
        """
        try:
            provider_name = validate_provider(os.getenv("LLM_PROVIDER"))
            provider = LLMProvider(provider_name)
        except ValueError:
            provider_name = os.getenv("LLM_PROVIDER", "openai")
            valids = [p.value for p in LLMProvider]
            print(
                f"⚠️  LLM_PROVIDER='{provider_name}' invalid. "
                f"Accepted values: {valids}. Using 'openai'."
            )
            provider = LLMProvider.OPENAI

        model_name = os.getenv("LLM_MODEL") or None
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

        return cls.create_provider(provider, temperature, model_name)


# ============================================================================
# Convenience Functions (Public API of the module)
# ============================================================================


def get_llm(
    provider: LLMProvider | None = None,
    temperature: float = 0.2,
    model_name: str | None = None,
) -> Any:
    """
    Returns a ready-to-use LLM.

    If `provider` is not specified, reads LLM_PROVIDER from the environment.

    Args:
        provider: LLMProvider enum value to specify the provider (optional)
        temperature: Sampling temperature for the LLM (default 0.2)
        model_name: Specific model name to use (overrides provider default, optional)

    Returns:
        An instance of the LLM from the specified provider, configured with the given temperature and model name.

    Examples:
        llm = get_llm()                                        # via .env
        llm = get_llm(provider=LLMProvider.OPENAI)             # gpt-4.1
        llm = get_llm(provider=LLMProvider.OPENAI,
                      model_name="gpt-4o-mini",
                      temperature=0.5)
        llm = get_llm(provider=LLMProvider.GOOGLE, temperature=0.7)
    """
    if provider is None:
        llm_provider = LLMFactory.from_env()
    else:
        llm_provider = LLMFactory.create_provider(provider, temperature, model_name)
    return llm_provider.get_llm()


def create_agent_easy(
    tools: list,
    system_prompt: str,
    provider: LLMProvider | None = None,
    temperature: float = 0.2,
    model_name: str | None = None,
    name: str | None = None,
) -> Any:
    """
    Creates an agent with linked tools.

    If `provider` is not specified, reads LLM_PROVIDER from the environment.

    Args:
        tools: List of LangChain tools to bind to the LLM.
        system_prompt: The system prompt to use for the agent.
        provider: LLMProvider enum value to specify the provider (optional)
        temperature: Sampling temperature for the LLM (default 0.2)
        model_name: Specific model name to use (overrides provider default, optional)
        name: An optional name for the `CompiledStateGraph`

    Returns:
        An agent instance with the LLM and tools ready for use.

    Examples:
        agent = create_agent_easy(tools, prompt)
        agent = create_agent_easy(tools, prompt,
                                  provider=LLMProvider.OPENAI,
                                  model_name="gpt-4o")
    """
    if provider is None:
        llm_provider = LLMFactory.from_env()
    else:
        llm_provider = LLMFactory.create_provider(provider, temperature, model_name)
    return llm_provider.create_agent_with_tools(tools=tools, system_prompt=system_prompt, name=name)


T = TypeVar("T")


def llm_call(
    prompt: str,
    temperature: float = 0.2,
    response_schema: type[T] | None = None,
) -> str | T:
    """Wrapper for LLM calls with multi-provider support and structured output.

    Env vars:
        LLM_PROVIDER: 'openai' | 'google' | 'groq' | 'openrouter'  (default: 'openai')
        LLM_MODEL:    model name (e.g. 'gpt-4o', 'google/gemini-2.5-flash', 'llama-3.3-70b-versatile')
    """
    try:
        provider = validate_provider(os.getenv("LLM_PROVIDER"))
        model = os.getenv("LLM_MODEL", "")
        llm_instance = get_llm(
            provider=LLMProvider(provider) if provider else None,
            model_name=model or None,
            temperature=temperature,
        )
        if response_schema is not None:
            structured_llm = llm_instance.with_structured_output(response_schema)
            return structured_llm.invoke(prompt)
        resp = llm_instance.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        print(f"   \u26a0\ufe0f  LLM error: {e}")
        return None if response_schema else ""


def parse_json_safe(text: str, default: Any = None) -> Any:
    """Extract and parse JSON from text; delegates to core.utils.parse_json_safe."""
    from ...core.utils import parse_json_safe as _parse

    return _parse(text, default)


# ============================================================================
# FAST TEST  —  python3 llm_providers.py
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 TESTING LLM PROVIDERS")
    print("=" * 60)

    tests = [
        ("1️⃣  Via environment variable (LLM_PROVIDER)", None, None, None),
        ("2️⃣  Google (default)", LLMProvider.GOOGLE, None, None),
        ("3️⃣  Groq (default)", LLMProvider.GROQ, None, None),
        ("4️⃣  OpenAI — gpt-4.1 (default)", LLMProvider.OPENAI, None, None),
        (
            "5️⃣  OpenAI — gpt-4o-mini (explicit)",
            LLMProvider.OPENAI,
            "gpt-4o-mini",
            0.5,
        ),
        (
            "6️⃣  OpenRouter — google/gemini-2.5-flash (default)",
            LLMProvider.OPENROUTER,
            None,
            None,
        ),
    ]

    for description, prov, model, temp in tests:
        print(f"\n{description}")
        try:
            kwargs: dict = {}
            if prov is not None:
                kwargs["provider"] = prov
            if model is not None:
                kwargs["model_name"] = model
            if temp is not None:
                kwargs["temperature"] = temp
            llm = get_llm(**kwargs)
            model_name = getattr(llm, "model_name", getattr(llm, "model", "?"))
            print(f"   ✅ {type(llm).__name__} — model: {model_name}")
        except ValueError as e:
            print(f"   ⚠️  {e}")
        except Exception as e:
            print(f"   ❌ {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("💡 To use OpenAI in any agent:")
    print("   export LLM_PROVIDER=openai")
    print("   export OPENAI_API_KEY=sk-...")
    print("   export LLM_MODEL=gpt-4.1-mini   # optional")
    print("\n💡 To use OpenRouter in any agent:")
    print("   export LLM_PROVIDER=openrouter")
    print("   export OPENROUTER_API_KEY=sk-or-...")
    print("   export LLM_MODEL=anthropic/claude-3.5-sonnet   # optional")
