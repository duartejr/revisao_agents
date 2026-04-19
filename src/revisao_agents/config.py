# 1. Standard Library Imports
import os
from typing import Any, Literal, TypeVar, cast

from dotenv import load_dotenv

# 2. Third-Party Imports
from pydantic import BaseModel, Field, field_validator

# 3. Internal Imports
from .core.utils import parse_json_safe  # noqa: F401 — re-export

load_dotenv()


def _env_clean(name: str, default: str = "") -> str:
    """Read env var and strip accidental wrapping quotes/whitespace.

    Args:
        name: name of the environment variable to read
        default: default value if env var is not set

    Returns:
        The cleaned value of the environment variable.
    """
    return os.getenv(name, default).strip().strip("'").strip('"')


# ── Configuration Constants ────────────────────────────────────────────────
VECTOR_DB_PATH = _env_clean("VECTOR_DB_PATH", "./vector_db/vector_index")
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# MongoDB Atlas
MONGODB_URI = _env_clean("MONGODB_URI", "")
MONGODB_DB = _env_clean("MONGODB_DB", "review_agent")
MONGODB_COLLECTION = _env_clean("MONGODB_COLLECTION", "chunks")
VECTOR_INDEX_NAME = "vector_index"

# OpenAI
OPENAI_API_KEY = _env_clean("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Parameters to techinical extraction and plan generation
TECHNICAL_MAX_RESULTS = 10
MAX_URLS_EXTRACT = 7
CTX_PLAN_CHARS = 1200
CTX_ABSTRACT_CHARS = 1400
MIN_SECTION_PARAGRAPHS = 8
MAX_IMAGES_SECTION = 6
DELAY_BETWEEN_SECTIONS = 5
MAX_REACT_ITERATIONS = 2
EXTRACT_MIN_CHARS = 500
SNIPPET_MIN_SCORE = 0.7

# Chunking and embeddings
CHUNK_SIZE = 2400
CHUNK_OVERLAP = 480
TOP_K_WRITER = 6
TOP_K_OBSERVATION = 5
TOP_K_VERIFICATION = 6
MAX_CORPUS_PROMPT = 25000
CHUNK_MAX_CHARS = 600
MAX_CHUNKS_TOTAL = 100

# Anchors verify similarity threshold (cosine similarity)
ANCHOR_MIN_SIM = 0.82

HIST_MAX_TURNS = 6
PLAN_MAX_CHARS = 3000
CHUNKS_PER_QUERY = 5
JUDGE_TOP_K = 12
JUDGE_MAX_CORPUS_CHARS = 8000

# Domains
PRIORITY_DOMAINS = [
    ".pdf",
    "doi.org",
    "scielo",
    "copernicus",
    "arxiv.org",
    "researchgate.net",
    "springer.com",
    "elsevier.com",
    "mdpi.com",
    "nature.com",
    "wiley.com",
    "tandfonline.com",
    "cambridge.org",
    "ieee.org",
    "sciencedirect.com",
    "semanticscholar.org",
    "bvsalud.org",
    "repositorio",
    "teses.usp",
    "bdtd",
    "ufsc.br",
    "usp.br",
    "unicamp.br",
    "ufrj.br",
    "inpe.br",
    "ufrgs.br",
    "ufmg.br",
    "ufc.br",
    "ufpe.br",
]

BLOCKED_DOMAINS_EXTRACT = [
    "jstor.org",
    "proquest.com",
    "web.b.ebscohost.com",
]

# Closing remarks
CLOSING_REMARKS = {
    "ok",
    "pronto",
    "pode continuar",
    "esta bom",
    "ta bom",
    "suficiente",
    "chega",
    "finalizar",
    "encerrar",
    "avancar",
    "proximo",
    "continua",
    "continuar",
    "aceito",
    "aprovado",
    "finaliza",
    "terminar",
}

# Output path constants
PLANS_DIR = _env_clean("PLANS_DIR", "./plans")
REVIEWS_DIR = _env_clean("REVIEWS_DIR", "./reviews")
SEARCH_LOGS_DIR = _env_clean("SEARCH_LOGS_DIR", "./search_logs")
CHUNKS_CACHE_DIR = _env_clean("CHUNKS_CACHE_DIR", "./chunks_cache")


def ensure_runtime_dirs() -> None:
    """Ensure that all necessary runtime output directories exist.

    Creates ``PLANS_DIR``, ``REVIEWS_DIR``, ``SEARCH_LOGS_DIR``, and
    ``CHUNKS_CACHE_DIR`` if they do not already exist.  Safe to call
    multiple times (uses ``exist_ok=True``).

    Returns:
        None
    """
    for directory in [PLANS_DIR, REVIEWS_DIR, SEARCH_LOGS_DIR, CHUNKS_CACHE_DIR]:
        os.makedirs(directory, exist_ok=True)


# ── Runtime config validation ───────────────────────────────────────────────

_PROVIDER_ENV_KEYS = {
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

_CANONICAL_PROVIDERS = frozenset(_PROVIDER_ENV_KEYS.keys())


def validate_provider(value: str | None) -> str:
    """Validate and canonicalize the LLM provider name from environment variable.

    Args:
        value: The raw provider name from the environment variable.

    Returns:
        The canonicalized provider name (lowercase) if valid.

    Raises:
        ValueError: If the provider name is not supported.
    """
    canonical = (value or "").strip().lower()

    if not canonical:
        return "openai"

    if canonical not in _CANONICAL_PROVIDERS:
        supported = " | ".join(sorted(_CANONICAL_PROVIDERS))
        raise ValueError(f"LLM_PROVIDER '{value}' is not supported. Accepted: {supported}")
    return canonical


def get_runtime_config_summary() -> dict:
    """Return concise runtime config status used by CLI/UI startup diagnostics.

    Args:
        None

    Returns:
        Dict with keys:
            - llm_provider: normalized provider name (lowercase), defaults to 'openai' if invalid
            - llm_model: model name from provider instance, or provider default if not specified
            - llm_provider_key: provider API env var name
            - llm_provider_key_present: bool indicating if the key is set
            - llm_provider_error: always empty string (fallback handled internally)
            - mongodb_uri_present: bool indicating if MONGODB_URI is set
            - tavily_key_present: bool indicating if TAVILY_API_KEY is set
            - openai_key_present: bool indicating if OPENAI_API_KEY is set
    """
    from .utils.llm_utils.llm_providers import LLMFactory

    provider_error = ""
    try:
        llm_provider_obj = LLMFactory.from_env()
        provider = llm_provider_obj.__class__.__name__.replace("Provider", "").lower()
        model = llm_provider_obj.model_name
    except ValueError as exc:
        provider_error = f"LLM_PROVIDER error: {exc}"
        provider = "<invalid-provider>"
        model = "<unknown>"

    provider_key_name = _PROVIDER_ENV_KEYS.get(provider, "")
    provider_key_ok = bool(_env_clean(provider_key_name, "")) if provider_key_name else False

    mongodb_uri = _env_clean("MONGODB_URI", "")
    openai_key = _env_clean("OPENAI_API_KEY", "")

    return {
        "llm_provider": provider,
        "llm_model": model,
        "llm_provider_key": provider_key_name or "<invalid-provider>",
        "llm_provider_key_present": provider_key_ok,
        "llm_provider_error": provider_error,
        "mongodb_uri_present": bool(mongodb_uri),
        "tavily_key_present": bool(_env_clean("TAVILY_API_KEY", "")),
        "openai_key_present": bool(openai_key),
    }


def print_runtime_config_summary() -> None:
    """Print a one-block startup summary of integration readiness."""
    summary = get_runtime_config_summary()
    print("\n" + "-" * 70)
    print("ENVIRONMENT — CONFIGURATION SUMMARY")
    print("-" * 70)
    print(f"LLM_PROVIDER          : {summary['llm_provider']}")
    if summary.get("llm_provider_error"):
        print(f"LLM_PROVIDER error    : {summary['llm_provider_error']}")
    print(f"LLM_MODEL             : {summary['llm_model']}")
    print(
        f"Provider key     : {summary['llm_provider_key']} "
        f"({'OK' if summary['llm_provider_key_present'] else 'MISSING'})"
    )
    print(f"MongoDB URI           : {'OK' if summary['mongodb_uri_present'] else 'MISSING'}")
    print(f"Tavily API Key        : {'OK' if summary['tavily_key_present'] else 'MISSING'}")
    print(f"OpenAI (embeddings)   : {'OK' if summary['openai_key_present'] else 'MISSING'}")
    print("-" * 70)


def validate_runtime_config(strict: bool = False) -> list[str]:
    """
    Validate that all required runtime configuration is present for the agent to function.

    This function checks that the following environment variables are set and valid:
      - MONGODB_URI: Required for vector corpus storage and all core features.
      - TAVILY_API_KEY: Required for web search and evidence retrieval.
      - OPENAI_API_KEY: Always required for embeddings, regardless of LLM provider.
      - LLM_PROVIDER and provider-specific key: Required for LLM completions (see supported providers).

    Args:
        strict (bool): If True, raise ValueError on any missing requirement; otherwise, return a list of issues.

    Returns:
        list[str]: List of strings describing any configuration issues found (empty if all required config is present).

    Raises:
        ValueError: If strict is True and any required configuration is missing.

    Example:
        >>> validate_runtime_config(strict=True)
        # Raises ValueError if any required config is missing
    """
    issues: list[str] = []

    try:
        provider = validate_provider(os.getenv("LLM_PROVIDER"))
    except ValueError as e:
        issues.append(f"LLM_PROVIDER error: {e}")
        provider = None

    if provider is not None:
        key_name = _PROVIDER_ENV_KEYS[
            provider
        ]  # safe — validate_provider already guaranteed it's valid
        if key_name != "OPENAI_API_KEY" and not _env_clean(key_name, ""):
            issues.append(f"Missing key for current provider: {key_name}")

    mongodb_uri = _env_clean("MONGODB_URI", "")
    openai_key = _env_clean("OPENAI_API_KEY", "")
    tavily_key = _env_clean("TAVILY_API_KEY", "")

    if not mongodb_uri:
        issues.append("MONGODB_URI missing")

    if not tavily_key:
        issues.append("TAVILY_API_KEY missing")

    if not openai_key:
        issues.append("Missing OPENAI_API_KEY (required for embeddings)")

    if strict and issues:
        raise ValueError("; ".join(issues))

    return issues


# ── LLM helpers ──────────────────────────────────────────────────────────────


def get_llm(temperature: float = 0.3) -> Any:
    """
    Returns a language model configured with available tools based on environment variables.

    This function attempts to load a provider-specific LLM (Google, Groq, OpenAI, or
    OpenRouter) and bind all discovered tools to it. If the primary utility
    module is missing, it falls back to a default Google Generative AI instance.

    Args:
        temperature (float): The sampling temperature to use, ranging from 0.0 to 1.0.
            Defaults to 0.3.

    Returns:
        Any: A LangChain-compatible LLM object (BaseChatModel) with bound tools.
    """
    try:
        from .tools import get_all_tools
        from .utils.llm_utils.llm_providers import LLMProvider
        from .utils.llm_utils.llm_providers import get_llm as _get_llm

        provider = LLMProvider(validate_provider(os.getenv("LLM_PROVIDER")))
        print(f"   🤖 Using LLM: {provider.name} (temp={temperature})")
        llm = _get_llm(provider=provider, temperature=temperature)

        # Bind all available tools to the LLM
        tools = get_all_tools()
        llm_with_tools = llm.bind_tools(tools)
        return llm_with_tools
    except ImportError:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        )
        # Also bind tools even in fallback
        try:
            from .tools import get_all_tools

            tools = get_all_tools()
            return llm.bind_tools(tools)
        except Exception:
            return llm


T = TypeVar("T", bound=BaseModel)


class LLMInvocationError(RuntimeError):
    """Raised when an LLM call fails to execute or parse as requested."""


def llm_call(
    prompt: str,
    temperature: float = 0.2,
    response_schema: type[T] | None = None,
) -> str | T:
    """
    Wrapper for LLM calls with multi-provider support and structured output.

    This function routes requests to the configured provider, automatically
    injects date context, and handles structured data extraction if a
    response schema is provided.

    Environment Variables:
        LLM_PROVIDER: 'openai' | 'google' | 'groq' | 'openrouter' (default: 'openai')
        LLM_MODEL: Specific model name (e.g., 'gpt-4o', 'gemini-2.5-flash',
                  'llama-3.3-70b-versatile', 'anthropic/claude-3.5-sonnet')

    Args:
        prompt (str): The text instruction or question for the LLM.
        temperature (float): Sampling temperature (0.0 to 1.0). Defaults to 0.2.
        response_schema (Optional[Type[T]]): A Pydantic model or class for
            structured output. If provided, the return type will match the schema.

    Returns:
        Union[str, T]: The string content from the LLM or an instance of
            the response_schema.

    Raises:
        LLMInvocationError: If the provider fails or an import error occurs.
    """
    from .utils.llm_utils.date_context import add_date_context_to_prompt

    try:
        provider = validate_provider(os.getenv("LLM_PROVIDER"))
        model = os.getenv("LLM_MODEL", "") or "<default>"

        # Add current date context to ensure agents know today's date
        prompt_with_date = add_date_context_to_prompt(prompt)

        from .utils.llm_utils.llm_providers import get_llm as _provider_get_llm

        llm = _provider_get_llm(temperature=temperature)

        if response_schema is not None:
            structured_llm = llm.with_structured_output(response_schema)
            return structured_llm.invoke(prompt_with_date)

        resp = llm.invoke(prompt_with_date)
        return resp.content if hasattr(resp, "content") else str(resp)

    except Exception as e:
        provider = os.getenv("LLM_PROVIDER", "openai")
        model = os.getenv("LLM_MODEL", "") or "<default>"
        msg = f"LLM call failed [{provider}/{model}]"
        print(f"   ⚠️  {msg}: {e}")
        raise LLMInvocationError(msg) from e


def get_checkpointer_vars() -> dict:
    """Helper to expose checkpointer config variables for testing and graph construction.

    Returns:
        dict: A dictionary containing the checkpoint type and path from environment variables.
    """
    return {
        "CHECKPOINT_TYPE": os.getenv("CHECKPOINT_TYPE", "memory").lower(),
        "CHECKPOINT_PATH": os.getenv("CHECKPOINT_PATH", "checkpoints/checkpoints.db"),
    }


# Valid depth values accepted by the Tavily Search API, mirroring the Literal
# constraint on TavilySearchConfig.depth and used at runtime for env-var validation.
_TAVILY_VALID_DEPTHS: tuple[str, ...] = ("ultra-fast", "fast", "basic", "advanced")


class TavilySearchConfig(BaseModel):
    """Configuration model for Tavily search parameters.

    Attributes:
        depth: Search depth for Tavily. One of ``'ultra-fast'``, ``'fast'``,
            ``'basic'`` (default), or ``'advanced'``. ``'advanced'`` costs
            2 API credits per request.
        num_results: Number of results to retrieve per query (1–10).
        include_answer: Whether to request an LLM-generated answer alongside
            the search results.
        include_usage: Whether to request credit usage metadata in the API
            response. When ``True``, each response includes a ``usage`` dict
            and a top-level ``request_id`` field.
    """

    depth: Literal["ultra-fast", "fast", "basic", "advanced"] = Field(
        default="basic",
        description=(
            "Search depth setting for Tavily. 'ultra-fast' prioritizes speed with fewer results, "
            "'fast' provides a balance, 'basic' is the default for general use, and 'advanced' "
            "retrieves more comprehensive results at the cost of speed."
        ),
    )

    num_results: int = Field(
        10,
        description="Number of search results to retrieve from Tavily. Adjust based on needs and API limits.",
        ge=1,
        le=10,
    )
    include_answer: bool = Field(
        True,
        description="Whether to include the answer in the search results. If False, only metadata will be returned.",
    )
    include_usage: bool = Field(
        True,
        description="Whether to include credit usage information in the Tavily API response.",
    )

    @field_validator("include_answer")
    def validate_include_answer(cls, v):
        if not isinstance(v, bool):
            raise ValueError("include_answer must be a boolean value")
        return v

    @classmethod
    def load_from_env(cls) -> "TavilySearchConfig":
        """Load Tavily search configuration from environment variables.

        Environment Variables:
            TAVILY_SEARCH_DEPTH: Search depth ('ultra-fast', 'fast', 'basic', 'advanced')
            TAVILY_NUM_RESULTS: Number of results to retrieve (1-10)
            TAVILY_INCLUDE_ANSWER: Whether to include answers in results ('true'/'false')
            TAVILY_INCLUDE_USAGE: Whether to include credit usage in the response ('true'/'false')

        Returns:
            A ``TavilySearchConfig`` populated from the current environment.

        Raises:
            ValueError: If ``TAVILY_SEARCH_DEPTH`` is not one of the accepted
                depth strings (``'ultra-fast'``, ``'fast'``, ``'basic'``,
                ``'advanced'``).
            ValueError: If ``TAVILY_NUM_RESULTS`` cannot be parsed as an integer.
            ValueError: If ``TAVILY_NUM_RESULTS`` is outside the range 1–10.
        """
        depth_raw = os.getenv("TAVILY_SEARCH_DEPTH", "basic").lower()
        if depth_raw not in _TAVILY_VALID_DEPTHS:
            raise ValueError(
                f"Invalid TAVILY_SEARCH_DEPTH: {depth_raw!r}. Must be one of {_TAVILY_VALID_DEPTHS}"
            )
        depth = cast(Literal["ultra-fast", "fast", "basic", "advanced"], depth_raw)
        num_results_str = os.getenv("TAVILY_NUM_RESULTS", "5")
        include_answer_str = os.getenv("TAVILY_INCLUDE_ANSWER", "true").lower()
        include_usage_str = os.getenv("TAVILY_INCLUDE_USAGE", "true").lower()

        try:
            num_results = int(num_results_str)
        except ValueError as err:
            raise ValueError(
                f"Invalid TAVILY_NUM_RESULTS: {num_results_str!r} is not an integer"
            ) from err
        if not (1 <= num_results <= 10):
            raise ValueError(
                f"Invalid TAVILY_NUM_RESULTS: {num_results} is out of range. Must be between 1 and 10."
            )

        include_answer = include_answer_str in ("true", "1", "yes")
        include_usage = include_usage_str in ("true", "1", "yes")

        return cls(
            depth=depth,
            num_results=num_results,
            include_answer=include_answer,
            include_usage=include_usage,
        )


TAVILY_CONFIG = TavilySearchConfig.load_from_env()
