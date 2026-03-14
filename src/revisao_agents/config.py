import os
from typing import Any, Optional, Type, TypeVar, Union

from dotenv import load_dotenv

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

# Caminhos e modelos
VECTOR_DB_PATH   = _env_clean("VECTOR_DB_PATH", "./vector_db_suelen/faiss_index")
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# MongoDB Atlas
MONGODB_URI        = _env_clean("MONGODB_URI", "")
MONGODB_DB         = _env_clean("MONGODB_DB", "review_agent")
MONGODB_COLLECTION = _env_clean("MONGODB_COLLECTION", "chunks")
VECTOR_INDEX_NAME  = "vector_index"

# OpenAI
OPENAI_API_KEY         = _env_clean("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Parâmetros de busca técnica
TECHNICAL_MAX_RESULTS  = 10
MAX_URLS_EXTRACT     = 7
CTX_PLAN_CHARS      = 1200
CTX_ABSTRACT_CHARS     = 1400
MIN_SECTION_PARAGRAPHS = 8
MAX_IMAGES_SECTION  = 6
DELAY_BETWEEN_SECTIONS   = 5
MAX_REACT_ITERATIONS = 2
EXTRACT_MIN_CHARS    = 500
SNIPPET_MIN_SCORE    = 0.7

# Chunking e embeddings
CHUNK_SIZE        = 1200
CHUNK_OVERLAP     = 240
TOP_K_WRITER      = 6
TOP_K_OBSERVATION = 5
TOP_K_VERIFICATION = 6
MAX_CORPUS_PROMPT = 25000
CHUNK_MAX_CHARS   = 600
MAX_CHUNKS_TOTAL  = 100
CHUNKS_CACHE_DIR  = _env_clean("CHUNKS_CACHE_DIR", "./chunks_cache")

# Anchors verify similarity threshold (cosine similarity)
ANCHOR_MIN_SIM = 0.82

HIST_MAX_TURNS   = 6
PLAN_MAX_CHARS  = 3000
CHUNKS_PER_QUERY = 5
JUIZ_TOP_K       = 12
JUIZ_MAX_CORPUS_CHARS = 8000

# Domínios
PRIORITY_DOMAINS = [
    ".pdf", "doi.org", "scielo", "copernicus", "arxiv.org",
    "researchgate.net", "springer.com", "elsevier.com", "mdpi.com",
    "nature.com", "wiley.com", "tandfonline.com", "cambridge.org",
    "ieee.org", "sciencedirect.com", "semanticscholar.org",
    "bvsalud.org", "repositorio", "teses.usp", "bdtd",
    "ufsc.br", "usp.br", "unicamp.br", "ufrj.br", "inpe.br",
    "ufrgs.br", "ufmg.br", "ufc.br", "ufpe.br",
]

BLOCKED_DOMAINS_EXTRACT = [
    "jstor.org", "proquest.com", "web.b.ebscohost.com",
]

# closing remarks
CLOSING_REMARKS = {
    "ok", "pronto", "pode continuar", "esta bom", "ta bom", "suficiente",
    "chega", "finalizar", "encerrar", "avancar", "proximo", "continua",
    "continuar", "aceito", "aprovado", "finaliza", "terminar",
}


# ── Runtime config validation ───────────────────────────────────────────────

_PROVIDER_ENV_KEYS = {
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def get_runtime_config_summary() -> dict:
    """Return concise runtime config status used by CLI/UI startup diagnostics.
    
    Args:
        None
    Returns:
        Dict with keys:
            - llm_provider: normalized provider name (lowercase)
            - llm_model: model name or "<default>"
            - llm_provider_key: name of the env var for the provider's API key
            - llm_provider_key_present: bool indicating if the key is set
            - mongodb_uri_present: bool indicating if MONGODB_URI is set
            - tavily_key_present: bool indicating if TAVILY_API_KEY is set
            - openai_key_present: bool indicating if OPENAI_API_KEY is set
    """
    provider = _env_clean("LLM_PROVIDER", "groq").lower()
    model = _env_clean("LLM_MODEL", "") or "<default>"
    provider_key_name = _PROVIDER_ENV_KEYS.get(provider, "")
    provider_key_ok = bool(_env_clean(provider_key_name, "")) if provider_key_name else False

    mongodb_uri = _env_clean("MONGODB_URI", "")
    openai_key = _env_clean("OPENAI_API_KEY", "")

    return {
        "llm_provider": provider,
        "llm_model": model,
        "llm_provider_key": provider_key_name or "<invalid-provider>",
        "llm_provider_key_present": provider_key_ok,
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
    print(f"LLM_MODEL             : {summary['llm_model']}")
    print(
        f"Provider key     : {summary['llm_provider_key']} "
        f"({'OK' if summary['llm_provider_key_present'] else 'MISSING'})"
    )
    print(f"MongoDB URI           : {'OK' if summary['mongodb_uri_present'] else 'MISSING'}")
    print(f"Tavily API Key        : {'OK' if summary['tavily_key_present'] else 'MISSING'}")
    print(f"OpenAI (embeddings)   : {'OK' if summary['openai_key_present'] else 'MISSING'}")
    print("-" * 70)


def validate_runtime_config(
    require_mongodb: bool = False,
    require_tavily: bool = False,
    require_openai_embeddings: bool = False,
    strict: bool = False,
) -> list[str]:
    """Validate runtime requirements and optionally raise on missing config.
    
    Args:
        require_mongodb: if True, MONGODB_URI must be set
        require_tavily: if True, TAVILY_API_KEY must be set
        require_openai_embeddings: if True, OPENAI_API_KEY must be set
        strict: if True, raise ValueError on any missing requirement; otherwise return list of issues
    Returns:    
        List of strings describing any configuration issues found (empty if all good)
    """
    issues: list[str] = []
    provider = _env_clean("LLM_PROVIDER", "groq").lower()
    mongodb_uri = _env_clean("MONGODB_URI", "")
    openai_key = _env_clean("OPENAI_API_KEY", "")

    if provider not in _PROVIDER_ENV_KEYS:
        issues.append(
            f"LLM_PROVIDER invalid: '{provider}'. Use: gemini | groq | openai | openrouter"
        )
    else:
        key_name = _PROVIDER_ENV_KEYS[provider]
        if not _env_clean(key_name, ""):
            issues.append(f"Missing key for current provider: {key_name}")

    if require_mongodb and not mongodb_uri:
        issues.append("MONGODB_URI missing")

    if require_tavily and not _env_clean("TAVILY_API_KEY", ""):
        issues.append("TAVILY_API_KEY missing")

    if require_openai_embeddings and not openai_key:
        issues.append("Missing OPENAI_API_KEY (required for embeddings)")

    if strict and issues:
        raise ValueError("; ".join(issues))

    return issues


# ── LLM helpers ──────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.3) -> Any:
    """
    Returns a language model configured with available tools based on environment variables.

    This function attempts to load a provider-specific LLM (Gemini, Groq, OpenAI, or 
    OpenRouter) and bind all discovered tools to it. If the primary utility 
    module is missing, it falls back to a default Google Generative AI instance.

    Args:
        temperature (float): The sampling temperature to use, ranging from 0.0 to 1.0. 
            Defaults to 0.3.

    Returns:
        Any: A LangChain-compatible LLM object (BaseChatModel) with bound tools.
    """
    try:
        from .utils.llm_utils.llm_providers import get_llm as _get_llm, LLMProvider
        from .tools import get_all_tools
        
        provider_map = {
            "GEMINI": LLMProvider.GEMINI,
            "GROQ":   LLMProvider.GROQ,
            "OPENAI": LLMProvider.OPENAI,
            "OPENROUTER": LLMProvider.OPENROUTER,
        }
        provider = provider_map.get(
            os.getenv("LLM_PROVIDER", "GEMINI").upper().strip(),
            LLMProvider.GEMINI,
        )
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
        except:
            return llm


from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMInvocationError(RuntimeError):
    """Raised when an LLM call fails to execute or parse as requested."""


def llm_call(
    prompt: str,
    temperature: float = 0.2,
    response_schema: Optional[Type[T]] = None,
) -> Union[str, T]:
    """
    Wrapper for LLM calls with multi-provider support and structured output.

    This function routes requests to the configured provider, automatically 
    injects date context, and handles structured data extraction if a 
    response schema is provided.

    Environment Variables:
        LLM_PROVIDER: 'openai' | 'gemini' | 'groq' | 'openrouter' (default: 'groq')
        LLM_MODEL: Specific model name (e.g., 'gpt-4o', 'gemini-2.0-flash', 
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
    
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    model = os.getenv("LLM_MODEL", "") or "<default>"
    
    # Add current date context to ensure agents know today's date
    prompt_with_date = add_date_context_to_prompt(prompt)

    try:
        from .utils.llm_utils.llm_providers import get_llm as _provider_get_llm
        llm = _provider_get_llm(temperature=temperature)

        if response_schema is not None:
            structured_llm = llm.with_structured_output(response_schema)
            return structured_llm.invoke(prompt_with_date)

        resp = llm.invoke(prompt_with_date)
        return resp.content if hasattr(resp, "content") else str(resp)

    except Exception as e:
        msg = f"LLM call failed [{provider}/{model}]"
        print(f"   ⚠️  {msg}: {e}")
        raise LLMInvocationError(msg) from e


def parse_json_safe(texto: str) -> dict | None:
    """
    Safely extracts and parses a JSON object from a string, even if surrounded by text.
    
    This is particularly useful for cleaning LLM outputs where the model might 
    include conversational filler or markdown blocks (e.g., ```json ... ```) 
    alongside the raw JSON.

    Args:
        text (str): The raw string content received from the LLM.

    Returns:
        Optional[Dict[str, Any]]: A dictionary representing the parsed JSON if 
            successful; None if no valid JSON structure is found or if parsing fails.
    """
    import re, json
    match = re.search(r"\{[\s\S]*\}", texto)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None