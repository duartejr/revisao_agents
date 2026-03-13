import os
from typing import Optional, Type, TypeVar, Union

from dotenv import load_dotenv

load_dotenv()


def _env_clean(name: str, default: str = "") -> str:
    """Read env var and strip accidental wrapping quotes/whitespace."""
    return os.getenv(name, default).strip().strip("'").strip('"')

# Caminhos e modelos
VECTOR_DB_PATH   = os.getenv("VECTOR_DB_PATH", "./vector_db_suelen/faiss_index")
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# MongoDB Atlas
MONGODB_URI        = _env_clean("MONGODB_URI", "")
MONGODB_DB         = os.getenv("MONGODB_DB", "revisao_agent")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chunks")
VECTOR_INDEX_NAME  = "vector_index"

# OpenAI
OPENAI_API_KEY         = _env_clean("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Parâmetros de busca técnica
TECHNICAL_MAX_RESULTS  = 10
MAX_URLS_EXTRACT     = 7
CTX_PLANO_CHARS      = 1200
CTX_RESUMO_CHARS     = 1400
SECAO_MIN_PARAGRAFOS = 8
MAX_IMAGES_SECTION  = 6
DELAY_ENTRE_SECOES   = 5
MAX_REACT_ITERATIONS = 2
EXTRACT_MIN_CHARS    = 500
SNIPPET_MIN_SCORE    = 0.7

# Chunking e embeddings
CHUNK_SIZE        = 1200
CHUNK_OVERLAP     = 240
TOP_K_WRITER      = 6
TOP_K_OBSERVACAO  = 5
TOP_K_VERIFICATION = 6
MAX_CORPUS_PROMPT = 25000
CHUNK_MAX_CHARS   = 600
MAX_CHUNKS_TOTAL  = 100
CHUNKS_CACHE_DIR  = os.getenv("CHUNKS_CACHE_DIR", "./chunks_cache")

# Verificação de âncoras
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

# Palavras de encerramento
ENCERRAMENTO = {
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
    """Return concise runtime config status used by CLI/UI startup diagnostics."""
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
    print("AMBIENTE — RESUMO DE CONFIGURAÇÃO")
    print("-" * 70)
    print(f"LLM_PROVIDER          : {summary['llm_provider']}")
    print(f"LLM_MODEL             : {summary['llm_model']}")
    print(
        f"Chave do provider     : {summary['llm_provider_key']} "
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
    """Validate runtime requirements and optionally raise on missing config."""
    issues: list[str] = []
    provider = _env_clean("LLM_PROVIDER", "groq").lower()
    mongodb_uri = _env_clean("MONGODB_URI", "")
    openai_key = _env_clean("OPENAI_API_KEY", "")

    if provider not in _PROVIDER_ENV_KEYS:
        issues.append(
            f"LLM_PROVIDER inválido: '{provider}'. Use: gemini | groq | openai | openrouter"
        )
    else:
        key_name = _PROVIDER_ENV_KEYS[provider]
        if not _env_clean(key_name, ""):
            issues.append(f"Chave ausente para provider atual: {key_name}")

    if require_mongodb and not mongodb_uri:
        issues.append("MONGODB_URI ausente")

    if require_tavily and not _env_clean("TAVILY_API_KEY", ""):
        issues.append("TAVILY_API_KEY ausente")

    if require_openai_embeddings and not openai_key:
        issues.append("OPENAI_API_KEY ausente (necessária para embeddings)")

    if strict and issues:
        raise ValueError("; ".join(issues))

    return issues


# ── LLM helpers ──────────────────────────────────────────────────────────────

def get_llm(temperature=0.3):
    """Retorna um modelo de linguagem configurado com ferramentas disponíveis."""
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
        print(f"   🤖 Usando LLM: {provider.name} (temp={temperature})")
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
    Wrapper para chamadas ao LLM com suporte a múltiplos providers e saída estruturada.

    Env vars:
        LLM_PROVIDER: 'openai' | 'gemini' | 'groq' | 'openrouter'  (default: 'groq')
        LLM_MODEL:    nome do modelo (ex: 'gpt-4o', 'gemini-2.0-flash', 'llama-3.3-70b-versatile', 'anthropic/claude-3.5-sonnet')
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
    """Extrai JSON de uma resposta LLM mesmo com texto ao redor."""
    import re, json
    match = re.search(r"\{[\s\S]*\}", texto)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None