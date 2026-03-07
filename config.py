import os
from dotenv import load_dotenv

load_dotenv()

# Caminhos e modelos
VECTOR_DB_PATH   = os.getenv("VECTOR_DB_PATH", "./vector_db_suelen/faiss_index")
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# MongoDB Atlas
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://usuario:senha@cluster.mongodb.net/")
MONGODB_DB  = os.getenv("MONGODB_DB", "revisao_agent")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chunks")
VECTOR_INDEX_NAME = "vector_index"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Parâmetros de busca técnica
TECNICO_MAX_RESULTS   = 10
MAX_URLS_EXTRACT      = 7
CTX_PLANO_CHARS       = 1200
CTX_RESUMO_CHARS      = 1400
SECAO_MIN_PARAGRAFOS  = 8
MAX_IMAGENS_SECAO     = 6
DELAY_ENTRE_SECOES    = 5
MAX_REACT_ITERATIONS  = 2
EXTRACT_MIN_CHARS     = 500
SNIPPET_MIN_SCORE     = 0.7  # antes 0.5


# Chunking e embeddings
CHUNK_SIZE            = 1200
CHUNK_OVERLAP         = 240
TOP_K_ESCRITA         = 6
TOP_K_OBSERVACAO      = 5
TOP_K_VERIFICACAO     = 6
MAX_CORPUS_PROMPT     = 25000
CHUNK_MAX_CHARS       = 600
MAX_CHUNKS_TOTAL      = 100
CHUNKS_CACHE_DIR      = os.getenv("CHUNKS_CACHE_DIR", "./chunks_cache")


# Verificação de âncoras
ANCORA_MIN_SIM_FAISS  = 0.82
ANCORA_MIN_SIM_FUZZY  = 0.72

HIST_MAX_TURNS        = 6
PLANO_MAX_CHARS       = 3000
CHUNKS_PER_QUERY      = 5

JUIZ_TOP_K            = 12
MAX_IMAGENS_SECAO     = 6

# Domínios
DOMINIOS_PRIORITARIOS = [
    ".pdf", "doi.org", "scielo", "copernicus", "arxiv.org",
    "researchgate.net", "springer.com", "elsevier.com", "mdpi.com",
    "nature.com", "wiley.com", "tandfonline.com", "cambridge.org",
    "ieee.org", "sciencedirect.com", "semanticscholar.org",
    "bvsalud.org", "repositorio", "teses.usp", "bdtd",
    "ufsc.br", "usp.br", "unicamp.br", "ufrj.br", "inpe.br",
    "ufrgs.br", "ufmg.br", "ufc.br", "ufpe.br",
]

DOMINIOS_BLOQUEADOS_EXTRACT = [
    "jstor.org", "proquest.com", "web.b.ebscohost.com",
]

# Palavras de encerramento (para outros fluxos)
ENCERRAMENTO = {
    "ok","pronto","pode continuar","esta bom","ta bom","suficiente",
    "chega","finalizar","encerrar","avancar","proximo","continua",
    "continuar","aceito","aprovado","finaliza","terminar",
}

# LLM provider
def get_llm(temperature=0.3):
    """Retorna um modelo de linguagem configurado."""
    try:
        from utils.llm_providers import get_llm as _get_llm, LLMProvider
        provider_map = {
            "GEMINI": LLMProvider.GEMINI,
            "GROQ":   LLMProvider.GROQ,
            "OPENAI": LLMProvider.OPENAI,
        }
        provider = provider_map.get(
            os.getenv("LLM_PROVIDER", "GEMINI").upper().strip(),
            LLMProvider.GEMINI,
        )
        print(f"   🤖 Usando LLM: {provider.name} (temp={temperature})")
        return _get_llm(provider=provider, temperature=temperature)
    except ImportError:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        )

import os
from typing import Type, TypeVar, Union, Optional
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def llm_call(
    prompt: str,
    temperature: float = 0.2,
    response_schema: Optional[Type[T]] = None,
) -> Union[str, T]:
    """
    Wrapper para chamadas ao LLM com suporte a múltiplos providers e saída estruturada.

    Env vars:
        LLM_PROVIDER: 'openai' | 'gemini' | 'groq'  (default: 'openai')
        LLM_MODEL:    nome do modelo (ex: 'gpt-4o', 'gemini-2.0-flash', 'llama-3.3-70b-versatile')

    Args:
        prompt:          Prompt a ser enviado ao LLM.
        temperature:     Temperatura da geração (default 0.2).
        response_schema: Classe Pydantic para estruturar a resposta (opcional).

    Returns:
        Instância do Pydantic fornecido, ou str se nenhum schema foi passado.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    model = os.getenv("LLM_MODEL", _default_model(provider))

    try:
        llm = _build_llm(provider, model, temperature)

        if response_schema is not None:
            structured_llm = llm.with_structured_output(response_schema)
            return structured_llm.invoke(prompt)

        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)

    except Exception as e:
        print(f"   ⚠️  LLM error [{provider}/{model}]: {e}")
        return None if response_schema else ""


# ── helpers ──────────────────────────────────────────────────────────────────

def _default_model(provider: str) -> str:
    defaults = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.5-flash",
        "groq":   "llama-3.3-70b-versatile",
    }
    return defaults.get(provider, "llama-3.3-70b-versatile")


def _build_llm(provider: str, model: str, temperature: float):
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature)

    raise ValueError(
        f"Provider '{provider}' não suportado. Use: openai | gemini | groq"
    )

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