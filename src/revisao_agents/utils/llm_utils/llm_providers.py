# llm_providers.py
"""
Sistema modular para alternar entre provedores de LLM (Gemini, Groq, OpenAI, OpenRouter).

Uso via variável de ambiente (recomendado):
    export LLM_PROVIDER=openai        # ou gemini / groq / openrouter
    export LLM_MODEL=gpt-4.1         # opcional -- sobrescreve o modelo padrão
    export LLM_TEMPERATURE=0.3        # opcional -- padrão: 0.2

Uso via código:
    from llm_providers import get_llm, LLMProvider
    llm = get_llm(provider=LLMProvider.OPENAI, temperature=0.4)
    llm = get_llm(provider=LLMProvider.OPENROUTER, model_name="google/gemini-2.0-flash-001")

Chaves de API necessárias no .env:
    GOOGLE_API_KEY      -> Gemini
    GROQ_API_KEY        -> Groq
    OPENAI_API_KEY      -> OpenAI
    OPENROUTER_API_KEY  -> OpenRouter
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional
from enum import Enum
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


# ============================================================================
# ENUM DE PROVEDORES
# ============================================================================

class LLMProvider(Enum):
    """Identificadores dos provedores suportados."""
    GEMINI    = "gemini"
    GROQ      = "groq"
    OPENAI    = "openai"
    OPENROUTER = "openrouter"


# ============================================================================
# CLASSE BASE
# ============================================================================

class BaseLLMProvider(ABC):
    """Interface comum para todos os provedores de LLM."""

    def __init__(self, temperature: float = 0.2, model_name: Optional[str] = None):
        self.temperature = temperature
        self.model_name  = model_name or self.get_default_model()
        self._llm        = None

    @abstractmethod
    def get_default_model(self) -> str:
        """Modelo padrão do provedor."""

    @abstractmethod
    def get_api_key(self) -> str:
        """Chave de API do provedor."""

    @abstractmethod
    def create_llm(self) -> Any:
        """Instancia e retorna o LLM."""

    def get_llm(self) -> Any:
        """Lazy loading — cria o LLM apenas na primeira chamada."""
        if self._llm is None:
            self._llm = self.create_llm()
        return self._llm

    def create_agent_with_tools(self, tools: List, system_prompt: str) -> Any:
        """Cria um agente ReAct com ferramentas vinculadas."""
        llm = self.get_llm()
        llm_with_tools = llm.bind_tools(tools)
        return create_agent(
            model=llm_with_tools,
            tools=tools,
            system_prompt=system_prompt,
        )


# ============================================================================
# PROVEDORES CONCRETOS
# ============================================================================

class GeminiProvider(BaseLLMProvider):
    """Google Gemini via langchain-google-genai."""

    def get_default_model(self) -> str:
        print('default model: gemini-2.5-flash')
        return "gemini-2.5-flash"

    def get_api_key(self) -> str:
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY não encontrada no .env")
        return key

    def create_llm(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=self.get_api_key(),
        )


class GroqProvider(BaseLLMProvider):
    """Groq via langchain-groq."""

    def get_default_model(self) -> str:
        # Outros disponíveis: llama-3.3-70b-versatile, mixtral-8x7b-32768
        return os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    def get_api_key(self) -> str:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY não encontrada no .env")
        return key

    def create_llm(self) -> ChatGroq:
        return ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            groq_api_key=self.get_api_key(),
        )


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI via langchain-openai.

    Modelos sugeridos:
        gpt-4.1          → mais capaz, ideal para escrita científica densa
        gpt-4.1-mini     → mais rápido e econômico, bom para tarefas estruturadas
        gpt-4o           → boa relação custo/qualidade
        gpt-4o-mini      → econômico para tarefas simples
        o3               → raciocínio avançado (mais lento)
        o4-mini          → raciocínio rápido

    Defina o modelo via variável de ambiente LLM_MODEL ou model_name=...
    """

    def get_default_model(self) -> str:
        model = os.getenv("LLM_MODEL", "gpt-4.1")
        print(f'default model: {model}')
        return model

    def get_api_key(self) -> str:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY não encontrada no .env")
        return key

    def create_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.get_api_key(),
        )


class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter via langchain-openai (OpenAI-compatible API).

    OpenRouter agrega múltiplos modelos. Modelos populares:
        google/gemini-2.0-flash-001      → rápido, versátil
        google/gemini-2.0-pro-001        → mais capaz
        anthropic/claude-3.5-sonnet      → excelente para escrita
        anthropic/claude-3-opus          → mais poderoso
        openai/gpt-4-turbo               → OpenAI via OpenRouter
        meta-llama/llama-3.3-70b-instruct → open-source potente

    Defina o modelo via variável de ambiente LLM_MODEL ou model_name=...
    Obtenha sua chave em: https://openrouter.ai/
    """

    def get_default_model(self) -> str:
        model = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-001")
        print(f'default model: {model}')
        return model

    def get_api_key(self) -> str:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY não encontrada no .env")
        return key

    def create_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.get_api_key(),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/duartejr/paper_reviwer",
                "X-Title": "Paper Reviewer"
            }
        )


# ============================================================================
# FACTORY
# ============================================================================

class LLMFactory:
    """Cria provedores de LLM por enum ou variável de ambiente."""

    _providers = {
        LLMProvider.GEMINI:     GeminiProvider,
        LLMProvider.GROQ:       GroqProvider,
        LLMProvider.OPENAI:     OpenAIProvider,
        LLMProvider.OPENROUTER: OpenRouterProvider,
    }

    @classmethod
    def create_provider(
        cls,
        provider: LLMProvider,
        temperature: float = 0.2,
        model_name: Optional[str] = None,
    ) -> BaseLLMProvider:
        """
        Instancia o provedor escolhido.

        Args:
            provider    : LLMProvider.GEMINI | .GROQ | .OPENAI | .OPENROUTER
            temperature : 0.0 – 1.0 (padrão 0.2)
            model_name  : sobrescreve o modelo padrão do provedor (opcional)
        """
        provider_class = cls._providers.get(provider)
        if not provider_class:
            raise ValueError(
                f"Provedor '{provider}' não suportado. "
                f"Opções: {[p.value for p in LLMProvider]}"
            )
        return provider_class(temperature=temperature, model_name=model_name)

    @classmethod
    def from_env(cls) -> BaseLLMProvider:
        """
        Lê as variáveis de ambiente e instancia o provedor correspondente.

        Variáveis:
            LLM_PROVIDER    : "gemini" | "groq" | "openai" | "openrouter"  (padrão: gemini)
            LLM_MODEL       : nome do modelo (opcional)
            LLM_TEMPERATURE : float 0.0–1.0 (opcional, padrão: 0.2)
        """
        provider_name = os.getenv("LLM_PROVIDER", "gemini").lower().strip()

        try:
            provider = LLMProvider(provider_name)
        except ValueError:
            validos = [p.value for p in LLMProvider]
            print(
                f"⚠️  LLM_PROVIDER='{provider_name}' inválido. "
                f"Valores aceitos: {validos}. Usando 'gemini'."
            )
            provider = LLMProvider.GEMINI

        model_name  = os.getenv("LLM_MODEL") or None
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

        return cls.create_provider(provider, temperature, model_name)


# ============================================================================
# FUNÇÕES DE CONVENIÊNCIA (API pública do módulo)
# ============================================================================

def get_llm(
    provider: Optional[LLMProvider] = None,
    temperature: float = 0.2,
    model_name: Optional[str] = None,
) -> Any:
    """
    Retorna um LLM pronto para uso.

    Se `provider` não for informado, lê LLM_PROVIDER do ambiente.

    Exemplos:
        llm = get_llm()                                        # via .env
        llm = get_llm(provider=LLMProvider.OPENAI)             # gpt-4.1
        llm = get_llm(provider=LLMProvider.OPENAI,
                      model_name="gpt-4o-mini",
                      temperature=0.5)
        llm = get_llm(provider=LLMProvider.GEMINI, temperature=0.7)
    """
    if provider is None:
        llm_provider = LLMFactory.from_env()
    else:
        llm_provider = LLMFactory.create_provider(provider, temperature, model_name)
    return llm_provider.get_llm()


def create_agent_easy(
    tools: List,
    system_prompt: str,
    provider: Optional[LLMProvider] = None,
    temperature: float = 0.2,
    model_name: Optional[str] = None,
) -> Any:
    """
    Cria um agente com ferramentas vinculadas.

    Se `provider` não for informado, lê LLM_PROVIDER do ambiente.

    Exemplos:
        agent = create_agent_easy(tools, prompt)
        agent = create_agent_easy(tools, prompt,
                                  provider=LLMProvider.OPENAI,
                                  model_name="gpt-4o")
    """
    if provider is None:
        llm_provider = LLMFactory.from_env()
    else:
        llm_provider = LLMFactory.create_provider(provider, temperature, model_name)
    return llm_provider.create_agent_with_tools(tools, system_prompt)


# ============================================================================
# TESTE RÁPIDO  —  python3 llm_providers.py
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 TESTANDO PROVEDORES DE LLM")
    print("=" * 60)

    testes = [
        ("1️⃣  Via variável de ambiente (LLM_PROVIDER)", None,                  None,         None),
        ("2️⃣  Gemini (padrão)",                         LLMProvider.GEMINI,   None,         None),
        ("3️⃣  Groq (padrão)",                           LLMProvider.GROQ,     None,         None),
        ("4️⃣  OpenAI — gpt-4.1 (padrão)",               LLMProvider.OPENAI,   None,         None),
        ("5️⃣  OpenAI — gpt-4o-mini (explícito)",        LLMProvider.OPENAI,   "gpt-4o-mini", 0.5),
        ("6️⃣  OpenRouter — google/gemini-2.0 (padrão)", LLMProvider.OPENROUTER, None,       None),
    ]

    for descricao, prov, model, temp in testes:
        print(f"\n{descricao}")
        try:
            kwargs: dict = {}
            if prov  is not None: kwargs["provider"]    = prov
            if model is not None: kwargs["model_name"]  = model
            if temp  is not None: kwargs["temperature"] = temp
            llm = get_llm(**kwargs)
            nome_modelo = getattr(llm, "model_name", getattr(llm, "model", "?"))
            print(f"   ✅ {type(llm).__name__} — modelo: {nome_modelo}")
        except ValueError as e:
            print(f"   ⚠️  {e}")
        except Exception as e:
            print(f"   ❌ {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("💡 Para usar OpenAI em qualquer agente:")
    print("   export LLM_PROVIDER=openai")
    print("   export OPENAI_API_KEY=sk-...")
    print("   export LLM_MODEL=gpt-4.1-mini   # opcional")
    print("\n💡 Para usar OpenRouter em qualquer agente:")
    print("   export LLM_PROVIDER=openrouter")
    print("   export OPENROUTER_API_KEY=sk-or-...")
    print("   export LLM_MODEL=anthropic/claude-3.5-sonnet   # opcional")


# ============================================================================
# HELPER FUNCTIONS FOR LLM CALLS
# ============================================================================

from typing import Type, TypeVar, Union

T = TypeVar("T")


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
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    model    = os.getenv("LLM_MODEL", _default_model(provider))

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
    print("=" * 60)