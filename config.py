# src/revisao_agents/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações do projeto (carregadas de .env ou variáveis de ambiente)."""

    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    temperature: float = 0.3

    # LangGraph / persistence
    checkpoint_type: str = "memory"  # ou "sqlite", "postgres", etc. no futuro

    # Paths (relativos à raiz do projeto)
    prompts_dir: str = "prompts/technical_writing"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()