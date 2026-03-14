# src/revisao_agents/tools/academic_corpus_search.py
"""
Tool oficial de busca no corpus MongoDB.
Wrapper simples e performático para uso direto pelos agents.
"""

from langchain_core.tools import tool
from typing import Optional

# Import da classe original (ainda na pasta utils por enquanto)
from ..utils.vector_utils.mongodb_corpus import CorpusMongoDB


@tool
def search_academic_corpus(
    query: str,
    limit: int = 5,
    section_title: str = "",
) -> str:
    """
    Busca no corpus MongoDB por conteúdo acadêmico relevante usando vector search.

    Args:
        query: Texto ou anchor a ser pesquisada (ex: "modelo de difusão estável").
        limit: Número máximo de fontes/chunks a retornar (padrão 5).
        section_title: Título da seção atual (usado apenas para log interno).

    Returns:
        String formatada pronta para colar no prompt do agent (com cabeçalhos de fontes).
        Inclui as fontes mais relevantes + contexto completo.
    """
    try:
        corpus = CorpusMongoDB()
        # Usa o render_prompt (melhor saída formatada do seu código original)
        contexto, urls_usadas, fonte_map = corpus.render_prompt(
            query=query,
            max_chars=8000,  # limite seguro para contexto de LLM
        )

        if not contexto.strip():
            return f"Nenhuma fonte relevante encontrada para: '{query}'"

        header = (
            f"=== FONTES ENCONTRADAS PARA: '{query}' ===\n"
            f"Seção: {section_title or 'Não informada'}\n"
            f"Total de chunks usados: {len(urls_usadas)}\n"
            f"{'='*60}\n\n"
        )

        return header + contexto

    except Exception as e:
        return f"Erro na busca do corpus: {str(e)}"