"""
Agents module - LangGraph node definitions for different review types.

This module contains all the node functions that make up the various
review workflows (academic, technical, technical writing, etc.)
"""

from .academic import (
    consulta_vetorial_node,
    plano_inicial_academico_node,
    refinar_consulta_academico_node,
    refinar_plano_academico_node,
    finalizar_plano_academico_node,
)

from .technical import (
    busca_tecnica_inicial_node,
    plano_inicial_tecnico_node,
    refinar_busca_tecnica_node,
    refinar_plano_tecnico_node,
    finalizar_plano_tecnico_node,
)

from .common import (
    human_pause_node,
    entrevista_node,
    roteador_entrevista,
)

from .technical_writing import (
    parsear_plano_node,
    escrever_secoes_node,
    consolidar_node,
)

__all__ = [
    # Academic agents
    "consulta_vetorial_node",
    "plano_inicial_academico_node",
    "refinar_consulta_academico_node",
    "refinar_plano_academico_node",
    "finalizar_plano_academico_node",
    # Technical agents
    "busca_tecnica_inicial_node",
    "plano_inicial_tecnico_node",
    "refinar_busca_tecnica_node",
    "refinar_plano_tecnico_node",
    "finalizar_plano_tecnico_node",
    # Common agents
    "human_pause_node",
    "entrevista_node",
    "roteador_entrevista",
    # Technical writing agents
    "parsear_plano_node",
    "escrever_secoes_node",
    "consolidar_node",
]
