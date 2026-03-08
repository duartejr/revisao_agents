# src/revisao_agents/core/__init__.py
"""
Módulos core compartilhados por todo o pacote revisao_agents.
Aqui ficam schemas Pydantic e utilities que não são específicas de agents/tools.
"""

# Re-exporta tudo para imports limpos em qualquer lugar do projeto
from .schemas import *