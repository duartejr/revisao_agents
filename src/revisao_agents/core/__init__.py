# src/revisao_agents/core/__init__.py
"""
Core modules shared across the revisao_agents package.
This includes Pydantic schemas and utilities that are not specific to agents/tools.
"""

# Re-export all modules for clean imports anywhere in the project
from .schemas import *
