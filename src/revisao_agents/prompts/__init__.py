# src/revisao_agents/prompts/__init__.py
"""
Project prompt hub.

All prompts are stored as YAML to simplify editing and maintenance. They are
loaded via utils.llm_utils.prompt_loader.load_prompt, not from this package's
Python namespace — the YAML subfolders here are read directly by file path.
"""
