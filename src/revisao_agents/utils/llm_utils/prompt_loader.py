"""
Prompt loader utility.

Provides load_prompt() to read a YAML prompt file and render it
by substituting {variables} with keyword arguments.

Usage:
    from ..utils.llm_utils.prompt_loader import load_prompt

    prompt = load_prompt("academic/initial_plan", theme=theme, ctx=ctx)
    resp = get_llm(temperature=prompt.temperature).invoke(prompt.text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Root of the prompts directory (sibling folder of revisao_agents/)
_PROMPTS_ROOT = Path(__file__).parent.parent.parent / "prompts"


@dataclass
class Prompt:
    """Rendered prompt ready to send to an LLM."""

    name: str
    text: str
    temperature: float = 0.3
    metadata: dict = field(default_factory=dict)


@lru_cache(maxsize=128)
def _load_yaml_raw(path: Path) -> dict:
    """Loads a YAML file and caches its raw content."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompt(prompt_path: str, **variables: Any) -> Prompt:
    """
    Load a YAML prompt file and substitute {variables} in its 'system' field.

    Args:
        prompt_path: Relative path from the prompts/ root, without .yaml extension.
                     Examples: "academic/initial_plan", "common/interview"
        **variables: Key-value pairs used to fill {placeholders} in the template.

    Returns:
        A Prompt dataclass with .text (rendered string) and .temperature.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        KeyError: If a required placeholder is missing from variables.
    """
    yaml_path = _PROMPTS_ROOT / f"{prompt_path}.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {yaml_path}\n  Searched under: {_PROMPTS_ROOT}"
        )

    raw = _load_yaml_raw(yaml_path)

    template: str = raw.get("system", "")
    temperature: float = float(raw.get("temperature", 0.3))
    name: str = raw.get("name", prompt_path)

    # Collect all extra top-level string fields (e.g. instructions_academic)
    extra_fields = {
        k: v
        for k, v in raw.items()
        if k
        not in {
            "name",
            "description",
            "version",
            "temperature",
            "system",
            "last_updated",
        }
        and isinstance(v, str)
    }
    all_vars = {**extra_fields, **variables}

    try:
        rendered = template.format(**all_vars)
    except KeyError as e:
        missing = e.args[0]
        # Collect all placeholders present in the template for a helpful error
        placeholders = re.findall(r"\{(\w+)\}", template)
        raise KeyError(
            f"Missing variable '{missing}' when rendering prompt '{name}'.\n"
            f"  Template placeholders: {sorted(set(placeholders))}\n"
            f"  Variables provided:    {sorted(all_vars.keys())}"
        ) from e

    # If a language is provided, prepend a mandatory enforcement header so the LLM
    # always writes exclusively in the requested language and never mixes idioms.
    lang = variables.get("language") or extra_fields.get("language")
    if lang:
        _LANG_LABELS = {
            "pt": "Brazilian Portuguese (pt-BR)",
            "en": "English",
        }
        lang_label = _LANG_LABELS.get(str(lang).lower(), str(lang))
        lang_header = (
            f"LANGUAGE ENFORCEMENT — MANDATORY:\n"
            f"Write ALL output exclusively in {lang_label}. "
            f"Do NOT mix languages. Do NOT use any other language for any sentence, heading, "
            f"explanation, or technical term. This rule overrides all other instructions.\n"
            f"{'─' * 60}\n\n"
        )
        rendered = lang_header + rendered

    # Metadata for debugging / logging
    metadata = {
        "description": raw.get("description", ""),
        "version": raw.get("version", ""),
        "source": str(yaml_path.relative_to(_PROMPTS_ROOT.parent.parent)),
    }

    return Prompt(name=name, text=rendered.strip(), temperature=temperature, metadata=metadata)


def get_prompt_field(prompt_path: str, field_name: str, **variables: Any) -> str:
    """
    Load a specific named field (not 'system') from a YAML prompt file.

    Useful for multi-section YAML files like common/interview.yaml that have
    both 'instructions_academic' and 'instructions_technical' fields.

    Args:
        prompt_path: Relative path to the YAML file (without .yaml).
        field_name:  Name of the field to read (e.g. "instructions_academic").
        **variables: Substitution variables.

    Returns:
        Rendered string.
    """
    yaml_path = _PROMPTS_ROOT / f"{prompt_path}.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {yaml_path}")

    raw = _load_yaml_raw(yaml_path)
    template: str = raw.get(field_name, "")

    if not template:
        raise ValueError(
            f"Field '{field_name}' not found in '{prompt_path}.yaml'.\n"
            f"  Available fields: {list(raw.keys())}"
        )

    return template.format(**variables).strip()
