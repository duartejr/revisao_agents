"""
Current date context utility — ensures agents know today's date.

This module provides functions to inject current date information into
LLM prompts and system messages, preventing agents from ignoring data
or information from after a hardcoded date.
"""

from datetime import UTC, datetime


def get_current_date_info() -> str:
    """
    Get a formatted string containing the current date and time.

    Returns:
        str: Current date in multiple formats for clarity.
    """
    now = datetime.now(UTC)
    day_name = now.strftime("%A")
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    return f"Current date: {date_str} ({day_name}) — {time_str} UTC"


def add_date_context_to_prompt(prompt_text: str) -> str:
    """
    Prepend current date information to a prompt to ensure the agent
    is aware of the current date when processing information.

    Args:
        prompt_text: The original prompt text

    Returns:
        str: Prompt with date context prepended
    """
    date_info = get_current_date_info()
    return f"⏰ {date_info}\n\n{prompt_text}"


def add_date_context_to_system_prompt(system_prompt: str) -> str:
    """
    Add current date awareness to a system prompt for agents.

    Args:
        system_prompt: The original system prompt

    Returns:
        str: System prompt with date context added
    """
    date_info = get_current_date_info()
    date_awareness = f"Current date: {date_info}"
    return f"{system_prompt}\n\n{date_awareness}"


__all__ = [
    "get_current_date_info",
    "add_date_context_to_prompt",
    "add_date_context_to_system_prompt",
]
