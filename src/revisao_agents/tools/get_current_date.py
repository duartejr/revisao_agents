"""
Get Current Date Tool — provides agents with the current date/time.

This tool ensures agents are aware of the actual current date,
preventing them from ignoring data or information from after a hardcoded date.
"""

from langchain_core.tools import tool
from datetime import datetime, timezone


@tool
def get_current_date() -> str:
    """
    Get the current date and time.
    
    Returns the current date and time in ISO format and human-readable format.
    Use this tool to know today's date when making decisions about recent data.
    
    Returns:
        str: A string containing the current date and time in multiple formats.
    
    Example:
        >>> get_current_date()
        "Current date: 2026-03-10 (Monday)\nTime: 14:35:42 UTC"
    """
    now = datetime.now(timezone.utc)
    
    # Format the date in multiple ways for clarity
    iso_format = now.isoformat(timespec='seconds')
    day_name = now.strftime('%A')
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    
    return (
        f"Current date: {date_str} ({day_name})\n"
        f"Time: {time_str} UTC\n"
        f"ISO 8601: {iso_format}"
    )


__all__ = ["get_current_date"]
