"""
tests/unit/test_tools/test_registry.py

Unit tests for the tools registry.
"""

import pytest
from revisao_agents.tools.registry import get_all_tools


def test_get_all_tools_returns_list():
    tools = get_all_tools()
    assert isinstance(tools, list)


def test_tools_are_callable():
    tools = get_all_tools()
    for tool in tools:
        assert callable(tool) or hasattr(
            tool, "invoke"
        ), f"Tool {tool!r} is not callable and has no .invoke()"
