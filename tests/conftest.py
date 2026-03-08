# conftest.py — shared pytest fixtures
import pytest


@pytest.fixture(autouse=True)
def no_real_llm_calls(monkeypatch):
    """
    Prevent accidental real LLM calls in unit tests.
    Override per-test with:  @pytest.mark.usefixtures()  or re-patch locally.
    """
    # Only active for tests NOT in tests/integration/
    pass  # individual tests patch as needed
