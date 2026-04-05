from unittest.mock import MagicMock

import pytest

from gradio_app.handlers import start_planning


class MockChatOpenAI:
    """Mock class for ChatOpenAI to simulate LLM behavior without API calls."""

    def __init__(self, *args, **kwargs):
        """Mock initializer that ignores all parameters."""
        # Ignore all init params (model, api_key, etc.)
        pass

    def invoke(self, prompt, *args, **kwargs) -> MagicMock:
        """Mock invoke method that returns a fixed response.

        Args:
            prompt: The prompt input (ignored in this mock).
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A MagicMock object simulating the LLM response with a 'content' attribute.

        """
        return MagicMock(content="Mocked LLM response for planning")

    def bind_tools(self, tools) -> "MockChatOpenAI":
        """Mock bind_tools method that simply returns self to allow chaining.
        This is used to simulate the behavior of binding tools to the LLM without implementing actual tool functionality.

        Args:
            tools: The tools to bind (ignored in this mock).

        Returns:
            self: Allows method chaining by returning the instance of the mock class.
        """
        return self


class MockCompletions:
    """Mock completions API that simulates OpenAI's chat.completions interface."""

    def create(self, **kwargs) -> MagicMock:
        """Simulate the creation of a chat completion by returning a fixed response structure.
        This method mimics the expected response format from OpenAI's chat completion API, including choices

        Args:
            **kwargs: Keyword arguments (ignored)

        Returns:
            A MagicMock object simulating the response from the chat completion API, with a structure that
            includes choices and usage information.
        """
        # Create a mock response that matches OpenAI's structure
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Mocked response"))]

        mock_response.parse.return_value = mock_response

        # Force the model_dump to return a real dictionary without MagicMocks for the main keys
        mock_response.model_dump.return_value = {
            "choices": [
                {
                    "message": {"content": "Mocked response", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            "error": None,
        }
        return mock_response

    @property
    def with_raw_response(self) -> "MockCompletions":
        """Return self to allow chaining with_raw_response.create()
        This is used to simulate the behavior of chaining the with_raw_response method in the OpenAI client.

        Returns:
            self: Allows method chaining by returning the instance of the mock class."""
        return self


class MockChat:
    """Mock chat API that provides completions."""

    def __init__(self):
        self.completions = MockCompletions()


class MockEmbeddings:
    """Mock embeddings API."""

    def create(self, **kwargs) -> MagicMock:
        """Simulate the creation of an embedding by returning a fixed vector.
        This method mimics the expected response format from an embedding API, returning a fixed-size vector
        of floats to represent the embedding.

        Args:
            **kwargs: Keyword arguments (ignored)

        Returns:
            A MagicMock object simulating the response from the embedding API, with a structure that
            matches the expected format for embedding responses.
        """
        return MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])


def mock_embedding(*args, **kwargs) -> list[float]:
    """Mock embedding function that returns a fixed-size vector of zeros.
    This is used to simulate the behavior of an embedding function without relying on external services.

    Args:
        *args: Positional arguments (ignored)
        **kwargs: Keyword arguments (ignored)

    Returns:
        A list of floats representing the embedding vector.
    """
    return [0.1] * 1536


def mock_llm_call(*args, **kwargs) -> str:
    """Mock LLM call function that returns a fixed response.
    This is used to simulate the behavior of an LLM call without relying on external services.

    Args:
        *args: Positional arguments (ignored)
        **kwargs: Keyword arguments (ignored)

    Returns:
        A fixed string response simulating an LLM output.
    """
    return "Mock LLM response"


@pytest.mark.parametrize("provider", ["gemini", "azure", "aws", "custom"])
def test_start_planning_with_invalid_provider(monkeypatch, provider):
    """Test that start_planning returns an error message when LLM_PROVIDER is set to an invalid provider.
    This ensures that start_planning correctly validates the LLM provider configuration and returns an appropriate error message when the provider is unsupported.

    Args:
        monkeypatch: pytest fixture for modifying environment variables
        provider: the invalid LLM provider to test

    Asserts:
        The status message includes an error about the invalid provider, and session_state is not set.
    """
    monkeypatch.setenv("LLM_PROVIDER", provider)
    history, session_state, status_msg, rendered_plan = start_planning("Tema", "academico", 2)
    assert "Invalid provider" in status_msg or "LLM_PROVIDER error" in status_msg
    assert not session_state


def test_start_planning_with_missing_api_keys(monkeypatch):
    """Test that start_planning returns an error message when required API keys are missing.
    This ensures that start_planning correctly validates the presence of required API keys in the configuration and returns an appropriate error message when they are missing.

     Args:
        monkeypatch: pytest fixture for modifying environment variables

    Asserts:
        The status message includes an error about missing API keys, and session_state is not set.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    history, session_state, status_msg, rendered_plan = start_planning("Tema", "academico", 2)
    assert "Missing OPENAI_API_KEY" in status_msg or "LLM_PROVIDER error" in status_msg
    assert not session_state


@pytest.mark.parametrize("provider", ["openai"])
def test_start_planning_with_valid_provider(monkeypatch, provider):
    """Test that start_planning initializes correctly with a valid provider and API keys.
    This ensures that start_planning successfully initializes the planning process when the LLM provider is valid and all required API keys are present.

    Args:
        monkeypatch: pytest fixture for modifying environment variables
        provider: the valid LLM provider to test

    Asserts:
        The status message indicates that planning has started, session_state is set with expected keys, and rendered_plan is empty.
    """
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("OPENAI_API_KEY", "openai_key-value")
    monkeypatch.setenv("MONGODB_URI", "mongodb_uri-value")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily_key-value")
    monkeypatch.setattr(
        "revisao_agents.utils.vector_utils.vector_store._generate_embedding", mock_embedding
    )

    # Mock para OpenAI client com estrutura completa esperada por LangChain
    class MockOpenAI:
        def __init__(self, *args, **kwargs):
            # Ignore all init params (api_key, etc.)
            self.chat = MockChat()
            self.embeddings = MockEmbeddings()

    monkeypatch.setattr("openai.OpenAI", MockOpenAI)

    # Mock get_llm para retornar instância do MockChatOpenAI nos nós do workflow
    mock_llm_instance = MockChatOpenAI()
    monkeypatch.setattr(
        "revisao_agents.utils.llm_utils.llm_providers.get_llm",
        lambda *args, **kwargs: mock_llm_instance,
    )

    history, session_state, status_msg, rendered_plan = start_planning("Tema", "academico", 2)

    assert "in progress" in status_msg and session_state
    assert rendered_plan == ""
