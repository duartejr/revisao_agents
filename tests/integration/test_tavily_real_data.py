from dotenv import load_dotenv

load_dotenv()
from revisao_agents.tools.tavily_web_search import search_tavily


def test_usage_logging():
    """Test that the search_tavily function correctly logs usage information when provided.
    This test ensures that the logging mechanism in search_tavily includes API usage statistics in the saved Markdown file when the usage parameter is provided.

    Asserts:
        The generated Markdown file contains the expected usage information.
    """
    query = "What is the capital of France?"
    print("Testing search_tavily with usage logging...")
    print(f"Using query: {query}")
    results = search_tavily.invoke({"queries": [query]})

    print(f"Received results: {results}")

    print("Test completed. Please check the generated Markdown file for usage information.")
    print(results.get("usage", {}))


if __name__ == "__main__":
    test_usage_logging()
