# Copilot Workspace Instructions

When handling any request that edits, adds, removes, or reviews functionality related to Tavily (API usage, wrappers, tools, integrations, or config):

1. Always consult the official Tavily documentation first: https://docs.tavily.com/
2. Validate whether the requested behavior is actually supported before implementing changes.
3. If the request is not supported or has constraints, explicitly state the limitation and propose practical alternatives.
4. Prefer implementation choices that match documented Tavily behavior instead of assumptions.
5. In your response, briefly mention which Tavily doc page/feature informed the decision.

If network/tool access prevents reading the docs in that moment, say so clearly and avoid claiming unsupported Tavily behavior as possible.

When handling any request that edits, adds, removes, or reviews functionality related to LangChain, LangGraph, deep agents, LangSmith:

1. Always consult the official LangChain documentation first: https://docs.langchain.com/
2. Validate whether the requested behavior is actually supported before implementing changes.
3. If the request is not supported or has constraints, explicitly state the limitation and propose practical alternatives.
4. Prefer implementation choices that match documented LangChain behavior instead of assumptions.
5. In your response, briefly mention which LangChain doc page/feature informed the decision.

If network/tool access prevents reading the docs in that moment, say so clearly and avoid claiming unsupported LangChain behavior as possible.

When handling any request that edits, adds, removes, or reviews functionality related to MLflow (tracking, models, artifacts, model registry, deployments, or integrations):

1. Always consult the official MLflow documentation first: https://mlflow.org/docs
2. Validate whether the requested behavior is actually supported before implementing changes.
3. If the request is not supported or has constraints, explicitly state the limitation and propose practical alternatives.
4. Prefer implementation choices that match documented MLflow behavior instead of assumptions.
5. In your response, briefly mention which MLflow doc page/feature informed the decision.

If network/tool access prevents reading the docs in that moment, say so clearly and avoid claiming unsupported MLflow behavior as possible.
