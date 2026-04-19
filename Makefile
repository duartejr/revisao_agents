# Revisão Agents – Makefile
# Usage: make <target>

.PHONY: help install install-dev lint format typecheck test test-cov all clean

PYTHON := python
SRC    := src/revisao_agents

help:           ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:        ## Install runtime dependencies
	uv sync

install-dev:    ## Install all dependencies including dev tools
	uv sync --extra dev
	uv run --extra dev pre-commit install

lint:           ## Run ruff linter
	uv run --extra dev ruff check $(SRC)

format:         ## Auto-fix style issues with ruff
	uv run --extra dev ruff check --fix $(SRC)
	uv run --extra dev ruff format $(SRC)

typecheck:      ## Run mypy type checker
	uv run --extra dev mypy $(SRC)

test:           ## Run all tests
	uv run --extra dev pytest tests/

test-cov:       ## Run tests with coverage report
	uv run --extra dev pytest tests/ --cov=$(SRC) --cov-report=term-missing --cov-report=html

all:            ## Run lint, typecheck, and tests
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test

clean:          ## Remove build artifacts and caches
	rm -rf .pytest_cache htmlcov .ruff_cache .mypy_cache dist build
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

MLFLOW_BACKEND_STORE_URI ?= sqlite:///./mlruns/mlflow.db
MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5000

mlflow-start:   ## Start MLflow UI with local SQLite backend (http://localhost:5000)
	uv run mlflow ui \
	  --backend-store-uri $(MLFLOW_BACKEND_STORE_URI) \
	  --host $(MLFLOW_HOST) \
      --port $(MLFLOW_PORT)
