# Revisão Agents – Makefile
# Usage: make <target>

.PHONY: help install install-dev lint format typecheck test test-cov clean

PYTHON := python
SRC    := src/revisao_agents

help:           ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:        ## Install runtime dependencies
	pip install -e .

install-dev:    ## Install all dependencies including dev tools
	pip install -e ".[dev]"
	pre-commit install

lint:           ## Run ruff linter
	ruff check $(SRC)

format:         ## Auto-fix style issues with ruff
	ruff check --fix $(SRC)
	ruff format $(SRC)

typecheck:      ## Run mypy type checker
	mypy $(SRC)

test:           ## Run all tests
	pytest tests/

test-cov:       ## Run tests with coverage report
	pytest tests/ --cov=$(SRC) --cov-report=term-missing --cov-report=html

clean:          ## Remove build artifacts and caches
	rm -rf .pytest_cache htmlcov .ruff_cache .mypy_cache dist build
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
