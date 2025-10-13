.PHONY: help install test lint format clean dev-install

help:
	@echo "Edge LLM Lab - Development Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install package"
	@echo "  make dev-install  - Install package with dev dependencies"
	@echo "  make test         - Run all tests"
	@echo "  make test-unit    - Run unit tests only"
	@echo "  make test-int     - Run integration tests only"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make lint         - Run linting checks"
	@echo "  make format       - Format code with black and isort"
	@echo "  make type-check   - Run mypy type checking"
	@echo "  make security     - Run security scan with bandit"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make build        - Build package"
	@echo "  make all          - Run format, lint, and test"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit -v

test-int:
	pytest tests/integration -v

test-cov:
	pytest tests/ --cov=src/edge_llm_lab --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	@echo "Running Ruff..."
	ruff check src/ tests/
	@echo "Checking Black formatting..."
	black --check src/ tests/
	@echo "Checking isort..."
	isort --check-only src/ tests/

format:
	@echo "Formatting with Black..."
	black src/ tests/
	@echo "Sorting imports with isort..."
	isort src/ tests/
	@echo "Auto-fixing with Ruff..."
	ruff check --fix src/ tests/

type-check:
	mypy src/

security:
	bandit -r src/ -f screen

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build dist htmlcov .coverage coverage.xml

build: clean
	python -m build

all: format lint test
	@echo "✅ All checks passed!"

