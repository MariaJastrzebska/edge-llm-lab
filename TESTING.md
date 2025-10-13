# Testing Guide

Instructions for testing edge-llm-lab following CI/CD methodology.

## Quick Start

```bash
# 1. Install dev dependencies
pip install -e ".[dev]"

# 2. Run tests
pytest

# 3. Check coverage
pytest --cov=src/edge_llm_lab --cov-report=term
```

## Local Testing (before push)

### 1. Unit Tests
```bash
pytest tests/unit -v
```

### 2. Integration Tests
```bash
pytest tests/integration -v
```

### 3. All Tests with Coverage
```bash
pytest --cov=src/edge_llm_lab --cov-report=html --cov-report=term
# View report: open htmlcov/index.html
```

### 4. Linting (Ruff)
```bash
ruff check src/ tests/
# Auto-fix: ruff check --fix src/ tests/
```

### 5. Formatting (Black)
```bash
black --check src/ tests/
# Auto-format: black src/ tests/
```

### 6. Import Sorting (isort)
```bash
isort --check-only src/ tests/
# Auto-sort: isort src/ tests/
```

### 7. Type Checking (MyPy)
```bash
mypy src/
```

### 8. Security Scan (Bandit)
```bash
bandit -r src/
```

## CI/CD Testing (GitHub Actions)

### Simulate CI Locally

You can use `act` to run GitHub Actions locally:

```bash
# Install act: brew install act (macOS)
act -j test
```

### CI/CD Workflow

The pipeline runs automatically on:
- Push to `main` or `develop`
- Pull Request to `main` or `develop`

Check status: https://github.com/MariaJastrzebska/edge-llm-lab/actions

## Pre-commit Hooks

Install pre-commit hooks (automatic checks before commit):

```bash
pre-commit install
```

Hooks will automatically run:
- Black (formatting)
- Ruff (linting)
- isort (import sorting)
- MyPy (type checking)

Manual run on all files:
```bash
pre-commit run --all-files
```

## Test Structure

```
tests/
├── unit/                       # Unit tests (fast, isolated)
│   ├── test_optimization.py   # Optimization system tests
│   └── test_agent_config.py   # Agent configuration tests
└── integration/                # Integration tests (end-to-end)
    └── test_config_loading.py # Configuration loading tests
```

## Writing New Tests

### Unit Test

```python
# tests/unit/test_mymodule.py
import pytest
from edge_llm_lab.mymodule import my_function

def test_my_function():
    result = my_function("input")
    assert result == "expected"
```

### Integration Test

```python
# tests/integration/test_workflow.py
import pytest
from edge_llm_lab.core.agent_config import AgentConfig

@pytest.mark.integration
def test_complete_workflow():
    config = AgentConfig.load_from_yaml("config.yaml")
    # Test complete workflow
    assert config is not None
```

## Debugging Tests

### Run Single Test
```bash
pytest tests/unit/test_optimization.py::TestOptimalKVCacheType::test_exact_match_f16 -v
```

### Stop at First Failure
```bash
pytest -x
```

### Show Print Statements
```bash
pytest -s
```

### Enter Debugger on Failure
```bash
pytest --pdb
```

## Coverage Targets

- **Goal**: >80% coverage for production code
- **Check**: `pytest --cov=src/edge_llm_lab --cov-report=term`
- **HTML Report**: `pytest --cov=src/edge_llm_lab --cov-report=html`

## Continuous Integration Status

Check CI/CD status:
1. Go to: https://github.com/MariaJastrzebska/edge-llm-lab
2. Click "Actions" tab
3. View recent workflow runs

CI/CD Badge for README:
```markdown
![CI](https://github.com/MariaJastrzebska/edge-llm-lab/workflows/CI%2FCD%20Pipeline/badge.svg)
```

## Troubleshooting

### Tests Fail Locally
1. Check you have latest dependencies: `pip install -e ".[dev]"`
2. Check Python version: `python --version` (requires >=3.10)

### Pre-commit Hooks Not Working
```bash
pre-commit uninstall
pre-commit install
pre-commit run --all-files
```

### Import Errors in Tests
```bash
# Make sure package is installed in editable mode
pip install -e .
```

## MLOps Best Practices

### Before Committing
1. Run tests locally: `pytest`
2. Check formatting: `black --check src/ tests/`
3. Check linting: `ruff check src/ tests/`
4. Review coverage: `pytest --cov=src/edge_llm_lab`

### Before Creating PR
1. All tests pass locally
2. No linting errors
3. Coverage maintained or improved
4. Documentation updated

### CI/CD Pipeline Stages
1. **Test** - Unit and integration tests on multiple Python versions
2. **Lint** - Code quality checks (Ruff, Black, isort)
3. **Security** - Bandit security scan
4. **Build** - Package building and validation
5. **Integration** - End-to-end integration tests

## Continuous Improvement

- Review test coverage weekly
- Add tests for bug fixes
- Refactor tests for maintainability
- Keep dependencies updated
- Monitor CI/CD performance
