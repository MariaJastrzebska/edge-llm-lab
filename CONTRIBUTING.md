# Contributing to Edge LLM Lab

Thank you for your interest in contributing to Edge LLM Lab! This document provides guidelines for development, testing, and contributions.

## 📋 Table of Contents

- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Quality](#code-quality)
- [CI/CD Pipeline](#cicd-pipeline)
- [Contributing Workflow](#contributing-workflow)
- [MLOps Practices](#mlops-practices)

## 🛠️ Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/MariaJastrzebska/edge-llm-lab.git
cd edge-llm-lab
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

## 🧪 Running Tests

### Run All Tests

```bash
pytest
```

### Run Unit Tests Only

```bash
pytest tests/unit -v
```

### Run Integration Tests Only

```bash
pytest tests/integration -v
```

### Run with Coverage

```bash
pytest --cov=src/edge_llm_lab --cov-report=html --cov-report=term
```

View coverage report:
```bash
open htmlcov/index.html  # On macOS
```

### Run Specific Test File

```bash
pytest tests/unit/test_optimization.py -v
```

### Run Tests in Parallel

```bash
pytest -n auto
```

## ✨ Code Quality

### Format Code with Black

```bash
black src/ tests/
```

### Lint with Ruff

```bash
ruff check src/ tests/
```

### Fix Lint Issues Automatically

```bash
ruff check --fix src/ tests/
```

### Sort Imports with isort

```bash
isort src/ tests/
```

### Type Check with MyPy

```bash
mypy src/
```

### Security Scan with Bandit

```bash
bandit -r src/
```

### Run All Quality Checks

```bash
# Format
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Security
bandit -r src/
```

## 🔄 CI/CD Pipeline

The project uses GitHub Actions for CI/CD. The pipeline runs automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

### Pipeline Jobs

1. **Test** - Runs on Python 3.10, 3.11, 3.12 on Ubuntu and macOS
   - Unit tests with pytest
   - Coverage reporting to Codecov

2. **Lint** - Code quality checks
   - Ruff linting
   - Black formatting check
   - isort import sorting check

3. **Security** - Security scanning
   - Bandit security analysis
   - Uploads security report

4. **Build** - Package building
   - Builds distribution packages
   - Validates package integrity
   - Uploads build artifacts

5. **Integration Test** - Integration tests
   - Runs integration test suite

### View CI/CD Results

Check the "Actions" tab in GitHub to see pipeline results.

## 🔀 Contributing Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, well-documented code
- Add tests for new functionality
- Follow existing code style

### 3. Run Tests Locally

```bash
pytest
black src/ tests/
ruff check src/ tests/
```

### 4. Commit Changes

Pre-commit hooks will run automatically:

```bash
git add .
git commit -m "feat: add new feature"
```

### Commit Message Convention

Follow conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test changes
- `refactor:` - Code refactoring
- `chore:` - Build/tooling changes

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## 🚀 MLOps Practices

### Version Control

- Use semantic versioning (MAJOR.MINOR.PATCH)
- Tag releases: `git tag -a v0.1.0 -m "Release v0.1.0"`
- Keep main branch stable

### Testing Strategy

1. **Unit Tests** - Test individual components
   - Fast, isolated tests
   - Mock external dependencies
   - Target: >80% coverage

2. **Integration Tests** - Test component interactions
   - Test configuration loading
   - Test end-to-end workflows
   - Use real (but minimal) data

3. **Performance Tests** - Track performance metrics
   - Monitor optimization effectiveness
   - Track latency and throughput

### Code Quality Metrics

- **Coverage**: Maintain >80% code coverage
- **Linting**: Zero lint errors in production code
- **Security**: Regular Bandit scans
- **Type Hints**: Gradually add type hints

### Experiment Tracking

When adding new optimizations:

1. Document in `src/edge_llm_lab/utils/optimization.py`
2. Add test in `tests/unit/test_optimization.py`
3. Update README with new optimization options
4. Log results in evaluation logs for comparison

### Continuous Improvement

- Review test coverage regularly
- Refactor code for maintainability
- Update dependencies quarterly
- Monitor GitHub Actions for failures

## 📝 Documentation

- Update README.md for user-facing changes
- Add docstrings to all public functions
- Include examples for new features
- Keep CONTRIBUTING.md up to date

## 🤝 Getting Help

- Open an issue for bugs or feature requests
- Join discussions in GitHub Discussions
- Check existing issues before creating new ones

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

