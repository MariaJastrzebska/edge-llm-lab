# Pre-Push Checklist

Before pushing to GitHub, ensure:

## ✅ Local Checks

```bash
# 1. Install in editable mode
pip install -e ".[dev]"

# 2. Run tests
pytest tests/

# 3. Check formatting
black --check src/ tests/

# 4. Check linting
ruff check src/ tests/

# 5. Check types (optional)
mypy src/

# 6. Run security scan (optional)
bandit -r src/
```

## 📦 Files to Commit

Check that these files exist:
- [ ] `.github/workflows/ci.yml`
- [ ] `tests/unit/test_optimization.py`
- [ ] `tests/unit/test_agent_config.py`
- [ ] `tests/integration/test_config_loading.py`
- [ ] `pyproject.toml` (with dev dependencies)
- [ ] `.gitignore`
- [ ] `.pre-commit-config.yaml`
- [ ] `README.md` (updated)
- [ ] `TESTING.md`
- [ ] `CONTRIBUTING.md`

## 🚀 Git Workflow

```bash
# 1. Check status
git status

# 2. Add files
git add .

# 3. Commit with conventional message
git commit -m "feat: add MLOps setup with CI/CD, tests, and CLI"

# 4. Push to GitHub
git push origin main  # or your branch name

# 5. Check CI/CD status
# Go to: https://github.com/YOUR_USERNAME/edge-llm-lab/actions
```

## 🔄 What Happens After Push

GitHub Actions will automatically:

1. **Test Job** (runs on Ubuntu & macOS, Python 3.10/3.11/3.12):
   - Install dependencies
   - Run pytest with coverage
   - Upload coverage to Codecov (if configured)

2. **Lint Job** (Ubuntu, Python 3.11):
   - Run Ruff linting
   - Check Black formatting
   - Check isort imports

3. **Security Job** (Ubuntu, Python 3.11):
   - Run Bandit security scan
   - Upload security report

4. **Build Job** (Ubuntu, Python 3.11):
   - Build Python package
   - Validate with twine
   - Upload package artifacts

5. **Integration Test Job** (Ubuntu, Python 3.11):
   - Run integration tests

## 📊 Viewing CI/CD Results

After push:
1. Go to your repository on GitHub
2. Click the "Actions" tab
3. See the workflow run in progress
4. Click on a job to see detailed logs

## 🐛 If CI Fails

1. Check the error in GitHub Actions logs
2. Fix locally
3. Run tests again: `pytest`
4. Commit and push the fix

## 💡 Tips

- All checks must pass before merge to main
- You can see which checks failed in the PR
- Green checkmark = all tests passed ✅
- Red X = some tests failed ❌

