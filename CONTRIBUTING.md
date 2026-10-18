# Contributing to Edge-LLM-Lab

Contributions are welcome: bug fixes, documentation, reproducible evaluation cases,
hardware measurements and focused code changes. @MariaJastrzebska maintains the
project and reviews contributions.

## Start with a focused change

Check existing issues and PRs. Discuss substantial API changes, new dependencies
or expensive experiments in an issue first. Small fixes can go straight to a PR.
A draft PR is welcome for early feedback; no special issue label is required.

## Development setup

Use Python 3.10–3.13; CI currently tests 3.10, 3.11 and 3.12.
Fork the public repository, then:

```bash
git clone https://github.com/YOUR-USERNAME/edge-llm-lab.git
cd edge-llm-lab
git remote add upstream https://github.com/MariaJastrzebska/edge-llm-lab.git
git switch -c fix/short-description
python -m pip install poetry
poetry install --with dev
poetry run python -m pytest tests/unit tests/test_posthoc.py
```

Invited collaborators can clone the original repository and push a feature branch
there. Private repositories require an invitation; private forks depend on policy.
Never push a contribution directly to `main`.

Dependencies include native llama-cpp-python compilation, which may need a C/C++
toolchain. Unit tests need no model weights or API credentials. Real inference
experiments need a GGUF model and compatible llama-server; see the
[README](README.md#run-an-evaluation).

For optional Inspect adapter changes, also run:

```bash
poetry run python -m pip install -r requirements-inspect.txt
poetry run python -m pytest tests/unit/test_inspect_adapter.py -v
```

## Pull request and review

1. Keep the change focused. Add regression tests for changed behavior and update
   relevant docs/examples. Documentation-only changes need no artificial tests.
2. Run relevant tests and the unit/posthoc command above. State actual commands,
   results and anything untested in the PR. Match surrounding style; avoid
   unrelated repository-wide formatting.
3. Commit, push your branch and open a PR to `main`, linking an issue if applicable.
4. Address review comments and resolve discussions after handling the concern.
   New changes may require another approval. The maintainer merges accepted PRs,
   preferably with squash merge; the feature branch can then be deleted.

CI installs CPU-only PyTorch on Linux to avoid downloading unused CUDA packages.
CI runs the unit matrix, Inspect replay tests and package build. `CI required`
reports whether all three passed. Lint/format and Bandit checks are currently
advisory: a green workflow does not certify those findings are resolved.

PR workflows use a read-only GitHub token on GitHub-hosted runners and receive no
project secrets. GitHub may require approval before a fork workflow runs.
Maintainers must inspect workflow changes before approving execution. Do not use
pull_request_target to execute contributed code or run untrusted PRs on a
self-hosted device runner.

## Research contributions

Describe data provenance and redistribution permission. Use synthetic or
appropriately shareable data; never commit patient records, private conversations,
personal data or credentials. Keep model weights and large generated artifacts
out of PRs.

Record dataset/version, split and case families, model and quantization, prompt,
seed, device, runtime configuration and reproduction commands. Separate development
from held-out test cases. Preserve failures and missing measurements. Distinguish
fresh runs from re-analysis; explain baselines and limitations. Report task quality
separately from latency and energy. See the
[evaluation boundaries](docs/methodology/evaluation-boundaries.md).

## Ownership and protection

[CODEOWNERS](.github/CODEOWNERS) names the maintainer. Owner review requests depend
on GitHub features/settings. The intended policy is PRs, one approval, resolved
conversations and passing required CI, with no force-push or deletion of the
default branch. Files alone do not activate protection: administrators must enable
and verify the rules in GitHub Settings. Exceptional maintainer bypasses should
still use a PR.

Keep discussions constructive. Do not post credentials or exploit details in public
issues. Use GitHub private vulnerability reporting if enabled; otherwise arrange a
private channel with the maintainer before sharing sensitive details.

No repository-level LICENSE file is currently present. Do not assume an MIT
license or add one on the owner's behalf. Clarify licensing with the maintainer
before contributing third-party material or redistributing the project.
