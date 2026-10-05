# Continuous Integration (CI) — 1-Page Guide

Continuous integration runs the same checks on a clean machine whenever code is pushed. It catches missing files, undeclared dependencies, formatting problems, and regressions that may not appear in one local environment.

## Recommended checks

```text
install package and development dependencies
        ↓
Ruff lint and format check
        ↓
mypy type check
        ↓
pytest
```

## GitHub Actions example

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: python -m pip install -e ".[dev]" ruff mypy
      - name: Ruff lint
        run: ruff check .
      - name: Ruff format check
        run: ruff format --check .
      - name: Type check
        run: mypy src/
      - name: Tests
        run: pytest -q
```

After pushing, open the repository’s **Actions** tab and inspect the workflow run. A green workflow means the checked commit passed the project’s agreed automated checks; it does not replace reviewing the algorithm’s correctness.

## Local versus CI

Local checks provide fast feedback while developing. CI provides shared evidence that the same repository works on a clean runner. Keep CI commands reproducible and aligned with the commands documented in this repository.
