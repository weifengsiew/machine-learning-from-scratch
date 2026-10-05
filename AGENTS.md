# Working Guidelines

- **Discuss code changes first:** propose and align on non-trivial implementation changes before editing. Documentation-only changes may be made directly when they are clearly requested.
- **Keep changes minimal:** implement the smallest change that satisfies the agreed behavior.
- **Work incrementally:** follow [Implement–Test–Iterate](docs/guides/implement-test-iterate.md) for algorithm changes and bug fixes.
- **Stay focused:** do not refactor unrelated code or add dependencies without a clear reason.
- **Keep one current implementation:** `algorithms/` contains the selected implementation for each algorithm. Do not add duplicate version folders.

## Definition of done

Before reporting a change as complete:

1. Run `pytest -q` from the repository root.
2. Run `git diff --check` to catch whitespace errors.
3. Run Ruff and mypy when they are installed or configured for the change.
4. Confirm every applicable command passes.
5. Report the exact commands and results.

If a check fails because of pre-existing code or an unavailable local tool, do not claim that the change is fully validated. Either fix the issue, narrow the check deliberately, or report the limitation.

## Repository structure

- `algorithms/` contains KNN, Decision Tree, Multinomial Naive Bayes, Random Forest, Neural Network, and supporting printer code.
- `docs/guides/` contains coding, testing, tooling, and CI guidance.
- `README.md` is the public project overview.

## Coding practices

See the detailed project guides:

- [Good coding practices](docs/guides/good-coding-practices.md) for functions, classes, naming, docstrings, and numerical code.
- [Testing with pytest](docs/guides/pytest-testing.md) for test structure and test levels.
- [Ruff linting and formatting](docs/guides/ruff-linter-formatter.md) for style checks.
- [Type checking with mypy](docs/guides/mypy-type-checker.md) for static analysis.
- [GitHub/GitLab continuous integration](docs/guides/github-gitlab-continuous-integration.md) for shared validation.

## Algorithm-specific expectations

- Preserve the selected algorithm interfaces and document meaningful changes from the homework versions.
- Keep one authoritative implementation per algorithm.
- Do not replace a from-scratch implementation with a scikit-learn estimator.
