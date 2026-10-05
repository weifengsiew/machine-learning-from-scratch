# Implement–Test–Iterate

Use this loop for new algorithms, bug fixes, and improvements:

1. Define the goal: what behavior should the model support, and why?
2. Write acceptance examples using concrete arrays, documents, labels, and expected outputs.
3. Add or update focused pytest tests before changing the implementation.
4. Implement the smallest clear change in `src/ml_from_scratch/`.
5. Run the full local checks: pytest, Ruff, and mypy when configured.
6. Review the diff for accidental changes, then commit only after checks pass.
7. Push and confirm the GitHub Actions result.

For example, a new numeric decision-tree feature should specify the attribute type, candidate thresholds, expected split behavior, and how unseen values are handled before implementation begins.

When documenting a change, record:

- where the implementation lives;
- which tests cover it;
- the commands used for validation;
- any known limitation or unsupported input.

Keep the original homework folders unchanged. The extracted package is the public API surface for this repository.
