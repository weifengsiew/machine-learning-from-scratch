# mypy Type Checker — Quick Guide

`mypy` checks Python type hints without running the program. Type hints make algorithm interfaces clearer, especially for arrays, labels, tokenized documents, and fitted model state.

Install and run it with:

```bash
python -m pip install mypy
mypy src/
```

Useful commands:

```bash
mypy src/ml_from_scratch/
mypy --show-error-codes src/
mypy --strict src/
```

Fix the source when possible. If an external library lacks usable type information, keep any suppression narrow and explain why it is needed:

```python
value = third_party_call()  # type: ignore[no-untyped-call]
```

Do not use broad `# type: ignore` comments to hide errors in algorithm logic. Run mypy again after each focused fix.
