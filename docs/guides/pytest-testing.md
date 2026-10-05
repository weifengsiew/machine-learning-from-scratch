# Testing in Python with `pytest`

Tests should check the algorithms at several levels:

- **Unit tests** check one calculation, such as entropy or a bag-of-words vector.
- **Integration tests** check a model workflow such as `fit()` followed by `predict()`.
- **End-to-end tests** check a complete example from prepared input to final predictions.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the suite with:

```bash
pytest -v
```

## Example model test

```python
def test_knn_predicts_nearest_class():
    X_train = np.array([[0.0], [1.0], [10.0]])
    y_train = np.array([0, 0, 1])

    model = KNNClassifier(k=1).fit(X_train, y_train)

    assert model.predict(np.array([[0.2], [9.0]])).tolist() == [0, 1]
```

Good algorithm tests should include:

- a small hand-checkable example;
- an edge case, such as an unfitted model or invalid parameter;
- a shape and label-type check;
- a reproducibility check when randomness is involved.

Prefer assertions about observable behavior over implementation details such as private node layout.

## Current test command

The repository’s focused tests are in `tests/test_algorithms.py`. Run them from `machine-learning-from-scratch/` so the package is installed or available on the Python path.
