# Good Coding Practices

This project is intentionally educational: each algorithm should make the underlying mathematics and data flow easy to follow.

## Functions versus classes

Use a function for a focused calculation that does not own changing state:

```python
def entropy(labels: np.ndarray) -> float:
    """Compute the entropy of a set of class labels."""
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return float(-np.sum(probabilities * np.log2(probabilities)))
```

Use a class when an algorithm needs fitted state that is reused during prediction:

```python
model = KNNClassifier(k=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Keep the public workflow explicit: `fit()` learns parameters and `predict()` uses them. Avoid hidden global state and avoid silently fitting inside `predict()`.

## Naming and interfaces

- Use `PascalCase` for classes and `snake_case` for functions and variables.
- Prefer descriptive names such as `regularization_strength` over abbreviations.
- Validate user-facing parameters early and raise `ValueError` with a useful message.
- Return `self` from `fit()` so models can be chained consistently.
- Keep algorithm modules independent; shared numerical helpers belong in small utility modules.

## Numerical code

- Convert inputs with `np.asarray()` at the boundary of a public method.
- Keep training and prediction transformations consistent.
- Protect logarithms and divisions from invalid values where appropriate.
- Make randomness reproducible through an explicit `random_state` or generator.
- Do not replace an algorithm with a scikit-learn model; the implementations here are meant to expose the mechanics.

## Documentation

Public classes and functions should have a short docstring describing their purpose, important arguments, and return value. Explain non-obvious mathematical choices close to the code that implements them.
