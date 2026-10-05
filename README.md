# Machine Learning From Scratch

Implementations of core machine-learning algorithms written from scratch with NumPy. This repository collects the algorithms developed across HW1–HW4 into a reusable, readable Python package.

## Included algorithms

- K-nearest neighbors (KNN)
- Decision trees with entropy/information gain or Gini impurity
- Decision trees with categorical and numeric attributes
- Random forests with bootstrap sampling and random feature selection
- Multinomial Naive Bayes with additive smoothing and log probabilities
- Feed-forward neural networks with sigmoid activations, backpropagation, mini-batches, and L2 regularization

The implementations do not call scikit-learn model classes. scikit-learn is used only in the original coursework for data splitting and preprocessing experiments.

## Project layout

```text
src/ml_from_scratch/     reusable algorithm implementations
tests/                   focused behavior tests
examples/                small runnable examples
pyproject.toml           package metadata and dependencies
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

## Quick example

```python
import numpy as np
from ml_from_scratch import KNNClassifier

model = KNNClassifier(k=3).fit(
    np.array([[0.0], [1.0], [10.0]]),
    np.array([0, 0, 1]),
)
print(model.predict(np.array([[0.5], [9.0]])))
```

## Origin

The code was developed as part of machine-learning coursework covering supervised learning, text classification, ensemble methods, and neural networks.
