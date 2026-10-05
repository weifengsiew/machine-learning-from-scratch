"""Decision tree classifier for categorical or numeric attributes."""

import numpy as np


def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def gini(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return float(1 - np.sum(probabilities**2))


def _score(X, y, column, criterion):
    impurity = entropy if criterion == "information_gain" else gini
    parent = impurity(y)
    child = sum(
        np.sum(mask) / len(y) * impurity(y[mask])
        for value in np.unique(X[:, column])
        for mask in [X[:, column] == value]
    )
    return parent - child if criterion == "information_gain" else parent - child


def _thresholds(values):
    unique = np.sort(np.unique(values.astype(float)))
    return (unique[:-1] + unique[1:]) / 2


class _Leaf:
    def __init__(self, label):
        self.label = label

    def predict(self, _):
        return self.label


class _Node:
    def __init__(self, column, threshold, majority):
        self.column, self.threshold, self.majority = column, threshold, majority
        self.children = {}

    def predict(self, row):
        value = float(row[self.column]) if self.threshold is not None else row[self.column]
        key = value <= self.threshold if self.threshold is not None else value
        child = self.children.get(key)
        return self.majority if child is None else child.predict(row)


class DecisionTreeClassifier:
    """Greedy decision tree supporting categorical and numeric columns."""

    def __init__(self, criterion="gini", max_majority=1.0, min_samples_split=2):
        if criterion not in {"gini", "information_gain"}:
            raise ValueError("criterion must be 'gini' or 'information_gain'")
        self.criterion = criterion
        self.max_majority = max_majority
        self.min_samples_split = min_samples_split
        self.root = None
        self.attribute_types = None

    def fit(self, X, y, attribute_types=None):
        X, y = np.asarray(X, dtype=object), np.asarray(y)
        self.attribute_types = attribute_types or ["categorical"] * X.shape[1]
        self.root = self._grow(X, y, list(range(X.shape[1])))
        return self

    def _grow(self, X, y, available):
        labels, counts = np.unique(y, return_counts=True)
        majority = labels[np.argmax(counts)]
        if (counts.max() / len(y) >= self.max_majority or
                len(y) < self.min_samples_split or not available or len(labels) == 1):
            return _Leaf(majority)

        best = None
        best_gain = -np.inf
        for column in available:
            candidates = [None]
            if self.attribute_types[column] == "numeric":
                candidates = _thresholds(X[:, column])
            for threshold in candidates:
                values = (X[:, column].astype(float) <= threshold) if threshold is not None else X[:, column]
                gain = _score(values.reshape(-1, 1), y, 0, self.criterion)
                if gain > best_gain:
                    best, best_gain = (column, threshold), gain
        if best is None:
            return _Leaf(majority)

        column, threshold = best
        node = _Node(column, threshold, majority)
        values = (X[:, column].astype(float) <= threshold) if threshold is not None else X[:, column]
        for value in np.unique(values):
            mask = values == value
            node.children[value] = self._grow(X[mask], y[mask], [c for c in available if c != column])
        return node

    def predict(self, X):
        if self.root is None:
            raise ValueError("fit must be called before predict")
        return np.asarray([self.root.predict(row) for row in np.asarray(X, dtype=object)])
