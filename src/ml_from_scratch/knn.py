"""K-nearest-neighbors classifier."""

from collections import Counter

import numpy as np


class KNNClassifier:
    """Classify examples by majority vote among their nearest neighbors."""

    def __init__(self, k: int = 3):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of examples")
        if self.k > len(X):
            raise ValueError("k cannot exceed the number of training examples")
        self.X_train, self.y_train = X, y
        return self

    def predict(self, X):
        if self.X_train is None:
            raise ValueError("fit must be called before predict")
        X = np.asarray(X)
        predictions = []
        for example in X:
            distances = np.sqrt(np.sum((self.X_train - example) ** 2, axis=1))
            neighbors = self.y_train[np.argsort(distances)[: self.k]]
            predictions.append(Counter(neighbors).most_common(1)[0][0])
        return np.asarray(predictions)
