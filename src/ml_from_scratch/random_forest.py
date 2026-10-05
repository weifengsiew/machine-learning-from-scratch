"""Random forest classifier built from the decision tree implementation."""

from collections import Counter

import numpy as np

from .decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(self, n_trees=25, criterion="gini", min_samples_split=2, random_state=None):
        self.n_trees = n_trees
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y, attribute_types=None):
        X, y = np.asarray(X, dtype=object), np.asarray(y)
        generator = np.random.default_rng(self.random_state)
        self.trees = []
        for _ in range(self.n_trees):
            indices = generator.integers(0, len(y), size=len(y))
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                max_majority=1.0,
            )
            tree.fit(X[indices], y[indices], attribute_types=attribute_types)
            self.trees.append(tree)
        return self

    def predict(self, X):
        if not self.trees:
            raise ValueError("fit must be called before predict")
        predictions = np.asarray([tree.predict(X) for tree in self.trees]).T
        return np.asarray([Counter(row).most_common(1)[0][0] for row in predictions])
