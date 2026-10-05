"""Educational implementation of the k-nearest-neighbors classifier."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def _euclidean_distance(first_instance: np.ndarray, second_instance: np.ndarray) -> float:
    """Compute Euclidean distance between two feature vectors.

    Args:
        first_instance: First feature vector.
        second_instance: Second feature vector.

    Returns:
        Euclidean distance between the two vectors.
    """
    distance = np.sqrt(np.sum((first_instance - second_instance) ** 2))
    return float(distance)


class KNN_classifier:
    """Classify examples using the majority label among nearby training points."""

    def __init__(self, k: int) -> None:
        """Initialize a KNN classifier.

        Args:
            k: Number of nearest neighbors to use for each prediction.
        """
        self.k = k
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> KNN_classifier:
        """Store training examples for neighbor lookup.

        Args:
            X_train: Training feature matrix with one row per instance.
            y_train: Training labels aligned with ``X_train``.

        Returns:
            This fitted classifier.
        """
        self.X_train = X_train
        self.y_train = y_train
        return self

    def _predict_single_instance(self, instance: np.ndarray) -> Any:
        """Predict one instance using the majority label of its neighbors.

        Args:
            instance: Feature vector to classify.

        Returns:
            Most common label among the ``k`` nearest training instances.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("KNN classifier must be fitted before prediction")

        distances = [
            _euclidean_distance(training_instance, instance)
            for training_instance in self.X_train
        ]
        nearest_indices = np.argsort(distances)[: self.k]
        nearest_labels = self.y_train[nearest_indices]
        return Counter(nearest_labels).most_common(1)[0][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for a matrix of feature vectors.

        Args:
            X: Feature matrix to classify.

        Returns:
            Array of predicted labels, one per row of ``X``.
        """
        predictions = np.array([self._predict_single_instance(instance) for instance in X])
        return predictions
