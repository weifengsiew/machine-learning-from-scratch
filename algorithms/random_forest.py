"""Educational implementation of a random forest classifier."""

from __future__ import annotations

from typing import Any

import numpy as np

from .decision_tree import decision_tree_classifier


class random_forest_classifier:
    """Combine randomized decision trees through majority voting."""

    def __init__(
        self,
        ntree: int,
        split_criterion: str,
        majority_class_threshold: float,
        minimum_size_for_split: int,
        minimum_split_quality_score: float,
        maximum_depth: int | None,
        random_seed: int,
    ) -> None:
        """Initialize a random forest classifier.

        Args:
            ntree: Number of decision trees to train.
            split_criterion: Tree split criterion, either ``information_gain`` or ``gini``.
            majority_class_threshold: Purity threshold for a majority-class leaf.
            minimum_size_for_split: Minimum number of instances required to split.
            minimum_split_quality_score: Minimum quality required for a split.
            maximum_depth: Maximum depth of each tree, or ``None`` for no limit.
            random_seed: Base seed used to make bootstrap samples reproducible.
        """
        self.ntree = ntree
        self.split_criterion = split_criterion
        self.majority_class_threshold = majority_class_threshold
        self.minimum_size_for_split = minimum_size_for_split
        self.minimum_split_quality_score = minimum_split_quality_score
        self.maximum_depth = maximum_depth
        self.random_seed = random_seed
        self.random_forest: list[decision_tree_classifier] = []

    def _get_bootstrap_dataset(
        self, X: np.ndarray, y: np.ndarray, random_seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a same-sized bootstrap sample with replacement.

        Args:
            X: Feature matrix to sample from.
            y: Labels aligned with ``X``.
            random_seed: Seed for the bootstrap sample.

        Returns:
            Tuple containing sampled features and sampled labels.
        """
        random_generator = np.random.default_rng(seed=random_seed)
        bootstrap_indices = random_generator.choice(
            np.arange(len(y)), size=len(y), replace=True
        )
        return X[bootstrap_indices], y[bootstrap_indices]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attribute_names: list[str],
        attribute_types: list[str],
    ) -> "random_forest_classifier":
        """Fit randomized decision trees on bootstrap samples.

        Args:
            X: Training feature matrix.
            y: Training labels aligned with ``X``.
            attribute_names: Names of the feature columns.
            attribute_types: Feature types, either ``numeric`` or ``categorical``.

        Returns:
            This fitted classifier.
        """
        self.random_forest = []
        for bootstrap in range(self.ntree):
            X_bootstrap, y_bootstrap = self._get_bootstrap_dataset(
                X=X, y=y, random_seed=self.random_seed + bootstrap
            )
            tree = decision_tree_classifier(
                split_criterion=self.split_criterion,
                majority_class_threshold=self.majority_class_threshold,
                minimum_size_for_split=self.minimum_size_for_split,
                minimum_split_quality_score=self.minimum_split_quality_score,
                maximum_depth=self.maximum_depth,
                random_attribute_selection=True,
                random_seed=self.random_seed + bootstrap,
            )
            tree.fit(
                X=X_bootstrap,
                y=y_bootstrap,
                attribute_names=attribute_names,
                attribute_types=attribute_types,
            )
            self.random_forest.append(tree)
        return self

    def _predict_with_trees(self, X: np.ndarray) -> np.ndarray:
        """Collect predictions from every tree in the forest.

        Args:
            X: Feature matrix to classify.

        Returns:
            Matrix whose rows contain predictions from individual trees.
        """
        return np.array([tree.predict(X=X) for tree in self.random_forest])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels by majority vote across the fitted trees.

        Args:
            X: Feature matrix to classify.

        Returns:
            Array of majority-vote predictions, one per row of ``X``.
        """
        trees_predictions = self._predict_with_trees(X=X)
        majority_predictions = []
        for predictions_for_instance in trees_predictions.T:
            predictions, counts = np.unique(
                predictions_for_instance, return_counts=True
            )
            majority_predictions.append(predictions[np.argmax(counts)])
        return np.array(majority_predictions)
