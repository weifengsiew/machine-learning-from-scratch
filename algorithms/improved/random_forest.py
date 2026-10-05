"""Functions for the random forest algorithm"""

import numpy as np

from .decision_tree import decision_tree_classifier


class random_forest_classifier:
    def __init__(
            self, ntree, split_criterion, majority_class_threshold,
            minimum_size_for_split, minimum_split_quality_score,
            maximum_depth, random_seed):
        
        self.ntree = ntree
        self.split_criterion = split_criterion
        self.majority_class_threshold = majority_class_threshold
        self.minimum_size_for_split = minimum_size_for_split
        self.minimum_split_quality_score = minimum_split_quality_score
        self.maximum_depth = maximum_depth
        self.random_seed = random_seed

    def get_bootstrap_dataset(self, X, y, random_seed):
        """Create a bootstrap dataset of the same size as the original dataset,
        by randomly sampling with replacement from the original dataset.

        Args:
            X (np.ndarray): Attributes.
            y (np.ndarray): Class labels.
            random_seed (int): Random seed for reproducibility.

        Returns:
            X_bootstrap (np.ndarray): Attributes of bootstrap dataset.
            y_bootstrap (np.ndarray): Class labels of bootstrap dataset.
        """
        random_generator = np.random.default_rng(seed=random_seed)

        bootstrap_size = len(y)

        bootstrap_indices = random_generator.choice(
            a=np.arange(stop=bootstrap_size), size=bootstrap_size, replace=True)

        return X[bootstrap_indices], y[bootstrap_indices]

    def fit(self, X, y, attribute_names, attribute_types):
        """Fit the random forest classifier.

        Args:
            X (np.ndarray): Attributes.
            y (np.ndarray): Class labels.
            attribute_names (list): Attribute names.
            attribute_types (list): Attribute types, either "numeric" or "categorical".

        Returns:
            self (random_forest_classifier): Fitted random forest classifier.
        """
        self.random_forest = []

        for bootstrap in range(self.ntree):

            X_bootstrap, y_bootstrap = self.get_bootstrap_dataset(X=X, y=y, random_seed=self.random_seed + bootstrap)

            decision_tree = decision_tree_classifier(
                split_criterion=self.split_criterion,
                majority_class_threshold=self.majority_class_threshold,
                minimum_size_for_split=self.minimum_size_for_split,
                minimum_split_quality_score=self.minimum_split_quality_score,
                maximum_depth=self.maximum_depth,
                random_attribute_selection=True, random_seed=self.random_seed + bootstrap)

            decision_tree.fit(X=X_bootstrap, y=y_bootstrap,
                              attribute_names=attribute_names,
                              attribute_types=attribute_types)

            self.random_forest.append(decision_tree)

        return self

    def predict_with_trees_in_forest(self, X):
        """Predict class labels of instances with each tree in the forest.

        Args:
            X (np.ndarray): Attributes.

        Returns:
            trees_predictions (np.ndarray): Predicted class labels of instances from each tree in the forest.
        """
        trees_predictions = np.array(object=[decision_tree.predict(X=X) for decision_tree in self.random_forest])

        return trees_predictions

    def predict(self, X):
        """Predict class labels of instances by majority vote among trees.

        Args:
            X (np.ndarray): Attributes.

        Returns:
            majority_predictions (np.ndarray): Predicted class labels of instances by majority vote among trees.
        """
        trees_predictions = self.predict_with_trees_in_forest(X=X)

        majority_predictions = []

        for trees_predictions_for_instance in trees_predictions.T:

            predictions, counts = np.unique(ar=trees_predictions_for_instance, return_counts=True)

            majority_prediction_for_instance = predictions[np.argmax(a=counts)]

            majority_predictions.append(majority_prediction_for_instance)

        majority_predictions = np.array(object=majority_predictions)

        return majority_predictions
