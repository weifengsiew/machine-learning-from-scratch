"""Functions and tree nodes for the decision tree classifier."""

from __future__ import annotations

import numpy as np


def entropy(y: np.ndarray) -> float:
    """Compute entropy.

    Args:
        y (np.ndarray): Class labels.

    Returns:
        entropy (float): Entropy of class labels.
    """
    _classes, class_counts = np.unique(y, return_counts=True)
    class_probabilities = class_counts / len(y)

    return -np.sum(class_probabilities * np.log2(class_probabilities))


def information_gain(X: np.ndarray, y: np.ndarray, attribute: int) -> float:
    """Compute information gain.

    Args:
        X (np.ndarray): Attributes.
        y (np.ndarray): Class labels.
        attribute (int): Index of attribute used to split data.

    Returns:
        information_gain (float): Reduction in entropy after splitting data.
    """
    original_entropy = entropy(y)
    average_entropy_after_split = 0

    for value in np.unique(X[:, attribute]):
        partition = X[:, attribute] == value
        y_partition = y[partition]
        weight = len(y_partition) / len(y)

        average_entropy_after_split = average_entropy_after_split + weight * entropy(
            y_partition
        )

    return original_entropy - average_entropy_after_split


def gini(y: np.ndarray) -> float:
    """Compute Gini impurity.

    Args:
        y (np.ndarray): Class labels.

    Returns:
        gini (float): Gini impurity of class labels.
    """
    _classes, class_counts = np.unique(y, return_counts=True)
    class_probabilities = class_counts / len(y)

    return 1 - np.sum(class_probabilities**2)


def gini_split(X: np.ndarray, y: np.ndarray, attribute: int) -> float:
    """Compute Gini impurity after splitting on an attribute.

    Args:
        X (np.ndarray): Attributes.
        y (np.ndarray): Class labels.
        attribute (int): Index of attribute used to split data.

    Returns:
        gini_split (float): Weighted average Gini impurity after splitting data.
    """
    average_gini_after_split = 0

    for value in np.unique(X[:, attribute]):
        partition = X[:, attribute] == value
        y_partition = y[partition]
        weight = len(y_partition) / len(y)

        average_gini_after_split = average_gini_after_split + weight * gini(y_partition)

    return average_gini_after_split


def get_best_categorical_attribute(
    X: np.ndarray, y: np.ndarray, testable_attributes: list[int], split_criterion: str
) -> tuple[int, float]:
    """Find best categorical attribute for splitting data.

    Args:
        X (np.ndarray): Attributes.
        y (np.ndarray): Class labels.
        testable_attributes (list): Indices of attributes available for splitting data.
        split_criterion (str): Split criterion, either "information_gain" or "gini".

    Returns:
        best_attribute (int): Index of best attribute for splitting data.
        best_split_quality_score (float): Split quality score of best attribute.
    """
    if split_criterion == "information_gain":
        information_gains = [
            information_gain(X, y, attribute) for attribute in testable_attributes
        ]

        best_attribute_index = np.argmax(information_gains)
        best_attribute = testable_attributes[best_attribute_index]
        best_split_quality_score = information_gains[best_attribute_index]
        return best_attribute, best_split_quality_score

    if split_criterion == "gini":
        gini_splits = [gini_split(X, y, attribute) for attribute in testable_attributes]

        best_attribute_index = np.argmin(gini_splits)
        best_attribute = testable_attributes[best_attribute_index]
        best_split_quality_score = gini(y) - gini_splits[best_attribute_index]
        return best_attribute, best_split_quality_score

    else:
        raise ValueError("split_criterion must be 'information_gain' or 'gini'")


def numeric_to_thresholded_value(numeric_value: float, threshold: float) -> str:
    """Convert numeric value to thresholded value.

    Args:
        numeric_value (float): Numeric value.
        threshold (float): Threshold used to convert the numeric value.

    Returns:
        thresholded_value (str): Thresholded value, either "<=threshold" or ">threshold".
    """
    if float(numeric_value) <= threshold:
        thresholded_value = f"<={threshold}"

    elif float(numeric_value) > threshold:
        thresholded_value = f">{threshold}"

    return thresholded_value


def numeric_to_thresholded_attribute(
    numeric_attribute: np.ndarray, threshold: float
) -> np.ndarray:
    """Convert numeric attribute to thresholded attribute.

    Args:
        numeric_attribute (np.ndarray): Numeric attribute values.
        threshold (float): Threshold used to convert the numeric attribute.

    Returns:
        thresholded_attribute (np.ndarray): Thresholded attribute values.
    """

    thresholded_attribute = [
        numeric_to_thresholded_value(numeric_value, threshold)
        for numeric_value in numeric_attribute
    ]

    thresholded_attribute = np.array(thresholded_attribute).reshape(-1, 1)

    return thresholded_attribute


def get_best_threshold(
    numeric_attribute: np.ndarray, y: np.ndarray, split_criterion: str
) -> float:
    """Find best threshold for numeric attribute.

    Args:
        numeric_attribute (np.ndarray): Numeric attribute values.
        y (np.ndarray): Class labels.
        split_criterion (str): Split criterion, either "information_gain" or "gini".

    Returns:
        best_threshold (float): Threshold that gives the best split.
    """
    # sort instances according to attribute values
    sorted_values = np.sort(np.unique(numeric_attribute))

    if len(sorted_values) == 1:
        return sorted_values[0]

    # threshold as mean values between consecutive sorted values
    thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2

    # pick threshold that maximises criterion of interest
    if split_criterion == "information_gain":
        information_gains = [
            information_gain(
                numeric_to_thresholded_attribute(numeric_attribute, threshold), y, 0
            )
            for threshold in thresholds
        ]

        best_threshold = thresholds[np.argmax(information_gains)]

        return best_threshold

    if split_criterion == "gini":
        gini_splits = [
            gini_split(
                numeric_to_thresholded_attribute(numeric_attribute, threshold), y, 0
            )
            for threshold in thresholds
        ]

        best_threshold = thresholds[np.argmin(gini_splits)]

        return best_threshold

    else:
        raise ValueError("split_criterion must be 'information_gain' or 'gini'")


def get_X_with_thresholded_attributes(
    X: np.ndarray,
    y: np.ndarray,
    testable_attributes: list[int],
    split_criterion: str,
    attribute_types: list[str],
) -> tuple[np.ndarray, list[float | None]]:
    """Convert numeric attributes to thresholded attributes.

    Args:
        X (np.ndarray): Attributes.
        y (np.ndarray): Class labels.
        testable_attributes (list): Indices of attributes available for splitting data.
        split_criterion (str): Split criterion, either "information_gain" or "gini".
        attribute_types (list): Attribute types, either "numeric" or "categorical".

    Returns:
        X_with_thresholded_attributes (np.ndarray): Numeric attributes thresholded. Categorical attributes unchanged.
        thresholds (list): Best threshold for each numeric attribute, and None for other attributes.
    """

    X_with_thresholded_attributes = X.copy().astype(object)
    thresholds = [None] * X.shape[1]

    for attribute in testable_attributes:
        if attribute_types[attribute] == "numeric":
            numeric_attribute = X[:, attribute].astype(float)

            threshold = get_best_threshold(numeric_attribute, y, split_criterion)

            thresholded_attribute = numeric_to_thresholded_attribute(
                numeric_attribute, threshold
            )

            X_with_thresholded_attributes[:, attribute] = thresholded_attribute[:, 0]
            thresholds[attribute] = threshold

    return X_with_thresholded_attributes, thresholds


def get_best_categorical_or_numerical_attribute(
    X: np.ndarray,
    y: np.ndarray,
    testable_attributes: list[int],
    split_criterion: str,
    attribute_types: list[str],
) -> tuple[int, float | None, float]:
    """Find best categorical or numeric attribute for splitting data.

    Args:
        X (np.ndarray): Attributes.
        y (np.ndarray): Class labels.
        testable_attributes (list): Indices of attributes available for splitting data.
        split_criterion (str): Split criterion, either "information_gain" or "gini".
        attribute_types (list): Attribute types, either "numeric" or "categorical".

    Returns:
        best_attribute (int): Index of best attribute for splitting data.
        threshold (float): Best threshold if best attribute is numeric, otherwise None.
        best_split_quality_score (float): Split quality score of best attribute.
    """

    X_with_thresholded_attributes, thresholds = get_X_with_thresholded_attributes(
        X, y, testable_attributes, split_criterion, attribute_types
    )

    best_attribute, best_split_quality_score = get_best_categorical_attribute(
        X_with_thresholded_attributes, y, testable_attributes, split_criterion
    )

    threshold = thresholds[best_attribute]

    return best_attribute, threshold, best_split_quality_score


def get_X_with_thresholded_best_attribute(
    X: np.ndarray,
    best_attribute: int,
    best_attribute_type: str,
    threshold: float | None,
) -> np.ndarray:
    """Convert best attribute to thresholded attribute.

    Args:
        X (np.ndarray): Attributes.
        best_attribute (int): Index of best attribute for splitting data.
        best_attribute_type (str): Attribute type, either "numeric" or "categorical".
        threshold (float): Best threshold if best attribute is numeric, otherwise None.

    Returns:
        X_with_thresholded_best_attribute (np.ndarray): Attributes with the best attribute thresholded (if it is numeric).
    """
    if best_attribute_type == "categorical":
        X_with_thresholded_best_attribute = X

    if best_attribute_type == "numeric":
        X_with_thresholded_best_attribute = X.copy().astype(object)
        thresholded_attribute = numeric_to_thresholded_attribute(
            X[:, best_attribute].astype(float), threshold
        )
        X_with_thresholded_best_attribute[:, best_attribute] = thresholded_attribute[
            :, 0
        ]

    return X_with_thresholded_best_attribute


class leaf_node:
    """Leaf containing the class label returned by a terminal tree branch."""

    def __init__(self, label: object) -> None:
        self.label = label

    def predict(self, X_i: np.ndarray | None = None) -> object:
        """Return this leaf's label.

        Args:
            X_i: Unused feature vector supplied by the tree interface.

        Returns:
            Label stored by the leaf.
        """
        return self.label

    def __eq__(self, other: object) -> bool:
        return isinstance(other, leaf_node) and self.label == other.label


class decision_node:
    """Internal node that routes examples to child subtrees."""

    def __init__(
        self,
        best_attribute: int,
        best_attribute_name: str,
        best_attribute_type: str,
        threshold: float | None,
        majority_class: object,
    ) -> None:
        self.best_attribute = best_attribute
        self.best_attribute_name = best_attribute_name
        self.best_attribute_type = best_attribute_type
        self.threshold = threshold
        self.majority_class = majority_class
        self.edges: dict[object, leaf_node | decision_node] = {}

    def _add_edge(self, label: object, subtree: leaf_node | decision_node) -> None:
        """Attach a subtree for one attribute value.

        Args:
            label: Attribute value labeling the edge.
            subtree: Child node reached by the edge.
        """
        self.edges[label] = subtree

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, decision_node)
            and self.best_attribute == other.best_attribute
            and self.best_attribute_name == other.best_attribute_name
            and self.best_attribute_type == other.best_attribute_type
            and self.threshold == other.threshold
            and self.majority_class == other.majority_class
            and self.edges == other.edges
        )

    def predict(self, X_i: np.ndarray) -> object:
        """Route one feature vector to a child subtree or fallback label.

        Args:
            X_i: Feature vector to classify.

        Returns:
            Prediction returned by the selected child subtree.
        """
        if self.best_attribute_type == "numeric":
            value = numeric_to_thresholded_value(
                X_i[self.best_attribute], self.threshold
            )

        if self.best_attribute_type == "categorical":
            value = X_i[self.best_attribute]

        if value not in self.edges:
            return self.majority_class

        return self.edges[value].predict(X_i)


class decision_tree_classifier:
    """Train a decision tree using categorical or numeric attributes."""

    def __init__(
        self,
        split_criterion: str,
        majority_class_threshold: float,
        minimum_size_for_split: int,
        minimum_split_quality_score: float,
        maximum_depth: int | None,
        random_attribute_selection: bool,
        random_seed: int,
    ) -> None:
        self.split_criterion = split_criterion
        self.majority_class_threshold = majority_class_threshold
        self.minimum_size_for_split = minimum_size_for_split
        self.minimum_split_quality_score = minimum_split_quality_score
        self.maximum_depth = maximum_depth
        self.random_attribute_selection = random_attribute_selection
        self.random_generator = np.random.default_rng(random_seed)
        self.attribute_names = None
        self.attribute_types = None
        self.root = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attribute_names: list[str],
        attribute_types: list[str],
    ) -> decision_tree_classifier:
        """Fit the decision tree classifier.

        Args:
            X (np.ndarray): Attributes.
            y (np.ndarray): Class labels.
            attribute_names (list): Attribute names.
            attribute_types (list): Attribute types, either "numeric" or "categorical".

        Returns:
            self (decision_tree_classifier): Fitted decision tree classifier.
        """
        self.attribute_names = attribute_names
        self.attribute_types = attribute_types

        testable_attributes = list(range(X.shape[1]))
        self.root = self._decision_tree(
            X=X, y=y, testable_attributes=testable_attributes, depth=0
        )

        return self

    def _predict_single_instance(self, X_i: np.ndarray) -> object:
        """Predict one instance by traversing the fitted tree.

        Args:
            X_i: Feature vector to classify.

        Returns:
            Predicted class label.
        """
        if self.root is None:
            raise ValueError("The decision tree must be fitted before prediction.")

        return self.root.predict(X_i)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X (np.ndarray): Attributes.

        Returns:
            predictions (np.ndarray): Predicted class labels.
        """
        if self.root is None:
            raise ValueError("The decision tree must be fitted before prediction.")

        predictions = [self._predict_single_instance(X_i) for X_i in X]

        return np.array(predictions)

    def _decision_tree(
        self, X: np.ndarray, y: np.ndarray, testable_attributes: list[int], depth: int
    ) -> leaf_node | decision_node:
        """Recursively build a subtree from a training partition.

        Args:
            X: Feature matrix for the current partition.
            y: Labels for the current partition.
            testable_attributes: Attribute indices still available for splitting.
            depth: Current depth in the tree.

        Returns:
            Root node for the recursively constructed subtree.
        """

        # dataset D is split into features X and label y
        # testable_attributes is the list of attributes that can still be tested
        classes, class_counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(class_counts)]
        majority_class_proportion = np.max(class_counts) / len(y)

        ###### stopping criteria ######

        # if enough instances in dataset belong to the majority class
        if majority_class_proportion >= self.majority_class_threshold:
            # define node as leaf node labeled with majority class and return it
            node = leaf_node(label=majority_class)
            return node

        # if dataset partition is too small
        if len(y) < self.minimum_size_for_split:
            # define node as leaf node labeled with majority class and return it
            node = leaf_node(label=majority_class)
            return node

        # if tree has reached maximum depth
        if depth >= self.maximum_depth:
            # define node as leaf node labeled with majority class and return it
            node = leaf_node(label=majority_class)
            return node

        # if no more attributes can be tested
        if len(testable_attributes) == 0:
            # define node as a leaf node labeled with majority class and return it
            node = leaf_node(label=majority_class)
            return node

        ###### randomly select testable attributes ######

        if self.random_attribute_selection:
            # m = sqrt(total attributes)
            m = int(np.ceil(np.sqrt(X.shape[1])))
            # randomly select m testable attributes from the complete set of attributes
            testable_attributes = self.random_generator.choice(
                list(range(X.shape[1])), size=m, replace=False
            ).tolist()

        ###### find best attribute ######

        # find best attribute to split dataset, among testable attributes
        best_attribute, threshold, best_split_quality_score = (
            get_best_categorical_or_numerical_attribute(
                X,
                y,
                testable_attributes,
                split_criterion=self.split_criterion,
                attribute_types=self.attribute_types,
            )
        )

        ###### stopping criteria ######

        if best_split_quality_score <= self.minimum_split_quality_score:
            node = leaf_node(label=majority_class)
            return node

        ###### remove best attribute from testable attributes ######

        if not self.random_attribute_selection:
            testable_attributes = [
                attribute
                for attribute in testable_attributes
                if attribute != best_attribute
            ]

        ###### split dataset using best attribute ######

        # define node as decision node that tests best attribute
        node = decision_node(
            best_attribute=best_attribute,
            best_attribute_name=self.attribute_names[best_attribute],
            best_attribute_type=self.attribute_types[best_attribute],
            threshold=threshold,
            majority_class=majority_class,
        )

        # best attribute is thresholded if it is numeric
        X_with_thresholded_best_attribute = get_X_with_thresholded_best_attribute(
            X=X,
            best_attribute=best_attribute,
            best_attribute_type=self.attribute_types[best_attribute],
            threshold=threshold,
        )

        # values of the best attribute
        best_attribute_values = np.unique(
            X_with_thresholded_best_attribute[:, best_attribute]
        )

        for value in best_attribute_values:
            # partition of dataset D with best attribute corresponding to value
            partition = X_with_thresholded_best_attribute[:, best_attribute] == value
            X_partition = X[partition]
            y_partition = y[partition]

            # if partition is empty, subtree is a leaf node labeled with majority class
            if len(y_partition) == 0:
                subtree = leaf_node(label=majority_class)

            # otherwise recursively construct the subtree
            else:
                subtree = self._decision_tree(
                    X=X_partition,
                    y=y_partition,
                    testable_attributes=testable_attributes,
                    depth=depth + 1,
                )

                # create edge from node to root of subtree, labeling edge with attribute value
                node._add_edge(label=value, subtree=subtree)

        return node
