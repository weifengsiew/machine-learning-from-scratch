import numpy as np


def entropy(y):
    classes, class_counts = np.unique(y, return_counts=True)
    class_probabilities = class_counts / len(y)

    return -np.sum(class_probabilities * np.log2(class_probabilities))


def information_gain(X, y, attribute):
    original_entropy = entropy(y)
    average_entropy_after_split = 0

    for value in np.unique(X[:, attribute]):
        partition = X[:, attribute] == value
        y_partition = y[partition]
        weight = len(y_partition) / len(y)

        average_entropy_after_split = (
            average_entropy_after_split + weight * entropy(y_partition))

    return original_entropy - average_entropy_after_split


def gini(y):
    classes, class_counts = np.unique(y, return_counts=True)
    class_probabilities = class_counts / len(y)

    return 1 - np.sum(class_probabilities ** 2)


def gini_split(X, y, attribute):
    average_gini_after_split = 0

    for value in np.unique(X[:, attribute]):
        partition = X[:, attribute] == value
        y_partition = y[partition]
        weight = len(y_partition) / len(y)

        average_gini_after_split = (
            average_gini_after_split + weight * gini(y_partition))

    return average_gini_after_split


def get_best_attribute(X, y, testable_attributes, split_criterion):
    if split_criterion == "information_gain":
        information_gains = [information_gain(X, y, attribute) for attribute in testable_attributes]

        best_attribute = testable_attributes[np.argmax(information_gains)]
        return best_attribute

    if split_criterion == "gini":
        gini_splits = [gini_split(X, y, attribute) for attribute in testable_attributes]

        best_attribute = testable_attributes[np.argmin(gini_splits)]
        return best_attribute

    raise ValueError("split_criterion must be 'information_gain' or 'gini'")


class leaf_node:
    def __init__(self, label):
        self.label = label

    def predict(self, X_i=None):
        return self.label


class decision_node:
    def __init__(self, best_attribute, majority_class):
        self.best_attribute = best_attribute
        self.majority_class = majority_class
        self.edges = {}

    def add_edge(self, label, subtree):
        self.edges[label] = subtree

    def predict(self, X_i):
        value = X_i[self.best_attribute]

        if value not in self.edges:
            return self.majority_class

        return self.edges[value].predict(X_i)


class decision_tree_classifier:
    def __init__(self, split_criterion, majority_class_threshold):
        self.split_criterion = split_criterion
        self.majority_class_threshold = majority_class_threshold
        self.root = None

    def fit(self, X, y):
        testable_attributes = list(range(X.shape[1]))
        self.root = self.decision_tree(
            X=X, y=y, testable_attributes=testable_attributes)

        return self

    def predict_single_instance(self, X_i):
        if self.root is None:
            raise ValueError("The decision tree must be fitted before prediction.")

        return self.root.predict(X_i)

    def predict(self, X):
        if self.root is None:
            raise ValueError("The decision tree must be fitted before prediction.")

        predictions = [self.predict_single_instance(X_i) for X_i in X]

        return np.array(predictions)

    def decision_tree(self, X, y, testable_attributes):
        # dataset D is split into features X and label y
        # testable_attributes is the list of attributes that can still be tested

        # create a new node
        node = None

        classes, class_counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(class_counts)]
        majority_class_proportion = np.max(class_counts) / len(y)

        # stopping criteria

        # if enough instances in dataset belong to the majority class
        if majority_class_proportion >= self.majority_class_threshold:
            # define node as leaf node labeled with majority class and return it
            node = leaf_node(label=majority_class)
            return node

        # if no more attributes can be tested
        if len(testable_attributes) == 0:
            # define node as a leaf node labeled with majority class and return it
            node = leaf_node(label=majority_class)
            return node

        # best attribute to split the dataset
        best_attribute = get_best_attribute(
            X, y, testable_attributes, split_criterion=self.split_criterion)

        # define node as decision node that tests best attribute
        node = decision_node(
            best_attribute=best_attribute, majority_class=majority_class)

        # remove best attribute from testable_attributes
        testable_attributes = [attribute for attribute in testable_attributes
                               if attribute != best_attribute]

        # values of the best attribute
        best_attribute_values = np.unique(X[:, best_attribute])

        for value in best_attribute_values:
            # partition of dataset D with best attribute corresponding to value
            partition = X[:, best_attribute] == value
            X_partition = X[partition]
            y_partition = y[partition]

            # if partition is empty, subtree is a leaf node labeled with majority class
            if len(y_partition) == 0:
                subtree = leaf_node(label=majority_class)

            # otherwise recursively construct the subtree
            else:
                subtree = self.decision_tree(
                    X=X_partition,
                    y=y_partition,
                    testable_attributes=testable_attributes)

            # create edge from node to root of subtree, labeling edge with attribute value
            node.add_edge(label=value, subtree=subtree)

        return node
