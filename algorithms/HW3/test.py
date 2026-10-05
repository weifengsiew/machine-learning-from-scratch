"""Unit tests for functions in decision_tree.py. 

Importantly, the correctness of the decision tree algorithm is assessed by fitting 
on the tenniss dataset, and checking if the correct decision tree is constructed.
"""

import numpy as np

from src.decision_tree import (
    decision_node,
    decision_tree_classifier,
    entropy,
    get_X_with_thresholded_attributes,
    get_X_with_thresholded_best_attribute,
    get_best_categorical_attribute,
    get_best_categorical_or_numerical_attribute,
    get_best_threshold,
    gini,
    gini_split,
    information_gain,
    leaf_node,
    numeric_to_thresholded_attribute,
    numeric_to_thresholded_value,
)


tennis_dataset_columns = [
    "Weather", "Temperature", "Humidity", "Windy", "PlayTennis"]

tennis_dataset = np.array([
    ["Sunny", "Hot", "High", "False", "No"],
    ["Sunny", "Hot", "High", "True", "No"],
    ["Overcast", "Hot", "High", "False", "Yes"],
    ["Rainy", "Mild", "High", "False", "Yes"],
    ["Rainy", "Cool", "Normal", "False", "Yes"],
    ["Rainy", "Cool", "Normal", "True", "No"],
    ["Overcast", "Cool", "Normal", "True", "Yes"],
    ["Sunny", "Mild", "High", "False", "No"],
    ["Sunny", "Cool", "Normal", "False", "Yes"],
    ["Rainy", "Mild", "Normal", "False", "Yes"],
    ["Sunny", "Mild", "Normal", "True", "Yes"],
    ["Overcast", "Mild", "High", "True", "Yes"],
    ["Overcast", "Hot", "Normal", "False", "Yes"],
    ["Rainy", "Mild", "High", "True", "No"]])


def test_entropy():
    PlayTennis = tennis_dataset_columns.index("PlayTennis")

    # y is the PlayTennis column vector
    y = tennis_dataset[:, PlayTennis]
    expected_entropy = -(9 / 14) * np.log2(9 / 14) - (5 / 14) * np.log2(5 / 14)

    assert entropy(y) == expected_entropy


def test_information_gain():
    Weather = tennis_dataset_columns.index("Weather")
    PlayTennis = tennis_dataset_columns.index("PlayTennis")

    # split data into features and labels
    X = tennis_dataset[:, :PlayTennis]
    y = tennis_dataset[:, PlayTennis]

    original_entropy = -(9 / 14) * np.log2(9 / 14) - (5 / 14) * np.log2(5 / 14)

    sunny_entropy = -(2 / 5) * np.log2(2 / 5) - (3 / 5) * np.log2(3 / 5)
    overcast_entropy = -(4 / 4) * np.log2(4 / 4) - 0
    rainy_entropy = -(3 / 5) * np.log2(3 / 5) - (2 / 5) * np.log2(2 / 5)

    average_entropy_after_split = (
        (5 / 14) * sunny_entropy
        + (4 / 14) * overcast_entropy
        + (5 / 14) * rainy_entropy)

    expected_information_gain = original_entropy - average_entropy_after_split

    assert information_gain(X, y, Weather) == expected_information_gain


def test_gini():
    PlayTennis = tennis_dataset_columns.index("PlayTennis")
    y = tennis_dataset[:, PlayTennis]

    expected_gini = 1 - ((9 / 14) ** 2 + (5 / 14) ** 2)

    assert gini(y) == expected_gini


def test_gini_split():
    Weather = tennis_dataset_columns.index("Weather")
    PlayTennis = tennis_dataset_columns.index("PlayTennis")

    # split data into features and labels
    X = tennis_dataset[:, :PlayTennis]
    y = tennis_dataset[:, PlayTennis]

    sunny_gini = 1 - ((2 / 5) ** 2 + (3 / 5) ** 2)
    overcast_gini = 1 - ((4 / 4) ** 2 + (0 / 4) ** 2)
    rainy_gini = 1 - ((3 / 5) ** 2 + (2 / 5) ** 2)

    expected_gini_split = (
        (5 / 14) * sunny_gini
        + (4 / 14) * overcast_gini
        + (5 / 14) * rainy_gini)

    assert gini_split(X, y, Weather) == expected_gini_split


def test_get_best_categorical_attribute():
    Weather = tennis_dataset_columns.index("Weather")
    PlayTennis = tennis_dataset_columns.index("PlayTennis")

    # split data into features and labels
    X = tennis_dataset[:, :PlayTennis]
    y = tennis_dataset[:, PlayTennis]

    testable_attributes = list(range(X.shape[1]))

    expected_best_attribute = Weather

    best_attribute, best_split_quality_score = get_best_categorical_attribute(
        X, y, testable_attributes, "information_gain")

    assert best_attribute == expected_best_attribute
    assert best_split_quality_score == information_gain(X, y, Weather)

    best_attribute, best_split_quality_score = get_best_categorical_attribute(
        X, y, testable_attributes, "gini")

    assert best_attribute == expected_best_attribute
    assert best_split_quality_score == gini(y) - gini_split(X, y, Weather)

    try:
        get_best_categorical_attribute(X, y, testable_attributes, "invalid")
        assert False
    except ValueError:
        assert True


def test_numeric_to_thresholded_value():
    threshold = 25.5

    assert numeric_to_thresholded_value(19, threshold) == "<=25.5"
    assert numeric_to_thresholded_value(25.5, threshold) == "<=25.5"
    assert numeric_to_thresholded_value(35, threshold) == ">25.5"


age_risk_dataset_columns = ["Age", "Gender", "Risk"]

age_risk_dataset = np.array([
    [18, "M", "High Risk"],
    [19, "F", "High Risk"],
    [21, "F", "High Risk"],
    [30, "M", "Low Risk"],
    [35, "F", "Low Risk"],
    [43, "M", "Low Risk"],
    [90, "M", "Low Risk"]])


def test_numeric_to_thresholded_attribute():
    Age = age_risk_dataset_columns.index("Age")
    numeric_attribute = age_risk_dataset[:, Age].astype(float)

    thresholded_attribute = numeric_to_thresholded_attribute(numeric_attribute, 25.5)

    expected_thresholded_attribute = np.array([
        ["<=25.5"], ["<=25.5"], ["<=25.5"], [">25.5"],
        [">25.5"], [">25.5"], [">25.5"]])

    assert np.array_equal(thresholded_attribute, expected_thresholded_attribute)


def test_get_best_threshold():
    Age = age_risk_dataset_columns.index("Age")
    Risk = age_risk_dataset_columns.index("Risk")

    numeric_attribute = age_risk_dataset[:, Age].astype(float)
    y = age_risk_dataset[:, Risk]

    expected_threshold = (21 + 30) / 2

    assert get_best_threshold(numeric_attribute, y, "information_gain") == expected_threshold
    assert get_best_threshold(numeric_attribute, y, "gini") == expected_threshold


def test_get_X_with_thresholded_attributes():
    Age = age_risk_dataset_columns.index("Age")
    Gender = age_risk_dataset_columns.index("Gender")
    Risk = age_risk_dataset_columns.index("Risk")

    # split data into features and labels
    X = age_risk_dataset[:, :Risk]
    y = age_risk_dataset[:, Risk]

    testable_attributes = [Age, Gender]
    attribute_types = ["numeric", "categorical"]

    X_with_thresholded_attributes, thresholds = (
        get_X_with_thresholded_attributes(
            X, y, testable_attributes, "information_gain", attribute_types))

    expected_X_with_thresholded_attributes = np.array([
        ["<=25.5", "M"], ["<=25.5", "F"], ["<=25.5", "F"],
        [">25.5", "M"], [">25.5", "F"], [">25.5", "M"],
        [">25.5", "M"]])

    expected_thresholds = [25.5, None]

    assert np.array_equal(
        X_with_thresholded_attributes, expected_X_with_thresholded_attributes)
    assert thresholds == expected_thresholds


def test_get_best_categorical_or_numerical_attribute():
    Age = age_risk_dataset_columns.index("Age")
    Gender = age_risk_dataset_columns.index("Gender")
    Risk = age_risk_dataset_columns.index("Risk")

    # split data into features and labels
    X = age_risk_dataset[:, :Risk]
    y = age_risk_dataset[:, Risk]

    testable_attributes = [Age, Gender]
    attribute_types = ["numeric", "categorical"]

    expected_best_attribute = Age
    expected_threshold = 25.5

    best_attribute, threshold, best_split_quality_score = (
        get_best_categorical_or_numerical_attribute(
            X, y, testable_attributes, "information_gain", attribute_types))

    expected_best_split_quality_score = information_gain(
        numeric_to_thresholded_attribute(age_risk_dataset[:, Age].astype(float), 25.5),
        y, 0)

    assert best_attribute == expected_best_attribute
    assert threshold == expected_threshold
    assert best_split_quality_score == expected_best_split_quality_score

    best_attribute, threshold, best_split_quality_score = (
        get_best_categorical_or_numerical_attribute(
            X, y, testable_attributes, "gini", attribute_types))

    thresholded_attribute = numeric_to_thresholded_attribute(
        age_risk_dataset[:, Age].astype(float), 25.5)
    expected_best_split_quality_score = gini(y) - gini_split(thresholded_attribute, y, 0)

    assert best_attribute == expected_best_attribute
    assert threshold == expected_threshold
    assert best_split_quality_score == expected_best_split_quality_score


def test_get_X_with_thresholded_best_attribute():
    Age = age_risk_dataset_columns.index("Age")
    Gender = age_risk_dataset_columns.index("Gender")
    Risk = age_risk_dataset_columns.index("Risk")

    # split data into features and labels
    X = age_risk_dataset[:, :Risk]

    # best_attribute_type="categorical"
    X_with_thresholded_best_attribute = get_X_with_thresholded_best_attribute(
        X, Gender, "categorical", threshold=None)
    assert X_with_thresholded_best_attribute is X

    # best_attribute_type="numeric"
    X_with_thresholded_best_attribute = get_X_with_thresholded_best_attribute(
        X, Age, "numeric", threshold=25.5)

    expected_X_with_thresholded_best_attribute = np.array([
        ["<=25.5", "M"], ["<=25.5", "F"], ["<=25.5", "F"],
        [">25.5", "M"], [">25.5", "F"], [">25.5", "M"],
        [">25.5", "M"]])

    assert np.array_equal(
        X_with_thresholded_best_attribute, expected_X_with_thresholded_best_attribute)


def test_decision_tree_classifier_fit():
    Weather = tennis_dataset_columns.index("Weather")
    Humidity = tennis_dataset_columns.index("Humidity")
    Windy = tennis_dataset_columns.index("Windy")
    PlayTennis = tennis_dataset_columns.index("PlayTennis")

    # split data into features and labels
    X = tennis_dataset[:, :PlayTennis]
    y = tennis_dataset[:, PlayTennis]

    # initialise decision tree classifier
    classifier = decision_tree_classifier("information_gain", 1.0, 1, 0, False, None)

    # train classifier
    classifier.fit(
        X, y,
        attribute_names=["Weather", "Temperature", "Humidity", "Windy"],
        attribute_types=["categorical", "categorical", "categorical", "categorical"])

    # constructed decision tree
    root = classifier.root

    # expected decision tree
    expected_root = decision_node(
        best_attribute=Weather, best_attribute_name="Weather",
        best_attribute_type="categorical", threshold=None, majority_class="Yes")

    expected_root.add_edge("Overcast", leaf_node("Yes"))

    windy_node = decision_node(
        best_attribute=Windy, best_attribute_name="Windy",
        best_attribute_type="categorical", threshold=None, majority_class="Yes")
    expected_root.add_edge("Rainy", windy_node)
    windy_node.add_edge("False", leaf_node("Yes"))
    windy_node.add_edge("True", leaf_node("No"))

    humidity_node = decision_node(
        best_attribute=Humidity, best_attribute_name="Humidity",
        best_attribute_type="categorical", threshold=None, majority_class="No")
    expected_root.add_edge("Sunny", humidity_node)
    humidity_node.add_edge("High", leaf_node("No"))
    humidity_node.add_edge("Normal", leaf_node("Yes"))

    assert root == expected_root


if __name__ == "__main__":

    test_entropy()
    test_information_gain()
    test_gini()
    test_gini_split()
    test_get_best_categorical_attribute()
    test_numeric_to_thresholded_value()
    test_numeric_to_thresholded_attribute()
    test_get_best_threshold()
    test_get_X_with_thresholded_attributes()
    test_get_best_categorical_or_numerical_attribute()
    test_get_X_with_thresholded_best_attribute()
    test_decision_tree_classifier_fit()
