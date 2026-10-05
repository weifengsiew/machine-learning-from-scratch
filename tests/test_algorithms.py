import numpy as np

from ml_from_scratch import (
    DecisionTreeClassifier,
    KNNClassifier,
    MultinomialNaiveBayes,
    NeuralNetwork,
    RandomForestClassifier,
    initialize_weights,
)


def test_knn_and_tree_learn_simple_data():
    X = np.array([[0], [1], [10], [11]], dtype=float)
    y = np.array([0, 0, 1, 1])
    assert np.array_equal(KNNClassifier(1).fit(X, y).predict(X), y)
    assert np.array_equal(DecisionTreeClassifier().fit(X, y, ["numeric"]).predict(X), y)


def test_random_forest_predicts_labels():
    X = np.array([[0], [1], [10], [11]], dtype=float)
    y = np.array([0, 0, 1, 1])
    model = RandomForestClassifier(n_trees=5, random_state=7).fit(X, y, ["numeric"])
    assert set(model.predict(X)) <= {0, 1}


def test_multinomial_naive_bayes():
    model = MultinomialNaiveBayes().fit(
        [["good", "calm"], ["bad", "loud"]], ["pos", "neg"])
    assert model.predict([["good"], ["bad"]]) == ["pos", "neg"]


def test_neural_network_can_fit_or_gate():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1])
    model = NeuralNetwork(initialize_weights(2, 1, 4, 1, random_state=4))
    model.fit(X, y, learning_rate=1.0, epochs=300, random_state=4)
    assert model.predict(X).shape == (4,)
