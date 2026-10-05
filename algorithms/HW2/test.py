import numpy as np

from src.loaders import instances_to_X_and_y
from src.preprocessors import doc_to_bow_vector, docs_to_bow_vectors
from src.metrics import (compute_accuracy,compute_precision,
                         compute_recall,compute_confusion_matrix)
from src.multinomial_naive_bayes import (
    compute_class_probabilities_given_doc,
    compute_class_probability_given_doc,
    compute_class_prior_probabilities,
    compute_word_probabilities_given_class)


def test_instances_to_X_and_y():
    pos_instances = [['good', 'awesome'], ['good'], ['good']]
    neg_instances = [['bad'], ['bad']]
    X, y = instances_to_X_and_y(pos_instances, neg_instances)
    expected_X = [['good', 'awesome'], ['good'], ['good'], ['bad'], ['bad']]
    expected_y = ['positive','positive','positive','negative','negative']
    assert X == expected_X and y == expected_y


def test_doc_to_bow_vector():
    vocab = ["good", "awesome", "bad"]
    doc = ["good", "awesome"]
    bow_vector = doc_to_bow_vector(doc, vocab)
    expected_bow_vector = [1, 1, 0]
    assert bow_vector == expected_bow_vector


def test_docs_to_bow_vectors():
    vocab = ["good", "awesome", "bad"]
    docs = [["awesome", "awesome", "bad"],
            ["good", "awesome"]]
    bow_vectors = docs_to_bow_vectors(docs, vocab)
    expected_bow_vectors = [[0, 2, 1], [1, 1, 0]]
    assert bow_vectors == expected_bow_vectors


def test_compute_class_prior_probabilities():
    y = ['positive','positive','positive','negative','negative']

    # log_scale=False
    class_prior_probabilities = compute_class_prior_probabilities(y, log_scale=False)
    expected_class_prior_probabilities = {'positive': 0.6, 'negative': 0.4}
    assert class_prior_probabilities == expected_class_prior_probabilities

    # log_scale=True
    class_prior_probabilities = compute_class_prior_probabilities(y, log_scale=True)
    expected_class_prior_probabilities = {'positive': np.log(0.6), 'negative': np.log(0.4)}
    assert class_prior_probabilities == expected_class_prior_probabilities


def test_compute_word_probabilities_given_class():
    X = [['good', 'good', 'good'],
         ['good', 'awesome'],
         ['bad', 'bad'],
         ['bad', 'bad']]
    y = ['positive', 'positive', 'negative', 'negative']
    vocab = ['good', 'bad', 'awesome']

    # alpha = 0, log_scale = False
    word_probabilities_given_class = compute_word_probabilities_given_class(
        X, y, vocab, alpha=0, log_scale=False)
    expected_word_probabilities_given_class = {
        'positive': np.array([0.8, 0, 0.2]),
        'negative': np.array([0, 1, 0])}
    assert (word_probabilities_given_class['positive'] == expected_word_probabilities_given_class['positive']).all()
    assert (word_probabilities_given_class['negative'] == expected_word_probabilities_given_class['negative']).all()

    # alpha = 1, log_scale = False
    word_probabilities_given_class = compute_word_probabilities_given_class(
        X, y, vocab, alpha=1, log_scale=False)
    expected_word_probabilities_given_class = {
        "positive": np.array([5 / 8, 1 / 8, 2 / 8]),
        "negative": np.array([1 / 7, 5 / 7, 1 / 7])}
    assert (word_probabilities_given_class['positive'] == expected_word_probabilities_given_class['positive']).all()
    assert (word_probabilities_given_class['negative'] == expected_word_probabilities_given_class['negative']).all()

    # alpha = 1, log_scale = True
    word_probabilities_given_class = compute_word_probabilities_given_class(
        X, y, vocab, alpha=1, log_scale=True)
    expected_word_probabilities_given_class = {
        "positive": np.array([np.log(5) - np.log(8), np.log(1) - np.log(8), np.log(2) - np.log(8)]),
        "negative": np.array([np.log(1) - np.log(7), np.log(5) - np.log(7), np.log(1) - np.log(7)])}
    assert (word_probabilities_given_class['positive'] == expected_word_probabilities_given_class['positive']).all()
    assert (word_probabilities_given_class['negative'] == expected_word_probabilities_given_class['negative']).all()


def test_compute_class_probability_given_doc():
    word_probabilities_given_class = np.array([0.5, 0.25, 0.25])
    bow_vector = np.array([2, 1, 0])

    # log_scale = False
    class_prior_probability = 0.6
    class_probability_given_doc = compute_class_probability_given_doc(
        class_prior_probability,
        word_probabilities_given_class,
        bow_vector,
        log_scale=False)
    expected_class_probability_given_doc = 0.6 * (0.5 ** 2) * (0.25 ** 1) * (0.25 ** 0)
    assert class_probability_given_doc == expected_class_probability_given_doc

    # log_scale = True
    class_prior_probability = np.log(0.6)
    word_log_probabilities_given_class = np.log(word_probabilities_given_class)
    class_probability_given_doc = compute_class_probability_given_doc(
        class_prior_probability,
        word_log_probabilities_given_class,
        bow_vector,
        log_scale=True)
    expected_class_probability_given_doc = (
        np.log(0.6) + 2 * np.log(0.5) + 1 * np.log(0.25) + 0 * np.log(0.25))
    assert class_probability_given_doc == expected_class_probability_given_doc


def test_compute_class_probabilities_given_doc():
    bow_vector = np.array([2, 1, 0])

    # log_scale = False
    class_prior_probabilities = {
        'positive': 0.6,
        'negative': 0.4}
    word_probabilities_given_class = {
        'positive': np.array([0.5, 0.25, 0.25]),
        'negative': np.array([0.1, 0.7, 0.2])}

    class_probabilities_given_doc = compute_class_probabilities_given_doc(
        class_prior_probabilities,
        word_probabilities_given_class,
        bow_vector,
        log_scale=False)

    expected_class_probabilities_given_doc = {
        'positive': 0.6 * (0.5 ** 2) * (0.25 ** 1) * (0.25 ** 0),
        'negative': 0.4 * (0.1 ** 2) * (0.7 ** 1) * (0.2 ** 0)}
    assert class_probabilities_given_doc == expected_class_probabilities_given_doc

    # log_scale = True
    class_prior_probabilities = {
        'positive': np.log(0.6),
        'negative': np.log(0.4)}
    word_probabilities_given_class = {
        'positive': np.log(np.array([0.5, 0.25, 0.25])),
        'negative': np.log(np.array([0.1, 0.7, 0.2]))}

    class_probabilities_given_doc = compute_class_probabilities_given_doc(
        class_prior_probabilities,
        word_probabilities_given_class,
        bow_vector,
        log_scale=True)

    expected_class_probabilities_given_doc = {
        'positive': np.log(0.6) + 2 * np.log(0.5) + 1 * np.log(0.25) + 0 * np.log(0.25),
        'negative': np.log(0.4) + 2 * np.log(0.1) + 1 * np.log(0.7) + 0 * np.log(0.2)}
    assert class_probabilities_given_doc == expected_class_probabilities_given_doc


def test_compute_accuracy():
    y_pred = ['positive', 'positive', 'positive', 'negative']
    y_true = ['negative', 'positive', 'positive', 'negative']
    accuracy = compute_accuracy(y_pred,y_true)
    expected_accuracy = 3/4
    assert accuracy == expected_accuracy


def test_compute_precision():
    y_pred = ['positive', 'positive', 'positive', 'negative']
    y_true = ['negative', 'positive', 'positive', 'negative']
    precision = compute_precision(y_pred, y_true, pos_class='positive')
    expected_precision = 2 / 3
    assert precision == expected_precision


def test_compute_recall():
    y_pred = ['positive', 'positive', 'positive', 'negative']
    y_true = ['negative', 'positive', 'positive', 'negative']
    recall = compute_recall(y_pred, y_true, pos_class='positive')
    expected_recall = 2 / 2
    assert recall == expected_recall


def test_compute_confusion_matrix():
    y_pred = ['positive', 'positive', 'positive', 'negative']
    y_true = ['negative', 'positive', 'positive', 'negative']

    confusion_matrix = compute_confusion_matrix(
        y_pred,
        y_true,
        pos_class='positive',
        neg_class='negative')

    expected_confusion_matrix = np.array([
        [2, 0],
        [1, 1]])

    assert (confusion_matrix == expected_confusion_matrix).all()


if __name__ == '__main__':
    print("Running unit tests.")

    test_instances_to_X_and_y()
    test_doc_to_bow_vector()
    test_docs_to_bow_vectors()
    test_compute_class_prior_probabilities()
    test_compute_word_probabilities_given_class()
    test_compute_class_probability_given_doc()
    test_compute_class_probabilities_given_doc()
    test_compute_accuracy()
    test_compute_precision()
    test_compute_recall()
    test_compute_confusion_matrix()

    print("Unit tests passed.")
