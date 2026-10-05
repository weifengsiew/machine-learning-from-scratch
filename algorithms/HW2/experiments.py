"""Functions to run experiments and plot experiment results"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from src.metrics import (compute_accuracy, compute_confusion_matrix,
                         compute_precision,compute_recall)

from src.multinomial_naive_bayes import multinomial_naive_bayes_classifier
from src.loaders import instances_to_X_and_y, load_test_set, load_training_set


def run_experiment(
    percentage_positives_train, percentage_negatives_train,
    percentage_positives_test, percentage_negatives_test,
    alpha, log_scale,
    pos_class, neg_class):

    """Run one experiment.

    This involves fitting the multinomial naive bayes classifier on the train set,
    predicting on the test set and computing evaluating metrics for predictions.

    Args:
        percentage_positives_train (float): Fraction of positive training documents to use.
        percentage_negatives_train (float): Fraction of negative training documents to use.
        percentage_positives_test (float): Fraction of positive test documents to use.
        percentage_negatives_test (float): Fraction of negative test documents to use.
        alpha (float): Laplace smoothing parameter.
        log_scale (bool): Whether to use probabilities (False) or log-probabilities (True).
        pos_class (str): Positive class.
        neg_class (str): Negative class.

    Returns:
        tuple: Accuracy, precision, recall, and confusion matrix for the experiment.
    """

    # load training set
    pos_train, neg_train, vocab = load_training_set(
        percentage_positives=percentage_positives_train,
        percentage_negatives=percentage_negatives_train)
    X_train, y_train = instances_to_X_and_y(pos_train, neg_train)

    # load test set
    pos_test, neg_test = load_test_set(
        percentage_positives=percentage_positives_test,
        percentage_negatives=percentage_negatives_test)
    X_test, y_test = instances_to_X_and_y(pos_test, neg_test)

    # initialize classifier
    classifier = multinomial_naive_bayes_classifier(alpha=alpha, log_scale=log_scale)

    # train classifier
    classifier.fit(X_train, y_train, vocab)

    # predict instances in test set
    y_pred = classifier.predict(X_test)

    # compute accuracy, precision, recall and confusion matrix
    accuracy = compute_accuracy(y_pred, y_test)
    precision = compute_precision(y_pred, y_test, pos_class)
    recall = compute_recall(y_pred, y_test, pos_class)
    confusion_matrix = compute_confusion_matrix(y_pred, y_test, pos_class, neg_class)

    return accuracy, precision, recall, confusion_matrix


def search_for_best_alpha(
    alphas,
    percentage_positives_train, percentage_negatives_train,
    percentage_positives_test, percentage_negatives_test,
    log_scale, n_jobs):
    """Search for the Laplace smoothing parameter with the highest accuracy.

    Args:
        alphas (list): Laplace smoothing parameter values to evaluate.
        percentage_positives_train (float): Fraction of positive training documents to use.
        percentage_negatives_train (float): Fraction of negative training documents to use.
        ercentage_positives_test (float): Fraction of positive test documents to use.
        percentage_negatives_test (float): Fraction of negative test documents to use.
        log_scale (bool): Whether to use probabilities (False) or log-probabilities (True).
        n_jobs (int): Number of parallel jobs to use.

    Returns:
        tuple: Alpha values, corresponding accuracies, and the alpha with the highest accuracy.
    """

    # load training set
    pos_train, neg_train, vocab = load_training_set(
        percentage_positives=percentage_positives_train,
        percentage_negatives=percentage_negatives_train)
    X_train, y_train = instances_to_X_and_y(pos_train, neg_train)

    # load test set
    pos_test, neg_test = load_test_set(
        percentage_positives=percentage_positives_test,
        percentage_negatives=percentage_negatives_test)
    X_test, y_test = instances_to_X_and_y(pos_test, neg_test)

    def compute_accuracy_for_alpha(alpha):
        # laplace smoothing with alpha and log probabilities
        classifier = multinomial_naive_bayes_classifier(alpha, log_scale=log_scale)
        # train classifier
        classifier.fit(X_train, y_train, vocab)
        # predict instances in test set
        y_pred = classifier.predict(X_test)
        # record accuracy
        return compute_accuracy(y_pred, y_test)

    accuracies = Parallel(n_jobs=n_jobs)(
        delayed(compute_accuracy_for_alpha)(alpha)
        for alpha in alphas)

    best_index = np.argmax(accuracies)
    best_alpha = alphas[best_index]

    return alphas, accuracies, best_alpha


def plot_experiment_results(
    accuracy, precision, recall, confusion_matrix,
    pos_class, neg_class, save_to_path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # plot accuracy, precision and recall
    metric_names = ["Accuracy", "Precision", "Recall"]
    metric_values = [accuracy, precision, recall]
    axes[0].bar(metric_names, metric_values)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Metrics")

    for index, value in enumerate(metric_values):
        axes[0].text(index, value + 0.02, round(value, 3), ha="center")

    # visualize confusion matrix
    axes[1].imshow(confusion_matrix, cmap="Blues")
    axes[1].set_title("Confusion Matrix")
    axes[1].set_xticks([0, 1], [pos_class, neg_class])
    axes[1].set_yticks([0, 1], [pos_class, neg_class])
    axes[1].set_xlabel("Predicted class")
    axes[1].set_ylabel("True class")

    for row in range(2):
        for col in range(2):
            axes[1].text(col, row, confusion_matrix[row, col], ha="center", va="center")

    plt.tight_layout()

    save_to_path = Path(save_to_path)
    save_to_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_to_path)

    plt.close(fig)


def plot_alpha_search_results(alphas, accuracies, save_to_path):
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, accuracies, marker='o')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()

    save_to_path = Path(save_to_path)
    save_to_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_to_path)

    plt.close()
