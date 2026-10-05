"""Functions for computing evaluation metrics"""

import numpy as np

def compute_accuracy(y_pred, y_true):
    """Compute accuracy.

    Args:
        y_pred (list): Predicted class labels.
        y_true (list): True class labels.

    Returns:
        accuracy (float): Fraction of predictions that match the true labels.
    """
    correct_pred = sum(pred_class == true_class for pred_class, true_class in zip(y_pred, y_true))
    all_pred = len(y_true)
    accuracy = correct_pred / all_pred
    return accuracy


def compute_precision(y_pred, y_true, pos_class):
    """Compute precision for the positive class.

    Args:
        y_pred (list): Predicted class labels.
        y_true (list): True class labels.
        pos_class (str): Positive class.

    Returns:
        precision (float): Fraction of true positives among predicted positive instances.
    """

    true_positive = sum(
        pred_class == pos_class and true_class == pos_class
        for pred_class, true_class in zip(y_pred, y_true))

    false_positive = sum(
        pred_class == pos_class and true_class != pos_class
        for pred_class, true_class in zip(y_pred, y_true))

    if true_positive + false_positive == 0:
        return np.nan

    precision = true_positive / (true_positive + false_positive)

    return precision


def compute_recall(y_pred, y_true, pos_class):
    """Compute recall for the positive class.

    Args:
        y_pred (list): Predicted class labels.
        y_true (list): True class labels.
        pos_class (str): Positive class.

    Returns:
        recall (float): Fraction of true positive instances correctly predicted as positive.
    """

    true_positive = sum(
        pred_class == pos_class and true_class == pos_class
        for pred_class, true_class in zip(y_pred, y_true))

    false_negative = sum(
        pred_class != pos_class and true_class == pos_class
        for pred_class, true_class in zip(y_pred, y_true))

    if true_positive + false_negative == 0:
        return np.nan

    recall = true_positive / (true_positive + false_negative)
    return recall


def compute_f1(y_pred, y_true, pos_class):
    """Compute F1 score for the positive class.

    Args:
        y_pred (list): Predicted class labels.
        y_true (list): True class labels.
        pos_class (str): Positive class.

    Returns:
        f1 (float): Harmonic mean of precision and recall.
    """

    precision = compute_precision(y_pred, y_true, pos_class)
    recall = compute_recall(y_pred, y_true, pos_class)

    if precision + recall == 0:
        return np.nan

    f1 = 2 * precision * recall / (precision + recall)
    return f1
