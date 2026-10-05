"""Functions to compute evaluation metrics"""

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

    recall = true_positive / (true_positive + false_negative)
    return recall


def compute_confusion_matrix(y_pred, y_true, pos_class, neg_class):
    """Return a confusion matrix.

    Args:
        y_pred (list): Predicted class labels.
        y_true (list): True class labels.
        pos_class (str): Positive class.
        neg_class (str): Negative class.
    Returns:
        confusion_matrix (np.ndarray): [[true_positive, false_negative],[false_positive, true_negative]].
    """
    
    true_positive = sum(
        pred_class == pos_class and true_class == pos_class
        for pred_class, true_class in zip(y_pred, y_true))

    false_negative = sum(
        pred_class == neg_class and true_class == pos_class
        for pred_class, true_class in zip(y_pred, y_true))

    false_positive = sum(
        pred_class == pos_class and true_class == neg_class
        for pred_class, true_class in zip(y_pred, y_true))

    true_negative = sum(
        pred_class == neg_class and true_class == neg_class
        for pred_class, true_class in zip(y_pred, y_true))

    confusion_matrix = np.array([
        [true_positive, false_negative],
        [false_positive, true_negative],])

    return confusion_matrix
