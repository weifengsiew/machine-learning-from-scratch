"""Functions for experimenting with different random forest parameter values"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .metrics import compute_accuracy, compute_f1, compute_precision, compute_recall
from .random_forest import random_forest_classifier


def get_stratified_k_fold_train_test_indices(X, y, k, random_seed):
    """Get indices for train and test instances for each fold, following stratified k-fold method.

    Args:
        X (np.ndarray): Attributes.
        y (np.ndarray): Class labels.
        k (int): No. of folds.
        random_seed (int): Random seed for reproducibility.

    Returns:
        k_fold_train_test_indices (list): Indices for train and test instances for each fold.
    """
    rng = np.random.default_rng(seed=random_seed)

    k_folds = [[] for fold in range(k)]

    for class_i in np.unique(y):
        class_i_indices = np.where(y == class_i)[0]
        rng.shuffle(class_i_indices)

        k_folds_for_class_i = np.array_split(class_i_indices, k)

        for fold_j, indices in enumerate(k_folds_for_class_i):
            k_folds[fold_j].extend(indices)

    k_fold_train_test_indices = []

    for fold_j in range(k):
        fold_j_test_indices = np.array(k_folds[fold_j], dtype=int)
        fold_j_train_indices = np.concatenate(
            k_folds[:fold_j] + k_folds[fold_j + 1:]).astype(int)

        k_fold_train_test_indices.append((fold_j_train_indices, fold_j_test_indices))

    return k_fold_train_test_indices


def k_fold_cross_validate_random_forest(
        X, y, k_fold_train_test_indices,
        ntree, minimum_size_for_split,
        attribute_names, attribute_types, pos_class):
    """Evaluate random forest with k-fold cross-validation.

    Args:
        X (np.ndarray): Attributes.
        y (np.ndarray): Class labels.
        k_fold_train_test_indices (list): Train-test indices for each fold.
        ntree (int): No. of trees in random forest.
        minimum_size_for_split (int): Minimum no. of instances required to split.
        attribute_names (list): Attribute names.
        attribute_types (list): Attribute types, either "numeric" or "categorical".
        pos_class (str): Positive class for precision, recall, and f1.

    Returns:
        k_fold_accuracies (list): Accuracy for each fold.
        k_fold_precisions (list): Precision for each fold.
        k_fold_recalls (list): Recall for each fold.
        k_fold_f1s (list): F1 for each fold.
    """
    k_fold_accuracies, k_fold_precisions = [], []
    k_fold_recalls, k_fold_f1s = [], []

    for fold_j_train_indices, fold_j_test_indices in k_fold_train_test_indices:
        X_train, X_test = X[fold_j_train_indices], X[fold_j_test_indices]
        y_train, y_test = y[fold_j_train_indices], y[fold_j_test_indices]

        random_forest = random_forest_classifier(
            ntree=ntree, split_criterion="information_gain",
            majority_class_threshold=1, minimum_size_for_split=minimum_size_for_split,
            minimum_split_quality_score=0, random_seed=0)

        random_forest.fit(
            X=X_train, y=y_train,
            attribute_names=attribute_names, attribute_types=attribute_types)

        y_pred = random_forest.predict(X=X_test)
        k_fold_accuracies.append(compute_accuracy(y_pred, y_test))
        k_fold_precisions.append(compute_precision(y_pred, y_test, pos_class))
        k_fold_recalls.append(compute_recall(y_pred, y_test, pos_class))
        k_fold_f1s.append(compute_f1(y_pred, y_test, pos_class))

    return k_fold_accuracies, k_fold_precisions, k_fold_recalls, k_fold_f1s


def random_forest_parameter_combination_experiment(
        X, y, k_fold_train_test_indices,
        ntree, minimum_size_for_split,
        attribute_names, attribute_types, pos_class):
    """Random forest experiment for one parameter combination.

    Args:
        X (np.ndarray): Attributes.
        y (np.ndarray): Class labels.
        k_fold_train_test_indices (list): Train-test indices for each fold.
        ntree (int): No. of trees in random forest.
        minimum_size_for_split (int): Minimum no. of instances required to split.
        attribute_names (list): Attribute names.
        attribute_types (list): Attribute types, either "numeric" or "categorical".
        pos_class (str): Positive class for precision, recall, and f1.

    Returns:
        result (dict): Parameter combination and mean/std of cross-validated metrics.
    """
    k_fold_accuracies, k_fold_precisions, k_fold_recalls, k_fold_f1s = (
        k_fold_cross_validate_random_forest(
            X=X, y=y, k_fold_train_test_indices=k_fold_train_test_indices,
            ntree=ntree, minimum_size_for_split=minimum_size_for_split,
            attribute_names=attribute_names, attribute_types=attribute_types,
            pos_class=pos_class))

    result = {
        "minimum_size_for_split": minimum_size_for_split, "ntree": ntree,
        "accuracy_mean": np.nanmean(a=k_fold_accuracies),
        "accuracy_std": np.nanstd(a=k_fold_accuracies),
        "precision_mean": np.nanmean(a=k_fold_precisions),
        "precision_std": np.nanstd(a=k_fold_precisions),
        "recall_mean": np.nanmean(a=k_fold_recalls),
        "recall_std": np.nanstd(a=k_fold_recalls),
        "f1_mean": np.nanmean(a=k_fold_f1s),
        "f1_std": np.nanstd(a=k_fold_f1s)}

    return result


def random_forest_parameter_grid_experiment(
        X, y, random_seed, parameter_grid,
        attribute_names, attribute_types, pos_class, n_jobs):
    """Random forest experiment for a parameter grid.

    Args:
        X (np.ndarray): Attributes.
        y (np.ndarray): Class labels.
        random_seed (int): Random seed for reproducibility.
        parameter_grid (dict): Parameter values to test.
        attribute_names (list): Attribute names.
        attribute_types (list): Attribute types, either "numeric" or "categorical".
        pos_class (str): Positive class for precision, recall, and f1.
        n_jobs (int): No. of parallel jobs.

    Returns:
        results (pd.DataFrame): Mean/std of cross-validated metrics for each parameter combination.
    """
    k_fold_train_test_indices = get_stratified_k_fold_train_test_indices(X=X, y=y, k=5, random_seed=random_seed)

    parameter_combinations = [
        {"ntree": ntree, "minimum_size_for_split": minimum_size_for_split}
        for ntree in parameter_grid["ntree"]
        for minimum_size_for_split in parameter_grid["minimum_size_for_split"]]

    results = Parallel(n_jobs=n_jobs)(
        delayed(random_forest_parameter_combination_experiment)(
            X=X, y=y, k_fold_train_test_indices=k_fold_train_test_indices,
            ntree=parameters["ntree"], minimum_size_for_split=parameters["minimum_size_for_split"],
            attribute_names=attribute_names, attribute_types=attribute_types,
            pos_class=pos_class)
        for parameters in parameter_combinations)

    results = pd.DataFrame(results)

    return results


def get_best_experiment_result(results, selection_metric):
    """Get experiment result with highest mean score for metric.

    Args:
        results (pd.DataFrame): Mean/std of cross-validated metrics for each parameter combination.
        selection_metric (str): Metric to select best experiment result.

    Returns:
        best_result (pd.Series): Experiment result with highest mean score.
    """
    best_result_index = results[f"{selection_metric}_mean"].idxmax()
    best_result = results.loc[best_result_index]

    return best_result


def save_experiment_results_to_csv(results, save_to):
    results.to_csv(save_to, index=False)


def plot_metric_against_parameter(results, parameter_name, metric_name, axis):
    
    parameter_values = results[parameter_name]
    metric_means = results[f"{metric_name}_mean"]
    metric_stds = results[f"{metric_name}_std"]
    best_metric_index = metric_means.idxmax()

    axis.errorbar(
        parameter_values, metric_means, yerr=metric_stds,
        marker="o", capsize=4)
    
    axis.errorbar(
        [parameter_values.loc[best_metric_index]],
        [metric_means.loc[best_metric_index]],
        yerr=[metric_stds.loc[best_metric_index]],
        marker="o", color="red", capsize=4)
    
    axis.set(xlabel=parameter_name, ylabel=f"Cross-validated {metric_name}")
    
    axis.grid(True)


def plot_minimum_size_for_split_experiment_results(results, dataset_name, save_to):
    
    fig, axis = plt.subplots(figsize=(6, 4))

    plot_metric_against_parameter(
        results=results, parameter_name="minimum_size_for_split",
        metric_name="f1", axis=axis)

    fig.suptitle(f"F1 against minimum_size_for_split for {dataset_name} dataset")
    
    fig.tight_layout()
    fig.savefig(save_to, bbox_inches="tight")
    plt.close(fig)


def plot_ntrees_experiment_results(results, dataset_name, save_to):
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    metric_names = ["accuracy", "precision", "recall", "f1"]

    for axis, metric_name in zip(axes.ravel(), metric_names):
        plot_metric_against_parameter(
            results=results, parameter_name="ntree",
            metric_name=metric_name, axis=axis)

    fig.suptitle(f"Metrics against ntree for {dataset_name} dataset")

    fig.tight_layout()
    fig.savefig(save_to, bbox_inches="tight")
    plt.close(fig)
