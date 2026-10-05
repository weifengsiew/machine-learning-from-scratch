"""Functions for neural network experiments"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from neural_network import (
    NeuralNetwork,
    compute_average_regularized_cost_over_instances,
    compute_metrics_over_instances,
    initialize_thetas,
)


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


def k_fold_cross_validate_neural_network(
        X, y, attribute_types, k_fold_train_test_indices,
        num_hidden_layers, num_neurons_per_hidden_layer,
        regularization_strength, step_size,
        num_iterations, batch_size, random_seed):
    """Evaluate neural network with k-fold cross-validation.

    Args:
        X (np.ndarray): Attributes.
        y (np.ndarray): Class labels.
        attribute_types (list): Attribute types.
        k_fold_train_test_indices (list): Train-test indices for each fold.
        num_hidden_layers (int): Number of hidden layers.
        num_neurons_per_hidden_layer (int): Number of neurons in each hidden layer.
        regularization_strength (float): Strength of L2 regularization for non-bias weights.
        step_size (float): Step size used when updating weights.
        num_iterations (int): Number of times to process training instances.
        batch_size (int): Number of training instances used for each weight update.
        random_seed (int): Random seed for reproducibility.

    Returns:
        k_fold_costs (list): Test cost for each fold.
        k_fold_accuracies (list): Test accuracy for each fold.
        k_fold_f1s (list): Test F1 for each fold.
    """
    k_fold_costs = []
    k_fold_accuracies = []
    k_fold_f1s = []

    from preprocessors import fit_preprocessor, preprocess_attributes

    for fold_number, (train_indices, test_indices) in enumerate(k_fold_train_test_indices):
        X_train_raw = X[train_indices]
        X_test_raw = X[test_indices]
        preprocessor = fit_preprocessor(X_train_raw, attribute_types)
        X_train = preprocess_attributes(preprocessor, X_train_raw)
        X_test = preprocess_attributes(preprocessor, X_test_raw)

        X_train = [row for row in X_train]
        y_train = [np.array([label]) for label in y[train_indices]]
        X_test = [row for row in X_test]
        y_test = [np.array([label]) for label in y[test_indices]]

        fold_random_seed = random_seed + fold_number
        thetas = initialize_thetas(
            num_input_neurons=X_train[0].shape[0],
            num_hidden_layers=num_hidden_layers,
            num_neurons_per_hidden_layer=num_neurons_per_hidden_layer,
            num_output_neurons=1,
            random_seed=fold_random_seed)

        network = NeuralNetwork(thetas)
        network.configure_for_fit(
            regularization_strength=regularization_strength,
            step_size=step_size)

        network.fit(
            X_train, y_train,
            num_iterations=num_iterations,
            batch_size=batch_size,
            random_seed=fold_random_seed,
            shuffle=True,
            verbose=False,
            record_history=False,
            X_test=X_test,
            y_test=y_test)

        test_cost = compute_average_regularized_cost_over_instances(network, X_test, y_test, regularization_strength)
        test_accuracy, test_f1 = compute_metrics_over_instances(network, X_test, y_test, pos_class=1)

        k_fold_costs.append(test_cost)
        k_fold_accuracies.append(test_accuracy)
        k_fold_f1s.append(test_f1)

    return k_fold_costs, k_fold_accuracies, k_fold_f1s


def neural_network_parameter_combination_experiment(
        X, y, attribute_types, k_fold_train_test_indices,
        num_hidden_layers, num_neurons_per_hidden_layer,
        regularization_strength, step_size,
        num_iterations, batch_size, random_seed):
    """Run neural network experiment for one parameter combination.

    Args:
        X (np.ndarray): Attributes.
        y (np.ndarray): Class labels.
        attribute_types (list): Attribute types.
        k_fold_train_test_indices (list): Train-test indices for each fold.
        num_hidden_layers (int): Number of hidden layers.
        num_neurons_per_hidden_layer (int): Number of neurons in each hidden layer.
        regularization_strength (float): Strength of L2 regularization for non-bias weights.
        step_size (float): Step size used when updating weights.
        num_iterations (int): Number of times to process training instances.
        batch_size (int): Number of training instances used for each weight update.
        random_seed (int): Random seed for reproducibility.

    Returns:
        result (dict): Parameter combination and mean/std of cross-validated metrics.
    """
    k_fold_costs, k_fold_accuracies, k_fold_f1s = (
        k_fold_cross_validate_neural_network(
            X=X, y=y, attribute_types=attribute_types,
            k_fold_train_test_indices=k_fold_train_test_indices,
            num_hidden_layers=num_hidden_layers,
            num_neurons_per_hidden_layer=num_neurons_per_hidden_layer,
            regularization_strength=regularization_strength,
            step_size=step_size,
            num_iterations=num_iterations,
            batch_size=batch_size,
            random_seed=random_seed))

    result = {
        "num_hidden_layers": num_hidden_layers,
        "num_neurons_per_hidden_layer": num_neurons_per_hidden_layer,
        "regularization_strength": regularization_strength,
        "cost_mean": np.nanmean(a=k_fold_costs),
        "cost_std": np.nanstd(a=k_fold_costs),
        "accuracy_mean": np.nanmean(a=k_fold_accuracies),
        "accuracy_std": np.nanstd(a=k_fold_accuracies),
        "f1_mean": np.nanmean(a=k_fold_f1s),
        "f1_std": np.nanstd(a=k_fold_f1s)}

    return result


def neural_network_parameter_grid_experiment(
        X, y, attribute_types, random_seed, parameter_grid,
        step_size, num_iterations, batch_size, n_jobs):
    """Run neural network experiment for a parameter grid.

    Args:
        X (np.ndarray): Attributes.
        y (np.ndarray): Class labels.
        attribute_types (list): Attribute types.
        random_seed (int): Random seed for reproducibility.
        parameter_grid (dict): Parameter values to test.
        step_size (float): Step size used when updating weights.
        num_iterations (int): Number of times to process training instances.
        batch_size (int): Number of training instances used for each weight update.
        n_jobs (int): No. of parallel jobs.

    Returns:
        results (pd.DataFrame): Mean/std of cross-validated metrics for each parameter combination.
    """
    k_fold_train_test_indices = get_stratified_k_fold_train_test_indices(
        X=X, y=y, k=5, random_seed=random_seed)

    parameter_combinations = []

    for num_hidden_layers in parameter_grid["num_hidden_layers"]:
        for num_neurons_per_hidden_layer in parameter_grid["num_neurons_per_hidden_layer"]:
            for regularization_strength in parameter_grid["regularization_strength"]:
                parameters = {
                    "num_hidden_layers": num_hidden_layers,
                    "num_neurons_per_hidden_layer": num_neurons_per_hidden_layer,
                    "regularization_strength": regularization_strength}

                parameter_combinations.append(parameters)

    results = Parallel(n_jobs=n_jobs)(
        delayed(neural_network_parameter_combination_experiment)(
            X=X, y=y, attribute_types=attribute_types,
            k_fold_train_test_indices=k_fold_train_test_indices,
            num_hidden_layers=parameters["num_hidden_layers"],
            num_neurons_per_hidden_layer=parameters["num_neurons_per_hidden_layer"],
            regularization_strength=parameters["regularization_strength"],
            step_size=step_size,
            num_iterations=num_iterations,
            batch_size=batch_size,
            random_seed=random_seed)
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


def create_experiment_results_table(results, dataset_name):
    """Create experiment results table.

    Args:
        results (pd.DataFrame): Mean/std of cross-validated metrics for each parameter combination.
        dataset_name (str): Dataset name.

    Returns:
        table (pd.DataFrame): Experiment results table for report.
    """
    table = results.copy()
    table.insert(0, "dataset", dataset_name)

    table = table[[
        "dataset",
        "num_hidden_layers",
        "num_neurons_per_hidden_layer",
        "regularization_strength",
        "accuracy_mean",
        "f1_mean"]]

    table = table.rename(columns={
        "dataset": "Dataset",
        "num_hidden_layers": "Hidden Layers",
        "num_neurons_per_hidden_layer": "Neurons Per Hidden Layer",
        "regularization_strength": "Lambda",
        "accuracy_mean": "Mean Accuracy",
        "f1_mean": "Mean F1-score"})
    table = table.sort_values(by="Mean F1-score", ascending=False)

    return table


def save_experiment_results_to_csv(results, save_to):
    """Save experiment results to CSV.

    Args:
        results (pd.DataFrame): Experiment results.
        save_to (str): Path to save experiment results.
    """
    results.to_csv(save_to, index=False)


def plot_history(history):
    """Plot cost, accuracy, and F1 against number of training instances seen.

    Args:
        history (pd.DataFrame): Training history.

    Returns:
        fig (matplotlib.figure.Figure): Figure.
        axes (np.ndarray): Plot axes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    plots = [
        ("train_cost", "test_cost", "Cost", "Regularized cost"),
        ("train_accuracy", "test_accuracy", "Accuracy", "Accuracy"),
        ("train_f1", "test_f1", "F1 Score", "F1 score")]

    for axis, (train_metric, test_metric, title, ylabel) in zip(axes, plots):
        axis.plot(
            history["train_instances_seen"], history[train_metric],
            marker="o", label="Train")
        axis.plot(
            history["train_instances_seen"], history[test_metric],
            marker="o", label="Test")
        axis.set_title(title)
        axis.set_xlabel("Training instances seen")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()

    plt.tight_layout()

    return fig, axes
