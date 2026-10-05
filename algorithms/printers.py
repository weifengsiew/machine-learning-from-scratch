"""Console-formatting helpers for verbose neural-network training output."""

from __future__ import annotations

import numpy as np


def format_vector(values: np.ndarray) -> str:
    """Format a numeric vector for readable console output.

    Args:
        values: Numeric values to format.

    Returns:
        Space-separated values formatted to five decimal places.
    """
    return "  ".join(f"{value:.5f}" for value in values)


def print_initial_network(thetas: list[np.ndarray], regularization_strength: float) -> None:
    """Print network shape, regularization, and initial weights.

    Args:
        thetas: Weight matrices for each network layer.
        regularization_strength: L2 regularization strength to display.
    """
    num_neurons_per_layer = [thetas[0].shape[1] - 1]

    for theta in thetas:
        num_neurons_per_layer.append(theta.shape[0])

    num_neurons_per_layer_text = " ".join(str(num_neurons) for num_neurons in num_neurons_per_layer)

    print(f"Regularization parameter lambda={regularization_strength:.3f}\n")
    print(f"Initializing the network with the following structure (number of neurons per layer): [{num_neurons_per_layer_text}]\n")

    for theta_number, theta in enumerate(thetas, start=1):
        print(f"Initial Theta{theta_number} (the weights of each neuron, including the bias weight, are stored in the rows):")

        for row in theta:
            print(f"\t{format_vector(row)}  ")
        print()


def print_training_set(X_train: np.ndarray, y_train: np.ndarray) -> None:
    """Print training feature vectors and target vectors.

    Args:
        X_train: Training feature matrix.
        y_train: Training target matrix.
    """
    print("Training set")

    for instance_number, (x, y) in enumerate(zip(X_train, y_train), start=1):
        print(f"\tTraining instance {instance_number}")
        print(f"\t\tx: [{format_vector(x)}]")
        print(f"\t\ty: [{format_vector(y)}]")

    print()


def print_iteration_header(iteration: int) -> None:
    """Print a heading for one training iteration.

    Args:
        iteration: One-based iteration number.
    """
    iteration_names = {
        1: "First",
        2: "Second",
        3: "Third",
    }
    iteration_name = iteration_names.get(iteration, f"Iteration {iteration}")

    if iteration <= 3:
        iteration_text = f"{iteration_name} iteration"
    else:
        iteration_text = iteration_name

    print("==============================")
    print(iteration_text)
    print("==============================")


def print_cost_section_header() -> None:
    """Print the heading for the cost-calculation section."""
    print("--------------------------------------------")
    print("Computing the error/cost, J, of the network")


def print_forward_propagation_output(
        instance_number: int, x: np.ndarray, y: np.ndarray,
        layers_preactivations: list[np.ndarray],
        layers_activations: list[np.ndarray], y_pred: np.ndarray,
        cost: float) -> None:
    """Print forward-propagation values for one training instance.

    Args:
        instance_number: One-based training-instance number.
        x: Input feature vector.
        y: Expected target vector.
        layers_preactivations: Weighted sums for network layers.
        layers_activations: Activations for network layers.
        y_pred: Predicted output vector.
        cost: Instance cost.
    """
    print(f"\tProcessing training instance {instance_number}")
    print(f"\tForward propagating the input [{format_vector(x)}]")

    for layer_number, activation in enumerate(layers_activations, start=1):
        if layer_number > 1:
            preactivation = layers_preactivations[layer_number - 2]
            print(f"\t\tz{layer_number}: [{format_vector(preactivation)}]")

        print(f"\t\ta{layer_number}: [{format_vector(activation)}]\n")

    print(f"\t\tf(x): [{format_vector(y_pred)}]")
    print(f"\tPredicted output for instance {instance_number}: [{format_vector(y_pred)}]")
    print(f"\tExpected output for instance {instance_number}: [{format_vector(y)}]")
    print(f"\tCost, J, associated with instance {instance_number}: {cost:.3f}\n")


def print_final_cost(final_cost: float) -> None:
    """Print the final regularized training cost.

    Args:
        final_cost: Regularized cost after processing the training set.
    """
    print(f"Final (regularized) cost, J, based on the complete training set: {final_cost:.5f}\n")


def print_backpropagation_section_header() -> None:
    """Print the heading for the backpropagation section."""
    print("\n--------------------------------------------")
    print("Running backpropagation")


def print_backpropagation_output(
        instance_number: int, deltas: list[np.ndarray],
        gradients: list[np.ndarray]) -> None:
    """Print deltas and gradients for one training instance.

    Args:
        instance_number: One-based training-instance number.
        deltas: Backpropagated error vectors.
        gradients: Weight gradients for each layer.
    """
    print(f"\tComputing gradients based on training instance {instance_number}")

    for layer_number, delta in reversed(list(enumerate(deltas, start=2))):
        print(f"\t\tdelta{layer_number}: [{format_vector(delta)}]")

    print()

    for theta_number, gradient in reversed(list(enumerate(gradients, start=1))):
        print(f"\t\tGradients of Theta{theta_number} based on training instance {instance_number}:")

        for row in gradient:
            print(f"\t\t\t{format_vector(row)}  ")

        print()


def print_final_gradients(final_gradients: list[np.ndarray]) -> None:
    """Print averaged regularized gradients.

    Args:
        final_gradients: Final gradient matrix for each network layer.
    """
    print("\tThe entire training set has been processed. Computing the average (regularized) gradients:")

    for theta_number, final_gradient in enumerate(final_gradients, start=1):
        print(f"\t\tFinal regularized gradients of Theta{theta_number}:")

        for row in final_gradient:
            print(f"\t\t\t{format_vector(row)}  ")

        print()


def print_updated_thetas(thetas: list[np.ndarray], iteration: int) -> None:
    """Print weights after one training iteration.

    Args:
        thetas: Updated weight matrices.
        iteration: One-based iteration number.
    """
    print("--------------------------------------------")
    print(f"\tUpdated theta values after iteration {iteration}:")

    for theta_number, theta in enumerate(thetas, start=1):
        print(f"\t\tUpdated Theta{theta_number}:")

        for row in theta:
            print(f"\t\t\t{format_vector(row)}  ")

        print()


def print_output_paths(example1_output_path: str, example2_output_path: str) -> None:
    """Print paths where verbose example output was written.

    Args:
        example1_output_path: Path for the first example output.
        example2_output_path: Path for the second example output.
    """
    print(f"Wrote {example1_output_path}")
    print(f"Wrote {example2_output_path}")
