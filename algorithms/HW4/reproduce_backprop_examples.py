"""Main workflow for reproducing backpropagation examples 1 and 2"""

from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
from neural_network import NeuralNetwork

def reproduce_backprop_example(thetas, X_train, y_train, regularization_strength):
    network = NeuralNetwork(thetas)
    network.configure_for_fit(regularization_strength=regularization_strength, step_size=0.0)
    network.fit(X_train, y_train, num_iterations=1, batch_size=len(X_train),
                random_seed=0, shuffle=False, verbose=True, record_history=False)


def reproduce_backprop_example1():
    thetas = [np.array([[0.40000, 0.10000],
                        [0.30000, 0.20000]]),
              np.array([[0.70000, 0.50000, 0.60000]])]
    X_train = [np.array([0.13000]), np.array([0.42000])]
    y_train = [np.array([0.90000]), np.array([0.23000])]
    regularization_strength = 0.0

    reproduce_backprop_example(thetas, X_train, y_train, regularization_strength)


def reproduce_backprop_example2():
    thetas = [
        np.array([[0.42000, 0.15000, 0.40000],
                  [0.72000, 0.10000, 0.54000],
                  [0.01000, 0.19000, 0.42000],
                  [0.30000, 0.35000, 0.68000]]),
        np.array([[0.21000, 0.67000, 0.14000, 0.96000, 0.87000],
                  [0.87000, 0.42000, 0.20000, 0.32000, 0.89000],
                  [0.03000, 0.56000, 0.80000, 0.69000, 0.09000]]),
        np.array([[0.04000, 0.87000, 0.42000, 0.53000],
                  [0.17000, 0.10000, 0.95000, 0.69000]]),
    ]
    X_train = [np.array([0.32000, 0.68000]),
               np.array([0.83000, 0.02000])]
    y_train = [np.array([0.75000, 0.98000]),
               np.array([0.75000, 0.28000])]
    regularization_strength = 0.25

    reproduce_backprop_example(thetas, X_train, y_train, regularization_strength)


def main(output_dir="backprop_examples/reproduced"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    example1_output_path = output_dir / "backprop_example1_reproduction.txt"
    example2_output_path = output_dir / "backprop_example2_reproduction.txt"

    with example1_output_path.open("w") as output_file:
        with redirect_stdout(output_file):
            reproduce_backprop_example1()

    with example2_output_path.open("w") as output_file:
        with redirect_stdout(output_file):
            reproduce_backprop_example2()

    print(f"Wrote {example1_output_path}")
    print(f"Wrote {example2_output_path}")


if __name__ == "__main__":
    main()
