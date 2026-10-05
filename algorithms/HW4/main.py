"""Main workflow for HW4 neural network experiments"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experimenters import (
    create_experiment_results_table,
    get_best_experiment_result,
    neural_network_parameter_grid_experiment,
    plot_history,
    save_experiment_results_to_csv,
)
from loaders import load_dataset, split_attributes_and_labels
from neural_network import NeuralNetwork, initialize_thetas
from preprocessors import fit_preprocessor, preprocess_attributes


def main():

    ### 1: parse arguments passed from run.sh ###

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=False)
    args = parser.parse_args()

    ### 2: define constants ###

    step_size = 0.3
    num_iterations = 20
    batch_size = 32
    random_seed = 0
    n_jobs = -1

    ### 3: load dataset and split into attributtes + labels ###

    data, column_names = load_dataset(dataset_path=args.dataset_path)
    X, y, attribute_names, attribute_types = split_attributes_and_labels(
        data=data, column_names=column_names)
    y = y.astype(float)

    parameter_grid = {
        "num_hidden_layers": [1, 3, 5],
        "num_neurons_per_hidden_layer": [4, 8, 16],
        "regularization_strength": [0.01, 0.05],
    }

    ### 4: experiment with parameter combinations and choose the best ###

    results = neural_network_parameter_grid_experiment(
        X=X,
        y=y,
        attribute_types=attribute_types,
        random_seed=random_seed,
        parameter_grid=parameter_grid,
        step_size=step_size,
        num_iterations=num_iterations,
        batch_size=batch_size,
        n_jobs=n_jobs)

    ### 5: save experiment results and figures ###

    table = create_experiment_results_table(
        results=results,
        dataset_name=args.dataset_path.stem)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    full_results_path = args.results_dir / f"{args.dataset_path.stem}_experiment_results_full.csv"
    table_path = args.results_dir / f"{args.dataset_path.stem}_experiment_results_table.csv"
    latex_table_path = args.results_dir / f"{args.dataset_path.stem}_experiment_results_table.tex"

    save_experiment_results_to_csv(results=results, save_to=full_results_path)
    print(f"Wrote {full_results_path}")

    save_experiment_results_to_csv(results=table, save_to=table_path)
    print(f"Wrote {table_path}")

    table.to_latex(latex_table_path, index=False, float_format="%.4f")
    print(f"Wrote {latex_table_path}")

    ### 6: select best hyperparameters from experiment ###

    best_result = get_best_experiment_result(results=results, selection_metric="f1")

    best_num_hidden_layers = int(best_result["num_hidden_layers"])
    best_num_neurons_per_hidden_layer = int(best_result["num_neurons_per_hidden_layer"])
    best_regularization_strength = float(best_result["regularization_strength"])

    ### 7: train-test split, preprocessing ###

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y)

    preprocessor = fit_preprocessor(X_train=X_train_raw, attribute_types=attribute_types)
    X_train = preprocess_attributes(preprocessor=preprocessor, X=X_train_raw)
    X_test = preprocess_attributes(preprocessor=preprocessor, X=X_test_raw)

    X_train = [row for row in X_train]
    y_train = [np.array([label]) for label in y_train_raw]
    X_test = [row for row in X_test]
    y_test = [np.array([label]) for label in y_test_raw]

    ### 8: record and plot training history ###

    thetas = initialize_thetas(
        num_input_neurons=X_train[0].shape[0],
        num_hidden_layers=best_num_hidden_layers,
        num_neurons_per_hidden_layer=best_num_neurons_per_hidden_layer,
        num_output_neurons=1,
        random_seed=random_seed)

    network = NeuralNetwork(thetas)
    network.configure_for_fit(
        regularization_strength=best_regularization_strength,
        step_size=step_size)

    history = network.fit(
        X_train, y_train,
        num_iterations=num_iterations,
        batch_size=batch_size,
        random_seed=random_seed,
        shuffle=True,
        verbose=False,
        record_history=True,
        X_test=X_test,
        y_test=y_test)

    history = pd.DataFrame(history)

    history_path = args.results_dir / f"{args.dataset_path.stem}_history.csv"
    save_experiment_results_to_csv(results=history, save_to=history_path)
    print(f"Wrote {history_path}")

    if args.figures_dir is not None:
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        history_figure_path = args.figures_dir / f"{args.dataset_path.stem}_history.png"
        fig, axes = plot_history(history)
        fig.savefig(history_figure_path, bbox_inches="tight")
        print(f"Wrote {history_figure_path}")


if __name__ == "__main__":
    main()
