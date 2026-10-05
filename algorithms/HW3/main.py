"""Main workflow for experimenting with different ntree values and plotting experiment results for a single dataset"""

import argparse
from pathlib import Path

from .experiment import (
    get_best_experiment_result,
    plot_minimum_size_for_split_experiment_results,
    plot_ntrees_experiment_results,
    random_forest_parameter_grid_experiment,
    save_experiment_results_to_csv,
)
from .loaders import load_dataset, split_attributes_and_labels


def main():

    ### 1: parse arguments passed from run.sh ###
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    args = parser.parse_args()

    ### 2: load dataset and split into attributtes + labels ###
    data, column_names = load_dataset(dataset_path=args.dataset_path)
    X, y, attribute_names, attribute_types = split_attributes_and_labels(data=data, column_names=column_names)

    ### 3: define constants ###

    pos_class = "1"
    n_jobs = -1
    random_seed = 0

    ### 4: experiment with minimum_size_for_split and choose the best ###

    minimum_size_for_split_experiment_grid = {"minimum_size_for_split": [10, 20, 30], "ntree": [25]}

    minimum_size_for_split_experiment_results = (
        random_forest_parameter_grid_experiment(
            X=X, y=y, random_seed=random_seed,
            parameter_grid=minimum_size_for_split_experiment_grid,
            attribute_names=attribute_names, attribute_types=attribute_types,
            pos_class=pos_class, n_jobs=n_jobs))

    best_minimum_size_for_split_experiment_result = get_best_experiment_result(
        results=minimum_size_for_split_experiment_results, selection_metric="f1")
    
    best_minimum_size_for_split = best_minimum_size_for_split_experiment_result["minimum_size_for_split"]

    ### 5: experiment with different ntree values ###

    ntrees_experiment_grid = {"minimum_size_for_split": [best_minimum_size_for_split], "ntree": [1, 5, 10, 20, 30, 40, 50]}

    ntrees_experiment_results = random_forest_parameter_grid_experiment(
        X=X, y=y, random_seed=random_seed,
        parameter_grid=ntrees_experiment_grid,
        attribute_names=attribute_names, attribute_types=attribute_types,
        pos_class=pos_class, n_jobs=n_jobs)

    ### 6: specify output paths for artifacts ###

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    minimum_size_for_split_experiment_results_path = args.results_dir / "minimum_size_for_split_experiment_results.csv"
    
    ntrees_experiment_results_path = args.results_dir / "ntrees_experiment_results.csv"
    
    minimum_size_for_split_figure_output_path = args.figures_dir / "minimum_size_for_split_experiment_results.png"
    
    ntrees_figure_output_path = args.figures_dir / "ntrees_experiment_results.png"

    ### 7: save the artifacts to output path ###

    save_experiment_results_to_csv(
        results=minimum_size_for_split_experiment_results,
        save_to=minimum_size_for_split_experiment_results_path)

    save_experiment_results_to_csv(
        results=ntrees_experiment_results, save_to=ntrees_experiment_results_path)

    plot_minimum_size_for_split_experiment_results(
        results=minimum_size_for_split_experiment_results, dataset_name=args.dataset_path.stem,
        save_to=minimum_size_for_split_figure_output_path)

    plot_ntrees_experiment_results(
        results=ntrees_experiment_results, dataset_name=args.dataset_path.stem,
        save_to=ntrees_figure_output_path)

if __name__ == "__main__":
    main()
