from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from src.decision_tree_implementation import decision_tree_classifier


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    return parser.parse_args()


def load_car_data(data_dir):
    data = np.genfromtxt(data_dir / "car.csv", delimiter=",", dtype=str, skip_header=1)
    return data


def run_decision_tree_experiment(data, split_criterion, majority_class_threshold):
    train_accuracies = []
    test_accuracies = []

    # 100 estimates of train and test accuracy
    for random_state in range(100):
        # shuffle the data
        data_shuffled = shuffle(data, random_state=random_state)

        # split data into features and labels
        X = data_shuffled[:, :-1]
        y = data_shuffled[:, -1]

        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state)

        # fit model
        classifier = decision_tree_classifier(
            split_criterion=split_criterion,
            majority_class_threshold=majority_class_threshold)
        
        classifier.fit(X_train, y_train)

        # predict on train and store accuracy
        y_pred = classifier.predict(X_train)
        train_accuracy = np.mean(y_pred == y_train)
        train_accuracies.append(train_accuracy)

        # predict on test and store accuracy
        y_pred = classifier.predict(X_test)
        test_accuracy = np.mean(y_pred == y_test)
        test_accuracies.append(test_accuracy)

    return train_accuracies, test_accuracies


def plot_accuracy_histogram(accuracies, dataset_name, filename):
    accuracy_mean = np.mean(accuracies)
    accuracy_std = np.std(accuracies)
    x_axis_lower_lim = min(0.6, min(accuracies))

    plt.figure(figsize=(8, 5))
    plt.hist(accuracies, bins=15, edgecolor="black")
    plt.xlim(x_axis_lower_lim, 1)
    plt.title(f"{dataset_name} Accuracy (mean={accuracy_mean:.3f}, sd={accuracy_std:.3f})")
    plt.xlabel("(Accuracy)")
    plt.ylabel(f"(Accuracy Frequency on {dataset_name})")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    data = load_car_data(args.data_dir)

    exp1_train_accuracies, exp1_test_accuracies = run_decision_tree_experiment(
        data=data,
        split_criterion="information_gain",
        majority_class_threshold=1.0)

    plot_accuracy_histogram(
        accuracies=exp1_train_accuracies,
        dataset_name="Training Data",
        filename=args.figures_dir / "Question2.1.pdf")

    plot_accuracy_histogram(
        accuracies=exp1_test_accuracies,
        dataset_name="Testing Data",
        filename=args.figures_dir / "Question2.2.pdf")
    
    exp2_train_accuracies, exp2_test_accuracies = run_decision_tree_experiment(
        data=data,
        split_criterion="gini",
        majority_class_threshold=1.0)

    plot_accuracy_histogram(
        accuracies=exp2_train_accuracies,
        dataset_name="Training Data",
        filename=args.figures_dir / "QuestionE.1_train.pdf")

    plot_accuracy_histogram(
        accuracies=exp2_test_accuracies,
        dataset_name="Testing Data",
        filename=args.figures_dir / "QuestionE.1_test.pdf")
    
    exp3_train_accuracies, exp3_test_accuracies = run_decision_tree_experiment(
        data=data,
        split_criterion="information_gain",
        majority_class_threshold=0.85)

    plot_accuracy_histogram(
        accuracies=exp3_train_accuracies,
        dataset_name="Training Data",
        filename=args.figures_dir / "QuestionE.2_train.pdf")

    plot_accuracy_histogram(
        accuracies=exp3_test_accuracies,
        dataset_name="Testing Data",
        filename=args.figures_dir / "QuestionE.2_test.pdf")


if __name__ == "__main__":
    main()
