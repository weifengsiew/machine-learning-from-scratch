from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from src.knn_implementation import KNN_classifier


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    return parser.parse_args()


def load_wdbc_data(data_dir):
    data = np.loadtxt(data_dir / "wdbc.csv", delimiter=",")
    return data


def knn_accuracies_for_random_split(data, normalize_features, random_state):
    # shuffle the data
    data_shuffled = shuffle(data, random_state=random_state)

    # split data into features and labels
    X = data_shuffled[:, :-1]
    y = data_shuffled[:, -1]

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    if normalize_features:
        # normalize features using z-score transformation
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    train_accuracies_for_split = {}
    test_accuracies_for_split = {}

    # odd values of k from 1 to 51
    for k in range(1, 52, 2):
        knn_classifier = KNN_classifier(k=k)

        # fit model
        knn_classifier.fit(X_train, y_train)

        # predict on train and store accuracy
        y_pred = knn_classifier.predict(X_train)
        train_accuracy = np.mean(y_pred == y_train)
        train_accuracies_for_split[k] = train_accuracy

        # predict on test and store accuracy
        y_pred = knn_classifier.predict(X_test)
        test_accuracy = np.mean(y_pred == y_test)
        test_accuracies_for_split[k] = test_accuracy

    return train_accuracies_for_split, test_accuracies_for_split


def run_knn_experiment(data, normalize_features):
    train_accuracies = {}
    test_accuracies = {}

    # odd values of k from 1 to 51
    for k in range(1, 52, 2):
        train_accuracies[k] = []
        test_accuracies[k] = []

    # 20 estimates of train and test accuracy for each k
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(
            knn_accuracies_for_random_split,
            repeat(data),
            repeat(normalize_features),
            range(20)))

    for train_accuracies_for_split, test_accuracies_for_split in results:
        for k in range(1, 52, 2):
            train_accuracies[k].append(train_accuracies_for_split[k])
            test_accuracies[k].append(test_accuracies_for_split[k])

    return train_accuracies, test_accuracies


def plot_accuracies(accuracies, label, y_axis_label, color, filename):
    accuracy_means = [np.mean(accuracies[k]) for k in accuracies]
    accuracy_stds = [np.std(accuracies[k]) for k in accuracies]

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        list(accuracies.keys()),
        accuracy_means,
        yerr=accuracy_stds,
        label=label,
        fmt="o",
        capsize=5,
        color=color,
    )
    plt.plot(list(accuracies.keys()), accuracy_means, color=color)
    plt.xlabel("(Value of k)")
    plt.ylabel(y_axis_label)
    plt.legend()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_train_and_test_accuracies_overlaid(train_accuracies, test_accuracies, filename):
    train_means = [np.mean(train_accuracies[k]) for k in train_accuracies]
    train_stds = [np.std(train_accuracies[k]) for k in train_accuracies]
    test_means = [np.mean(test_accuracies[k]) for k in test_accuracies]
    test_stds = [np.std(test_accuracies[k]) for k in test_accuracies]

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        list(train_accuracies.keys()),
        train_means,
        yerr=train_stds,
        label="Train Accuracy",
        fmt="o",
        capsize=5,
        color="blue")
    plt.plot(list(train_accuracies.keys()), train_means, color="blue")
    plt.errorbar(
        list(test_accuracies.keys()),
        test_means,
        yerr=test_stds,
        label="Test Accuracy",
        fmt="o",
        capsize=5,
        color="orange")
    plt.plot(list(test_accuracies.keys()), test_means, color="orange")
    plt.xlabel("(Value of k)")
    plt.ylabel("(Accuracy)")
    plt.legend()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    data = load_wdbc_data(args.data_dir)

    exp1_train_accuracies, exp1_test_accuracies = run_knn_experiment(
        data=data, normalize_features=True)

    plot_accuracies(
        accuracies=exp1_train_accuracies,
        label="Train Accuracy",
        y_axis_label="(Accuracy over training data)",
        color="blue",
        filename=args.figures_dir / "Question1.1.pdf")

    plot_accuracies(
        accuracies=exp1_test_accuracies,
        label="Test Accuracy",
        y_axis_label="(Accuracy over testing data)",
        color="orange",
        filename=args.figures_dir / "Question1.2.pdf")

    plot_train_and_test_accuracies_overlaid(
        train_accuracies=exp1_train_accuracies,
        test_accuracies=exp1_test_accuracies,
        filename=args.figures_dir / "Question1.4.pdf")

    exp2_train_accuracies, exp2_test_accuracies = run_knn_experiment(
        data=data, normalize_features=False)

    plot_accuracies(
        accuracies=exp2_test_accuracies,
        label="Test Accuracy",
        y_axis_label="(Accuracy over testing data)",
        color="orange",
        filename=args.figures_dir / "Question1.6.pdf")


if __name__ == "__main__":
    main()
