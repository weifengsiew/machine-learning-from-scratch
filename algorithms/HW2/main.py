"""Main workflow to reproduce experiments and corresponding figures"""

import random

from src.experiments import (plot_alpha_search_results, plot_experiment_results,
                             run_experiment,search_for_best_alpha)


def main():

    # for reproducibility
    random.seed(42)

    pos_class = 'positive'
    neg_class = 'negative'

    # Question 1
    # use 20% of train and 20% of test set
    # no additive smoothing (so alpha = 0) 
    # probabilities are not log-scaled
    # fit on train, predict on test, return accuracy, precision, recall, confusion matrix

    accuracy, precision, recall, confusion_matrix = run_experiment(
        percentage_positives_train=0.2, percentage_negatives_train=0.2,
        percentage_positives_test=0.2, percentage_negatives_test=0.2,
        alpha=0, log_scale=False,
        pos_class=pos_class, neg_class=neg_class)

    plot_experiment_results(
        accuracy, precision, recall, confusion_matrix,
        pos_class, neg_class, save_to_path="figures/Question1.pdf")

    print("Question 1 done.")

    # Question 2
    # use 20% of train and 20% of test set
    # Additive smoothing with alpha = 1 
    # log-scaled probabilities
    # fit on train, predict on test, return accuracy, precision, recall, confusion matrix

    accuracy, precision, recall, confusion_matrix = run_experiment(
        percentage_positives_train=0.2, percentage_negatives_train=0.2,
        percentage_positives_test=0.2, percentage_negatives_test=0.2,
        alpha=1, log_scale=True,
        pos_class=pos_class, neg_class=neg_class)
    
    plot_experiment_results(
        accuracy, precision, recall, confusion_matrix,
        pos_class, neg_class, save_to_path="figures/Question2a.pdf")

    # Question 2
    # use 20% of train and 20% of test set
    # Additive smoothing with various alpha values
    # log-scaled probabilities
    # for each alpha, fit on train, predict on test, calculate accuracy
    # return accuracy for each alpha, and best alpha with max accuracy
    # plot accuracy against alpha

    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    alphas, accuracies, best_alpha = search_for_best_alpha(
        alphas=alphas,
        percentage_positives_train=0.2, percentage_negatives_train=0.2,
        percentage_positives_test=0.2, percentage_negatives_test=0.2,
        log_scale=True, n_jobs=-1)
    
    plot_alpha_search_results(
        alphas, accuracies, save_to_path="figures/Question2b.pdf")

    print("Question 2 done.")

    # Question 3
    # use 100% of train and 100% of test set
    # Additive smoothing with best alpha found earlier
    # log-scaled probabilities
    # fit on train, predict on test, return accuracy, precision, recall, confusion matrix

    accuracy, precision, recall, confusion_matrix = run_experiment(
        percentage_positives_train=1, percentage_negatives_train=1,
        percentage_positives_test=1, percentage_negatives_test=1,
        alpha=best_alpha, log_scale=True,
        pos_class=pos_class, neg_class=neg_class)
    
    plot_experiment_results(
        accuracy, precision, recall, confusion_matrix,
        pos_class, neg_class, save_to_path="figures/Question3.pdf")

    print("Question 3 done.")

    # Question 4
    # use 30% of train and 100% of test set
    # Additive smoothing with best alpha found earlier
    # log-scaled probabilities
    # fit on train, predict on test, return accuracy, precision, recall, confusion matrix
    
    accuracy, precision, recall, confusion_matrix = run_experiment(
        percentage_positives_train=0.3, percentage_negatives_train=0.3,
        percentage_positives_test=1, percentage_negatives_test=1,
        alpha=best_alpha, log_scale=True,
        pos_class=pos_class, neg_class=neg_class)
    
    plot_experiment_results(
        accuracy, precision, recall, confusion_matrix,
        pos_class, neg_class, save_to_path="figures/Question4.pdf")

    print("Question 4 done.")

    # Question 6
    # use imbalanced train set and 100% of test set
    # Additive smoothing with best alpha found earlier
    # log-scaled probabilities
    # fit on train, predict on test, return accuracy, precision, recall, confusion matrix

    accuracy, precision, recall, confusion_matrix = run_experiment(
        percentage_positives_train=0.1, percentage_negatives_train=0.5,
        percentage_positives_test=1, percentage_negatives_test=1,
        alpha=best_alpha, log_scale=True,
        pos_class=pos_class, neg_class=neg_class)
    
    plot_experiment_results(
        accuracy, precision, recall, confusion_matrix,
        pos_class, neg_class, save_to_path="figures/Question6.pdf")

    print("Question 6 done.")


if __name__ == '__main__':
    print("Running code for experiments and generating figures.")
    main()
    print("Completed experiments and generated figures.")
