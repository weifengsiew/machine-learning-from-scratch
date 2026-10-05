from collections import Counter

import numpy as np


def euclidean_distance(X_i, X_j):
    distance = np.sqrt(np.sum((X_i - X_j) ** 2))
    return distance


class KNN_classifier:
    def __init__(self, k):
        # number of neighbors k
        self.k = k
        # train set features
        self.X_train = None
        # train set labels
        self.y_train = None

    def fit(self, X_train, y_train):
        # train set features
        self.X_train = X_train
        # train set labels
        self.y_train = y_train

        return self

    def predict_single_instance(self, X_j):
        # indices for all train instances
        indices = np.arange(self.X_train.shape[0])
        # euclidean distance between each train instance and the j-th instance to be predicted
        distances = [euclidean_distance(self.X_train[index], X_j) for index in indices]
        # sort indices from smallest to largest distance
        sorted_indices = np.argsort(distances)
        # indices of the k nearest neighbors
        knn_indices = sorted_indices[: self.k]
        # labels of the k nearest neighbors
        y_knn = self.y_train[knn_indices]
        # predict most common label among the k nearest neighbors
        y_j_pred = Counter(y_knn).most_common(1)[0][0]
        return y_j_pred

    def predict(self, X):
        # predict labels for all instances to be predicted
        y_pred = np.array([self.predict_single_instance(X[j]) for j in range(X.shape[0])])
        return y_pred
