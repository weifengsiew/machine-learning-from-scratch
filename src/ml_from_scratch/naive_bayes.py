"""Multinomial Naive Bayes for tokenized documents."""

from collections import Counter

import numpy as np


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0, use_log_probabilities=True):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.use_log_probabilities = use_log_probabilities

    def fit(self, documents, labels, vocabulary=None):
        self.vocabulary = list(vocabulary or sorted({word for doc in documents for word in doc}))
        self.index = {word: i for i, word in enumerate(self.vocabulary)}
        self.classes, counts = np.unique(labels, return_counts=True)
        self.class_priors = {c: count / len(labels) for c, count in zip(self.classes, counts)}
        self.word_probabilities = {}
        for label in self.classes:
            word_counts = Counter(word for doc, y in zip(documents, labels) if y == label for word in doc)
            counts = np.array([word_counts[word] for word in self.vocabulary], dtype=float)
            probabilities = (counts + self.alpha) / (counts.sum() + self.alpha * len(self.vocabulary))
            self.word_probabilities[label] = np.log(probabilities) if self.use_log_probabilities else probabilities
        return self

    def _bow(self, document):
        vector = np.zeros(len(self.vocabulary))
        for word in document:
            if word in self.index:
                vector[self.index[word]] += 1
        return vector

    def predict(self, documents):
        if not hasattr(self, "classes"):
            raise ValueError("fit must be called before predict")
        predictions = []
        for document in documents:
            bow = self._bow(document)
            scores = {}
            for label in self.classes:
                probabilities = self.word_probabilities[label]
                scores[label] = (np.log(self.class_priors[label]) + bow @ probabilities
                                 if self.use_log_probabilities else
                                 self.class_priors[label] * np.prod(probabilities ** bow))
            predictions.append(max(scores, key=scores.get))
        return predictions
