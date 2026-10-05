"""Educational implementation of multinomial Naive Bayes for token counts."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .preprocessors import doc_to_bow_vector

Document = list[str]
Label = Any


def compute_class_prior_probabilities(
    labels: list[Label], log_scale: bool
) -> dict[Label, float]:
    """Compute prior probability for each document class.

    Args:
        labels: Class label for each training document.
        log_scale: Return logarithms instead of ordinary probabilities when true.

    Returns:
        Mapping from each class label to its prior probability or log probability.
    """
    class_counts = Counter(labels)
    total_documents = sum(class_counts.values())
    class_priors: dict[Label, float] = {}
    for class_label, count in class_counts.items():
        probability = count / total_documents
        class_priors[class_label] = float(
            np.log(probability) if log_scale else probability
        )
    return class_priors


def compute_word_probabilities_given_class(
    documents: list[Document],
    labels: list[Label],
    vocabulary: list[str],
    alpha: float,
    log_scale: bool,
) -> dict[Label, np.ndarray]:
    """Compute smoothed word probabilities conditional on each class.

    Args:
        documents: Tokenized training documents.
        labels: Class label aligned with each document.
        vocabulary: Ordered vocabulary used for probability vectors.
        alpha: Additive smoothing strength.
        log_scale: Return log probabilities when true.

    Returns:
        Mapping from class labels to word-probability arrays ordered by vocabulary.
    """
    class_word_counts: dict[Label, Counter[str]] = {}
    for document, class_label in zip(documents, labels):
        class_word_counts.setdefault(class_label, Counter()).update(document)

    probabilities_by_class: dict[Label, np.ndarray] = {}
    for class_label, word_counts in class_word_counts.items():
        counts = np.fromiter(
            (word_counts.get(word, 0) for word in vocabulary),
            dtype=np.float64,
            count=len(vocabulary),
        )
        denominator = counts.sum() + alpha * len(vocabulary)
        if log_scale:
            probabilities = np.log(counts + alpha) - np.log(denominator)
        else:
            probabilities = (counts + alpha) / denominator
        probabilities_by_class[class_label] = probabilities
    return probabilities_by_class


def compute_class_probability_given_doc(
    class_prior_probability: float,
    word_probabilities_given_class: np.ndarray,
    bow_vector: np.ndarray,
    log_scale: bool,
) -> float:
    """Compute one class score for a bag-of-words vector.

    Args:
        class_prior_probability: Prior probability or log prior for the class.
        word_probabilities_given_class: Word probabilities for the class.
        bow_vector: Word-count vector ordered by the vocabulary.
        log_scale: Interpret inputs as logarithms when true.

    Returns:
        Probability score or log-probability score for the class and document.
    """
    if log_scale:
        score = class_prior_probability + np.sum(
            bow_vector * word_probabilities_given_class
        )
    else:
        score = class_prior_probability * np.prod(
            word_probabilities_given_class**bow_vector
        )
    return float(score)


def compute_class_probabilities_given_doc(
    class_prior_probabilities: dict[Label, float],
    word_probabilities_given_class: dict[Label, np.ndarray],
    bow_vector: np.ndarray,
    log_scale: bool,
) -> dict[Label, float]:
    """Compute class scores for one bag-of-words vector.

    Args:
        class_prior_probabilities: Prior or log-prior for each class.
        word_probabilities_given_class: Word probabilities for each class.
        bow_vector: Word-count vector ordered by the vocabulary.
        log_scale: Interpret inputs as logarithms when true.

    Returns:
        Mapping from each class label to its probability score.
    """
    return {
        class_label: compute_class_probability_given_doc(
            class_prior_probability,
            word_probabilities_given_class[class_label],
            bow_vector,
            log_scale,
        )
        for class_label, class_prior_probability in class_prior_probabilities.items()
    }


class multinomial_naive_bayes_classifier:
    """Classify tokenized documents using multinomial Naive Bayes."""

    def __init__(self, alpha: float, log_scale: bool) -> None:
        """Initialize a multinomial Naive Bayes classifier.

        Args:
            alpha: Additive smoothing strength.
            log_scale: Store and calculate probabilities in log space when true.
        """
        self.alpha = alpha
        self.log_scale = log_scale
        self.X_train: list[Document] | None = None
        self.y_train: list[Label] | None = None
        self.vocab: list[str] | None = None
        self.vocab_index: dict[str, int] | None = None
        self.class_prior_probabilities: dict[Label, float] | None = None
        self.word_probabilities_given_class: dict[Label, np.ndarray] | None = None

    def fit(
        self, X_train: list[Document], y_train: list[Label], vocab: list[str]
    ) -> multinomial_naive_bayes_classifier:
        """Estimate class priors and smoothed word probabilities.

        Args:
            X_train: Tokenized training documents.
            y_train: Class labels aligned with ``X_train``.
            vocab: Ordered vocabulary used to create count vectors.

        Returns:
            This fitted classifier.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.vocab = list(vocab)
        self.vocab_index = {word: index for index, word in enumerate(self.vocab)}
        self.class_prior_probabilities = compute_class_prior_probabilities(
            self.y_train, log_scale=self.log_scale
        )
        self.word_probabilities_given_class = compute_word_probabilities_given_class(
            self.X_train,
            self.y_train,
            self.vocab,
            alpha=self.alpha,
            log_scale=self.log_scale,
        )
        return self

    def predict_one(self, document: Document) -> Label:
        """Predict the most likely class for one tokenized document.

        Args:
            document: Tokenized document to classify.

        Returns:
            Class label with the highest probability score.
        """
        if self.vocab is None or self.class_prior_probabilities is None:
            raise RuntimeError(
                "Naive Bayes classifier must be fitted before prediction"
            )
        if self.word_probabilities_given_class is None:
            raise RuntimeError("Naive Bayes word probabilities are not initialized")
        bow_vector = np.array(doc_to_bow_vector(document, self.vocab))
        class_scores = compute_class_probabilities_given_doc(
            self.class_prior_probabilities,
            self.word_probabilities_given_class,
            bow_vector,
            log_scale=self.log_scale,
        )
        return max(class_scores, key=class_scores.get)

    def predict(self, documents: list[Document]) -> list[Label]:
        """Predict the most likely class for each tokenized document.

        Args:
            documents: Tokenized documents to classify.

        Returns:
            Predicted class label for each document.
        """
        return [self.predict_one(document) for document in documents]
