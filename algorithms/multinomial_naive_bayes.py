"""Functions to implement the multinomial naive bayes algorithm"""

from collections import Counter
import numpy as np
from .preprocessors import doc_to_bow_vector

def compute_class_prior_probabilities(y, log_scale):
    """Compute prior probabilities for each document class

    Args:
        y (list): Class labels, one label per document.
        log_scale (bool): Whether to return probabilities (False) or log-probabilities (True).

    Returns:
        class_prior_probabilities (dict): Classes mapped to probabilities or log-probabilities.
    """

    class_counts = Counter(y)
    total_docs = sum(class_counts.values())
    class_prior_probabilities = {}

    for class_i, count in class_counts.items():
        probability = count / total_docs
        if log_scale:
            probability = np.log(probability)
        class_prior_probabilities[class_i] = probability

    return class_prior_probabilities


def compute_word_probabilities_given_class(X, y, vocab, alpha, log_scale):
    """Compute word probabilities conditional on each document class.

    Args:
        X (list): Documents, where each document is a list of words.
        y (list): Class labels, one label per document in X.
        vocab (list): Vocabulary containing all unique words.
        alpha (float): Additive smoothing parameter.
        log_scale (bool): Whether to return probabilities (False) or log-probabilities (True).

    Returns:
        word_probabilities_given_class (dict): Classes mapped to arrays of word
            probabilities or log-probabilities, ordered according to vocab.
    """
    
    class_to_word_counts = {}
    for doc, class_i in zip(X, y):
        class_to_word_counts.setdefault(class_i, Counter()).update(doc)

    word_probabilities_given_class = {}

    for class_i, word_counts in class_to_word_counts.items():
        counts = np.fromiter(
            (word_counts.get(word, 0) for word in vocab),
            dtype=np.float64,
            count=len(vocab))

        if log_scale:
            probabilities = np.log(counts + alpha) - np.log(counts.sum() + alpha * len(vocab))
        else:
            probabilities = (counts + alpha) / (counts.sum() + alpha * len(vocab))

        word_probabilities_given_class[class_i] = probabilities

    return word_probabilities_given_class


def compute_class_probability_given_doc(
    class_prior_probability,
    word_probabilities_given_class,
    bow_vector,
    log_scale):
    """Compute probability of one class given one document.

    Args:
        class_prior_probability (float): Prior probability of the class.
        word_probabilities_given_class (np.ndarray): Probability of each word given the class.
        bow_vector (np.ndarray): Bag-of-Words vector for the document.
        log_scale (bool): Whether probabilities are log-transformed (True) or not (False).

    Returns:
        class_probability_given_doc (float): Probability of the class given the document.
    """

    if log_scale:
        class_probability_given_doc = (
            class_prior_probability + np.sum(bow_vector * word_probabilities_given_class))
    else:
        class_probability_given_doc = (
            class_prior_probability * np.prod(word_probabilities_given_class ** bow_vector))

    return class_probability_given_doc


def compute_class_probabilities_given_doc(
    class_prior_probabilities,
    word_probabilities_given_class,
    bow_vector,
    log_scale):
    """Compute probability scores of all classes given one document.

    Args:
        class_prior_probabilities (dict): Classes mapped to their prior probabilities.
        word_probabilities_given_class (dict): Probability of each word given the class.
        bow_vector (np.ndarray): Bag-of-Words vector for the document.
        log_scale (bool): Whether probabilities are log-transformed (True) or not (False).

    Returns:
        class_probabilities_given_doc (dict): Probability of each class given the document.
    """

    class_probabilities_given_doc = {}

    for class_i, class_prior_probability in class_prior_probabilities.items():
        class_probability_given_doc = compute_class_probability_given_doc(
            class_prior_probability,
            word_probabilities_given_class[class_i],
            bow_vector,
            log_scale=log_scale)

        class_probabilities_given_doc[class_i] = class_probability_given_doc

    return class_probabilities_given_doc


class multinomial_naive_bayes_classifier:
    def __init__(self, alpha, log_scale):
        """Initialise the multinomial naive bayes classifier

        Args:
             alpha (float): Additive smoothing parameter.
            log_scale (bool): Whether to return probabilities (False) or log-probabilities (True).
        """
        
        self.alpha = alpha
        self.log_scale = log_scale
        self.X_train = None
        self.y_train = None
        self.vocab = None
        self.vocab_index = None
        self.class_prior_probabilities = None
        self.word_probabilities_given_class = None

    def fit(self, X_train, y_train, vocab):
        """Fit the multinomial naive bayes classifier

        Args:
            X_train (list): Documents, where each document is a list of words.
            y_train (list): Class labels, one label per document in X.
            vocab (list): Vocabulary containing all unique words.

        Returns:
            self: Fitted classifier.
        """

        self.X_train = X_train
        self.y_train = y_train
        self.vocab = list(vocab)
        self.vocab_index = {word: index for index, word in enumerate(self.vocab)}
        self.class_prior_probabilities = compute_class_prior_probabilities(
            self.y_train, log_scale=self.log_scale)
        self.word_probabilities_given_class = compute_word_probabilities_given_class(
            self.X_train, self.y_train, self.vocab,
            alpha=self.alpha, log_scale=self.log_scale)
        return self

    def predict_one(self, doc):
        """Predict class of one document.

        Args:
            doc (list): Document represented as a list of words.

        Returns:
            predicted_class: Class with the highest probability given the document.
        """
     
        bow_vector = np.array(doc_to_bow_vector(doc, self.vocab))
        class_probabilities_given_doc = compute_class_probabilities_given_doc(
            self.class_prior_probabilities,
            self.word_probabilities_given_class,
            bow_vector,
            log_scale=self.log_scale)
        predicted_class = max(class_probabilities_given_doc, key=class_probabilities_given_doc.get)

        return predicted_class

    def predict(self, docs):
        """Predict class of multiple documents.
        
        Args:
            docs (list): Documents, where each document is a list of words.

        Returns:
            predicted_classes (list): Predicted class of documents.
        """

        predicted_classes = [self.predict_one(doc) for doc in docs]
        return predicted_classes
