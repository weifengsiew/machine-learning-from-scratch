"""Functions to preprocess documents prior to training or prediction"""

import re
from collections import Counter
from nltk.corpus import stopwords
import nltk

REPLACE_NO_SPACE = re.compile("[._;:!*`¦\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
nltk.download('stopwords', quiet=True)


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = REPLACE_NO_SPACE.sub("", text)
    text = REPLACE_WITH_SPACE.sub(" ", text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    words = text.split()
    return [w for w in words if w not in stop_words]


def doc_to_bow_vector(doc, vocab):
    """Convert one document to a Bag-of-Words vector.

    Args:
        doc (list): Document represented as a list of words.
        vocab (list): Vocabulary containing all unique words.

    Returns:
        bow_vector (list): Word-count vector ordered according to vocab.
    """

    word_counts = Counter(doc)
    bow_vector = [word_counts[word] for word in vocab]
    return bow_vector


def docs_to_bow_vectors(docs, vocab):
    """Convert multiple documents to Bag-of-Words vectors.

    Args:
        docs (list): Documents, where each document is a list of words.
        vocab (list): Vocabulary containing all unique words.

    Returns:
        bow_vectors (list): Bag-of-Words vectors
    """
    bow_vectors = [doc_to_bow_vector(doc, vocab) for doc in docs]
    return bow_vectors
