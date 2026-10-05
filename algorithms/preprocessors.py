"""Small preprocessing helpers used by the text algorithms."""

from collections import Counter


def doc_to_bow_vector(doc, vocab):
    """Convert a tokenized document into counts ordered by ``vocab``."""

    word_counts = Counter(doc)
    return [word_counts[word] for word in vocab]
