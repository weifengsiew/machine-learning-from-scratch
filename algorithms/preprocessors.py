"""Small preprocessing helpers used by the text algorithms."""

from __future__ import annotations

from collections import Counter


def doc_to_bow_vector(doc: list[str], vocab: list[str]) -> list[int]:
    """Convert a tokenized document into counts ordered by ``vocab``.

    Args:
        doc: Tokens from one document.
        vocab: Ordered vocabulary defining vector positions.

    Returns:
        Word-count vector with one count for each vocabulary entry.
    """

    word_counts = Counter(doc)
    return [word_counts[word] for word in vocab]
