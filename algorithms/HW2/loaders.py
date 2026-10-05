"""Functions to load datasets"""

import random
from pathlib import Path
import pandas as pd
from src.preprocessors import preprocess_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_training_set(percentage_positives, percentage_negatives):
    vocab = set()
    positive_instances = []
    negative_instances = []

    df = pd.read_csv(DATA_DIR / 'train-positive.csv')
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_positives:
            continue
        contents = preprocess_text(contents)
        positive_instances.append(contents)
        vocab = vocab.union(set(contents))

    df = pd.read_csv(DATA_DIR / 'train-negative.csv')
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_negatives:
            continue
        contents = preprocess_text(contents)
        negative_instances.append(contents)
        vocab = vocab.union(set(contents))

    return positive_instances, negative_instances, list(vocab)


def load_test_set(percentage_positives, percentage_negatives):
    positive_instances = []
    negative_instances = []

    df = pd.read_csv(DATA_DIR / 'test-positive.csv')
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_positives:
            continue
        contents = preprocess_text(contents)
        positive_instances.append(contents)
    df = pd.read_csv(DATA_DIR / 'test-negative.csv')
    
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_negatives:
            continue
        contents = preprocess_text(contents)
        negative_instances.append(contents)

    return positive_instances, negative_instances


def instances_to_X_and_y(pos_instances, neg_instances):
    """Combine positive and negative documents and create class labels.

    Args:
        pos_instances (list): Positive documents.
        neg_instances (list): Negative documents.

    Returns:
        X (list): Positive documents combined with negative documents.
        y (list): Class labels corresponding to X.
    """

    X = pos_instances + neg_instances
    y = ['positive'] * len(pos_instances) + ['negative'] * len(neg_instances)
    return X,y
