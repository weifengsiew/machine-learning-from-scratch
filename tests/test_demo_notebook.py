import json
from pathlib import Path

import pandas as pd


def test_parkinsons_notebook_is_valid_json():
    notebook_path = Path("parkinsons_demo.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"]


def test_parkinsons_split_is_small_and_reproducible():
    train = pd.read_csv("data/parkinsons_train.csv")
    test = pd.read_csv("data/parkinsons_test.csv")
    assert len(train) == 156
    assert len(test) == 39
    assert list(train.columns) == list(test.columns)
