"""Functions for loading datasets and splitting into attributes and labels"""

import numpy as np


def load_dataset(dataset_path):
    """Load dataset.

    Args:
        dataset_path (Path): Path to dataset CSV.

    Returns:
        data (np.ndarray): Dataset values.
        column_names (np.ndarray): Column names.
    """
    data_with_column_names = np.loadtxt(fname=dataset_path, delimiter=",", dtype=str)
    column_names = data_with_column_names[0]
    data = data_with_column_names[1:]
    return data, column_names


def split_attributes_and_labels(data, column_names):
    """Split attributes and labels.

    Args:
        data (np.ndarray): Dataset values.
        column_names (np.ndarray): Column names.

    Returns:
        X (np.ndarray): Attributes.
        y (np.ndarray): Class labels.
        attribute_names (list): Attribute names.
        attribute_types (list): Attribute types.
    """
    label_index = list(column_names).index("label")

    X = np.delete(arr=data, obj=label_index, axis=1)
    y = data[:, label_index]
    attribute_names = np.delete(arr=column_names, obj=label_index).tolist()

    attribute_types = []

    for attribute_name in attribute_names:
        if attribute_name.endswith("_num"):
            attribute_types.append("numeric")
        elif attribute_name.endswith("_cat"):
            attribute_types.append("categorical")
        else:
            raise ValueError(f"Unknown attribute type for {attribute_name}")

    return X, y, attribute_names, attribute_types
