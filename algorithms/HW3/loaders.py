"""Functions for loading datasets and splitting into attributes + labels"""

import numpy as np

def load_dataset(dataset_path):
    data_with_column_names = np.loadtxt(fname=dataset_path, delimiter=",", dtype=str)

    column_names = data_with_column_names[0]
    data = data_with_column_names[1:]

    return data, column_names


def split_attributes_and_labels(data, column_names):
    label = list(column_names).index("label")

    X = np.delete(arr=data, obj=label, axis=1)
    y = data[:, label]

    attribute_names = np.delete(arr=column_names, obj=label).tolist()

    attribute_types = []
    for attribute_name in attribute_names:
        if attribute_name.endswith("_num"):
            attribute_types.append("numeric")
        elif attribute_name.endswith("_cat"):
            attribute_types.append("categorical")
        else:
            raise ValueError(f"Unknown attribute type for {attribute_name}")

    return X, y, attribute_names, attribute_types
