"""Functions for preprocessing attributes"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessor(attribute_types):
    """Create preprocessor for categorical and numerical attributes.

    Args:
        attribute_types (list): Attribute types.

    Returns:
        preprocessor (ColumnTransformer): Preprocessor.
    """
    categorical_columns = []

    for column_index, attribute_type in enumerate(attribute_types):
        if attribute_type == "categorical":
            categorical_columns.append(column_index)

    numeric_columns = []

    for column_index, attribute_type in enumerate(attribute_types):
        if attribute_type == "numeric":
            numeric_columns.append(column_index)

    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        ("numeric", StandardScaler(), numeric_columns),
    ])

    return preprocessor


def fit_preprocessor(X_train, attribute_types):
    """Fit preprocessor on training instances.

    Args:
        X_train (np.ndarray): Attributes of training instances.
        attribute_types (list): Attribute types.

    Returns:
        preprocessor (ColumnTransformer): Fitted preprocessor.
    """
    preprocessor = create_preprocessor(attribute_types)
    preprocessor.fit(X_train)
    return preprocessor


def preprocess_attributes(preprocessor, X):
    """Preprocess attributes using fitted preprocessor.

    Args:
        preprocessor (ColumnTransformer): Fitted preprocessor.
        X (np.ndarray): Attributes.

    Returns:
        X_preprocessed (np.ndarray): Preprocessed attributes.
    """
    X_preprocessed = preprocessor.transform(X)
    return X_preprocessed
