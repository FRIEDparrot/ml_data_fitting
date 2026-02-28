import numpy as np

def normalize_inputs(
        X_train: np.ndarray,
        lower_bounds,
        upper_bounds,
        X_test: np.ndarray|None = None):
    """
    General min-max normalization for inputs based on provided lower and upper bounds.

    :param X_train: Training input data (numpy array)
    :param lower_bounds: List of lower bounds for each column
    :param upper_bounds: List of upper bounds for each column
    :param X_test: Optional test input data (numpy array)
    :return: Normalized X_train, and X_test if provided
    """
    # Check shapes and lengths
    num_columns = X_train.shape[1]
    if len(lower_bounds) != num_columns or len(upper_bounds) != num_columns:
        raise ValueError(
            f"Length of lower_bounds ({len(lower_bounds)}) and upper_bounds ({len(upper_bounds)}) must equal the number of columns in X_train ({num_columns})")

    # Check that lower < upper for each column
    for i in range(num_columns):
        if lower_bounds[i] >= upper_bounds[i]:
            raise ValueError(
                f"Lower bound ({lower_bounds[i]}) must be less than upper bound ({upper_bounds[i]}) for column {i}")

    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy() if X_test is not None else None

    for i in range(num_columns):
        X_train_norm[:, i] = (X_train[:, i] - lower_bounds[i]) / (upper_bounds[i] - lower_bounds[i])
        if X_test is not None:
            X_test_norm[:, i] = (X_test[:, i] - lower_bounds[i]) / (upper_bounds[i] - lower_bounds[i])

    if X_test is not None:
        return X_train_norm, X_test_norm
    return X_train_norm
