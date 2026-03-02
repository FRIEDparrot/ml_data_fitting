import numpy as np
from ml_data_fitting.core.regressors import RegressionModel

# ─────────────────────────────────────────────────────────────────────────────
# Core helper: train a list of methods on one target, return predictions dict
# ─────────────────────────────────────────────────────────────────────────────
def fast_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test:  np.ndarray,
    methods: list,
    max_iterations: int = 1000,
) -> tuple[dict, dict]:
    """
    Train each method in `methods` on (x_train, y_train) and predict x_test.

    Returns
    -------
    train_preds_dict : {method_name: y_pred_array}
        Dictionary of training predictions for each method.
    test_preds_dict : {method_name: y_pred_array}
        Dictionary of testing predictions for each method.
    """
    assert (y_train.ndim == 1 or (y_train.ndim == 2 and y_train.shape[1] == 1))
    if y_train.ndim == 0:
        y_train = y_train[:, 1]   # reshape to (n_samples, 1) if 1D array
    train_preds_dict = {}
    test_preds_dict = {}
    for method in methods:
        print(f"  Fitting {method}...")
        model = RegressionModel(method=method, max_iterations=max_iterations)
        model.fit(x_train, y_train)
        train_preds_dict[method] = model.predict(x_train)
        test_preds_dict[method] = model.predict(x_test)
    return train_preds_dict, test_preds_dict