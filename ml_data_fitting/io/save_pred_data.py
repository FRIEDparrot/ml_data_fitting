import os
import pandas as pd
from typing import Optional, List
import numpy as np


def _compute_accuracy(y_true, y_pred, threshold: float = 0.05) -> float:
    rel_err = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12)
    return (rel_err <= threshold).mean(axis=0) * 100

def save_accuracy_csv(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    save_path: str,
    column_names: Optional[List[str]] = None,
    threshold: float = 0.05,
) -> None:
    """
    Save y_true and y_pred to a CSV file with columns: y_true | y_pred.

    threshold : relative error threshold for accuracy calculation (not saved in CSV).
    column_names: list of k method names; defaults to ["Method 1", ...].
    save_path : destination CSV path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if y_train.ndim == 1:
        y_train = y_train[:, np.newaxis]
    if y_test.ndim == 1:
        y_test = y_test[:, np.newaxis]
    if column_names is None:
        column_names = ["Method" + str(i) for i in range(y_train.shape[1])]
        assert len(column_names) == y_train_pred.shape[1], "column_names length must match number of methods in y_train_pred"

    # use method as x-axis
    result = {
        "methods": column_names,
        "train_accuracy (%)": _compute_accuracy(
        y_train, y_train_pred, threshold=threshold
    ),  "test_accuracy (%)": _compute_accuracy(
        y_test, y_test_pred, threshold=threshold
    )}
    df = pd.DataFrame(result)
    df.to_csv(save_path, index=False)
    print(f"  Saved predictions → {save_path}")


def save_predictions_csv(
    y_true:    np.ndarray,
    preds:     dict,
    save_path: str,
) -> None:
    """
    Save y_true and all y_pred columns to a CSV file.

    Columns:  y_true | y_pred_Method1 | y_pred_Method2 | ...

    Parameters
    ----------
    y_true    : 1-D ground truth array.
    preds     : {method_name: y_pred_array} from train_and_predict().
    save_path : destination CSV path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    df = pd.DataFrame({"y_true": y_true})
    for method, y_pred in preds.items():
        df[f"y_pred_{method}"] = y_pred
    df.to_csv(save_path, index=False)
    print(f"  Saved predictions → {save_path}")


def save_tolerance_acc_csv(
        y_true:          np.ndarray,
        y_pred:          np.ndarray,
        save_path:       str,
        threshold_range: tuple          = (0.01, 0.5),
        threshold_steps: int            = 50,
        column_names:    Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute cumulative tolerance-accuracy curves and save to CSV.

    The threshold is the first column, followed by one accuracy column
    per prediction method.

    CSV layout:
        threshold (%) | accuracy_Method1 (%) | accuracy_Method2 (%) | ...

    Parameters
    ----------
    y_true          : (n,) ground-truth values.
    y_pred          : (n,) for a single method, or (n, k) for k methods.
    save_path       : destination CSV path (directories created if needed).
    threshold_range : (min, max) as fractions, e.g. (0.01, 0.20).
    threshold_steps : number of evenly-spaced threshold points.
    column_names    : list of k method names; defaults to ["Method 1", ...].

    Returns
    -------
    df : pd.DataFrame  — the same table that was written to disk.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()

    # Normalise y_pred to 2-D: (n_samples, n_methods)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    n, k = y_pred.shape

    if column_names is None:
        column_names = [f"Method {i + 1}" for i in range(k)]
    if len(column_names) != k:
        raise ValueError(
            f"column_names has {len(column_names)} entries but y_pred has {k} columns."
        )

    thresholds = np.linspace(threshold_range[0], threshold_range[1], threshold_steps)

    # Build dataframe: first column = threshold in %
    df = pd.DataFrame({"threshold (%)": thresholds * 100})

    for i, name in enumerate(column_names):
        rel_err = np.abs(y_pred[:, i] - y_true) / (np.abs(y_true) + 1e-12)
        accuracy = np.array([(rel_err <= t).mean() * 100 for t in thresholds])
        df[f"accuracy_{name} (%)"] = accuracy

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"  Saved tolerance-accuracy table → {save_path}  ({threshold_steps} rows, {k} method(s))")

    return df


def save_all_pred_data(
    y_true_dict: dict,
    y_train_dict: dict,
    y_test_dict: dict,
    data_name:str,
    result_base_dir:str= "results",
    acc_threshold: float | list[float] = 0.05,
    tol_acc_threshold_range: tuple = (0.01, 0.10),
    tol_acc_threshold_steps: int = 50,
):
    """
    :param y_true_dict: { "train", "test" } true values
    :param y_train_dict: { method_name: y_pred_array } for training set
    :param y_test_dict:
    :param data_name:
    :param result_base_dir:
    :param acc_threshold: relative error threshold for accuracy calculation (e.g. 0.05 for 5% error)
    :param tol_acc_threshold_range: (min, max) range for tolerance accuracy curve thresholds (e.g. (0.01, 0.10) for 1% to 10% error)
    :param tol_acc_threshold_steps: number of threshold points to evaluate for tolerance accuracy curve
    :return:
    """
    # Check if methods was matched
    method_names = list(y_train_dict.keys())  # Preserve original order
    test_methods = set(y_test_dict.keys())
    assert len(method_names) == len(y_train_dict) == len(y_test_dict)
    assert set(method_names) == test_methods, f"Mismatch in methods: train={set(method_names)}, test={test_methods}"

    y_train_pred = np.column_stack(list(y_train_dict[name] for name in method_names))
    y_test_pred = np.column_stack(list(y_test_dict[name] for name in method_names))

    if not isinstance(acc_threshold, list):
        acc_threshold = [acc_threshold]

    for thresh in acc_threshold:
        save_accuracy_csv(
            y_train=y_true_dict["train"],
            y_test=y_true_dict["test"],
            y_train_pred=y_train_pred,
            y_test_pred=y_test_pred,
            column_names=method_names,
            save_path=f"{result_base_dir}/{data_name}_accuracy({int(thresh * 100)}%).csv",
            threshold=thresh,
        )

    # Save ŷ table for stress
    save_predictions_csv(
        y_true=y_true_dict["test"],
        preds=y_test_dict,
        save_path=f"{result_base_dir}/{data_name}_predictions.csv",
    )
    save_tolerance_acc_csv(
        y_true=y_true_dict["test"],
        y_pred=y_test_pred,
        column_names=method_names,
        threshold_range=tol_acc_threshold_range,
        threshold_steps=tol_acc_threshold_steps,
        save_path=f"{result_base_dir}/{data_name}_tolerance_accuracy.csv",
    )