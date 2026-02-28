import re
import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from scipy.stats import pearsonr
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from typing import Any, Optional, List, Dict
import pandas as pd


class RegressionModel:
    """
    Single-output regression wrapper for stress & mass prediction.
    Designed for small datasets (~130 samples) with <5% MAPE target.

    Now single-output only — one model per target — enabling full per-target
    metric reporting: MAPE, CVRMSE, MaxAPE, Pearson r, and R².

    Note on R²:
        R² can be negative when target variance is very low relative to
        prediction error. In that case, prefer CVRMSE and Pearson r as the
        primary quality indicators.
    """

    SUPPORTED_METHODS = [
        'CatBoost', 'GradientBoost', 'RandomForest',
        'XGBoost', 'RBF', 'GaussianProcess', 'Ridge',
        'LightGBM', 'HistGradientBoost', 'ExtraTrees',
    ]

    def __init__(self, method: str = 'CatBoost', max_iterations: int = 500):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method: '{method}'. "
                f"Choose from: {self.SUPPORTED_METHODS}"
            )
        self.method = method
        self.max_iterations = max_iterations
        self.model: Any = self._select_model()
        self.is_fitted: bool = False
        self.feature_names: Optional[list] = None
        self.target_name: Optional[str] = None

    def _select_model(self) -> Any:
        """Build single-output model with tuned hyperparameters."""

        if self.method == 'CatBoost':
            return CatBoostRegressor(
                iterations=self.max_iterations,
                learning_rate=0.03,
                depth=4,
                l2_leaf_reg=5,
                bagging_temperature=0.8,
                random_strength=1.5,
                border_count=64,
                verbose=0,
                random_seed=42
            )

        elif self.method == 'GradientBoost':
            return GradientBoostingRegressor(
                n_estimators=self.max_iterations,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
                min_samples_leaf=4,
                max_features=0.8,
                random_state=42
            )

        elif self.method == 'RandomForest':
            return RandomForestRegressor(
                n_estimators=self.max_iterations,
                max_depth=8,
                min_samples_leaf=3,
                min_samples_split=6,
                max_features=0.7,
                bootstrap=True,
                oob_score=True,
                random_state=42
            )

        elif self.method == 'XGBoost':
            return XGBRegressor(
                objective='reg:squarederror',
                n_estimators=self.max_iterations,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=2.0,
                min_child_weight=3,
                verbosity=0,
                random_state=42
            )

        elif self.method == 'LightGBM':
            # LightGBM is a fast, high-performance gradient boosting implementation
            return LGBMRegressor(
                n_estimators=self.max_iterations,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=-1,
            )

        elif self.method == 'HistGradientBoost':
            # Scikit-learn's histogram-based gradient boosting (fast and robust)
            return HistGradientBoostingRegressor(
                max_iter=self.max_iterations,
                learning_rate=0.03,
                max_depth=8,
                min_samples_leaf=5,
                max_bins=255,
                random_state=42,
            )

        elif self.method == 'ExtraTrees':
            return ExtraTreesRegressor(
                n_estimators=self.max_iterations,
                max_depth=12,
                min_samples_leaf=3,
                min_samples_split=6,
                max_features=0.7,
                bootstrap=False,
                random_state=42,
            )

        elif self.method == 'RBF':
            return SVR(
                kernel='rbf',
                C=100,
                epsilon=0.01,
                gamma='scale',
                cache_size=500
            )

        elif self.method == 'GaussianProcess':
            return GaussianProcessRegressor(
                kernel=Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3))
                       + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                alpha=1e-3,
                normalize_y=True,
                n_restarts_optimizer=10,
                random_state=42
            )

        elif self.method == 'Ridge':
            return Ridge(
                alpha=1.0,
                fit_intercept=True,
                solver='auto',
            )

        return None

    def fit(self,
            X: np.ndarray, y: np.ndarray,
            feature_names: Optional[list] = None,
            target_name: Optional[str] = None) -> 'RegressionModel':
        """
        Fit the model to a single target.

        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples,)  — 1D array for one target
        feature_names:  column name for each feature (optional, for reporting)
        target_name:  name of the target variable (optional, for reporting)
        """
        if y.ndim != 1:
            raise ValueError(
                "y must be 1D (single target). "
                "Create separate RegressionModel instances per target."
            )
        self.feature_names = feature_names
        self.target_name = target_name or "Target"
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions as (n_samples,) array."""
        self._check_fitted()
        return self.model.predict(X)

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Compute all metrics for a single target.

        Metrics
        -------
        MAPE_%   : Mean Absolute Percentage Error (%)
        CVRMSE_% : RMSE / mean(y_true) × 100  — scale-free spread of errors
        MaxAPE_% : Maximum Absolute Percentage Error — worst-case error
        Pearson_r: Linear correlation coefficient [-1, 1]
                   Preferred over R² when target variance is low.
                   r > 0.95 → strong predictive relationship.
        R2       : Coefficient of determination. Can be negative for
                   low-variance targets even when MAPE is acceptable.
        """
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        cvrmse = rmse / np.mean(y_true) * 100
        max_ape = np.max(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        pearson_r, _ = pearsonr(y_true, y_pred)

        return {
            "MAPE_%": round(mape, 3),
            "CVRMSE_%": round(cvrmse, 3),
            "MaxAPE_%": round(max_ape, 3),
            "Pearson_r": round(pearson_r, 4),
            "R2": round(r2, 4),
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 cv_folds: int = 5) -> dict:
        """
        Cross-validated OOF evaluation with full metric suite.

        Returns dict of metrics for this target.
        """
        if y.ndim != 1:
            raise ValueError("y must be 1D for single-target evaluation.")

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        oof_preds = np.zeros_like(y, dtype=float)

        for train_idx, test_idx in kf.split(X):
            fold_model = self.__class__(
                method=self.method,
                max_iterations=self.max_iterations
            )._select_model()
            fold_model.fit(X[train_idx], y[train_idx])
            oof_preds[test_idx] = fold_model.predict(X[test_idx])

        metrics = self._compute_metrics(y, oof_preds)
        name = self.target_name or "Target"

        mape_flag = "✅" if metrics["MAPE_%"] < 5.0 else "❌"
        cvrmse_flag = "✅" if metrics["CVRMSE_%"] < 5.0 else "❌"
        pearson_flag = "✅" if metrics["Pearson_r"] > 0.75 else ("⚠️" if metrics["Pearson_r"] > 0.50 else "❌")

        print(f"\n{'=' * 75}")
        print(f"  CV Evaluation — {self.method} | Target: {name} ({cv_folds}-fold OOF)")
        print(f"{'=' * 75}")
        print(f"  {'Metric':<14} {'Value':>10}   {'Flag'}")
        print(f"  {'-' * 14}-+-{'-' * 10}---{'-' * 6}")
        print(f"  {'MAPE':<14} {metrics['MAPE_%']:>9.3f}%  {mape_flag}  (< 5% target)")
        print(f"  {'CVRMSE':<14} {metrics['CVRMSE_%']:>9.3f}%  {cvrmse_flag}  (< 5% target)")
        print(f"  {'MaxAPE':<14} {metrics['MaxAPE_%']:>9.3f}%      (worst-case error)")
        print(f"  {'Pearson r':<14} {metrics['Pearson_r']:>10.4f}  {pearson_flag}  (> 0.75)")
        print(f"  {'R²':<14} {metrics['R2']:>10.4f}       (may be neg. if low-variance target)")
        print(f"{'=' * 75}")
        print(f"  ⚠  R² note: target CV = {np.std(y) / np.mean(y) * 100:.2f}%. "
              f"{'Pearson r is the primary indicator here.' if np.std(y) / np.mean(y) < 0.10 else 'R² is reliable here.'}")
        print(f"{'=' * 75}\n")

        return metrics

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")

    def __repr__(self):
        return (
            f"RegressionModel(method='{self.method}', "
            f"max_iterations={self.max_iterations}, "
            f"target='{self.target_name}', "
            f"fitted={self.is_fitted})"
        )

def run_regression_eval(
        x: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]]= None,
        methods: Optional[list] = None,
        max_iterations: int = 500,
        cv_folds: int = 5,
        save_path: str = "regression_results.json",
) -> dict:
    """
    Evaluate all methods × all targets in one call

    Parameters
    ----------
    x       : feature matrix (n_samples, n_features)
    y       : target matrix (n_samples, n_targets)
    feature_names : list of feature names (for reporting)
    target_names : list of target names (for reporting)
    methods : list of method names; defaults to all supported
    max_iterations : max iterations for iterative models (default 500)
    cv_folds : number of CV folds for evaluation (default 5)
    save_path: if provided, saves results to Excel file

    Returns
    -------
    results[target][method] = {MAPE_%, CVRMSE_%, MaxAPE_%, Pearson_r, R2}
    """
    methods = methods or RegressionModel.SUPPORTED_METHODS
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(x.shape[1])]
    assert len(feature_names) == x.shape[1]
    if target_names is None:
        target_names = [f"Target_{i}" for i in range(y.shape[1])]
    assert len(target_names) == y.shape[1]

    results = {}
    for target_name, y_i in zip(target_names, y.T):
        # transpose 2 times to get the column as 1D array for each target
        results[target_name] = {}
        for method in methods:
            model = RegressionModel(method=method, max_iterations=max_iterations)
            model.target_name = target_name
            metrics = model.evaluate(x, y_i, cv_folds=cv_folds)
            results[target_name][method] = metrics

    if save_path:
        save_eval_to_json(results, filename=save_path)
    return results

def sanitize_sheet_name(name: str) -> str:
    """
    Sanitize sheet name for Excel compatibility.
    Excel sheet names cannot contain: [ ] : * ? / \
    Max length: 31 characters
    """
    # Replace invalid characters with underscore
    invalid_chars = r'[\[\]:*?/\\]'
    sanitized = re.sub(invalid_chars, '_', name)

    # Truncate to 31 characters (Excel limit)
    if len(sanitized) > 31:
        sanitized = sanitized[:31]

    # Ensure not empty
    if not sanitized:
        sanitized = "Sheet"

    return sanitized

def save_eval_to_json(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    filename: str = "regression_results.json"
) -> None:
    """
    Save regression results to JSON file.
    """
    assert filename.endswith(".json"), "To save JSON file, filename must end with .json"
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {filename}")

def load_eval_from_json(filename: str) -> Dict:
    """
    Load regression evaluation results from JSON file.
    """
    assert filename.endswith(".json"), "To load JSON file, filename must end with .json"
    with open(filename, 'r') as f:
        all_results = json.load(f)
    return all_results


def save_eval_to_excel(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    filename: str = "regression_results.xlsx"
) -> None:
    """
    Save regression results to Excel file with multiple sheets.
    Each sheet corresponds to one target variable.
    Rows = methods, Columns = metrics (MAPE_%, CVRMSE_%, MaxAPE_%, Pearson_r, R2)


    all_results format:
        {target1: {method1 : {metric1: value, ...}, method2: {...}}, target2: {...}, ...}
    """
    assert filename.endswith(".xlsx"), "To save excel file, filename must end with .xlsx"

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:

        for target_name, methods_dict in all_results.items():
            # Create DataFrame: rows=methods, columns=metrics
            df_data = []

            for method, metrics in methods_dict.items():
                row = {'Method': method}
                row.update(metrics)
                df_data.append(row)

            df = pd.DataFrame(df_data)

            # Ensure Method is first column
            cols = ['Method'] + [col for col in df.columns if col != 'Method']
            df = df[cols]

            # Sanitize sheet name for Excel compatibility
            sheet_name = sanitize_sheet_name(target_name)

            # Handle duplicate sheet names (if sanitization causes collision)
            original_sheet_name = sheet_name
            counter = 1
            while sheet_name in writer.sheets:
                suffix = f"_{counter}"
                sheet_name = original_sheet_name[:31-len(suffix)] + suffix
                counter += 1

            # Write to Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Auto-adjust column widths
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).map(len).max(),
                    len(str(col))
                ) + 2
                # Handle more than 26 columns (AA, AB, etc.)
                if idx < 26:
                    col_letter = chr(65 + idx)
                else:
                    col_letter = chr(65 + (idx // 26) - 1) + chr(65 + (idx % 26))
                worksheet.column_dimensions[col_letter].width = min(max_length, 50)

    print(f"Results saved to {filename}")