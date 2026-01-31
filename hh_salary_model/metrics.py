"""
Metrics helpers for regression tasks.

This module uses scikit-learn metrics to compute:
- MAE
- MSE
- RMSE
- R^2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class RegressionReport:
    """Container for common regression metrics."""
    mae: float
    mse: float
    rmse: float
    r2: float


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionReport:
    """
    Compute regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values, shape (n_samples,).
    y_pred : np.ndarray
        Predicted target values, shape (n_samples,).

    Returns
    -------
    RegressionReport
        MAE, MSE, RMSE, and R^2 values.
    """
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if yt.shape[0] != yp.shape[0]:
        raise ValueError(
            f"y_true and y_pred length mismatch: {yt.shape[0]} != {yp.shape[0]}"
        )

    mae = float(mean_absolute_error(yt, yp))
    mse = float(mean_squared_error(yt, yp))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(yt, yp))

    return RegressionReport(mae=mae, mse=mse, rmse=rmse, r2=r2)
