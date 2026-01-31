"""
Sklearn-based regression model for salary prediction.

We intentionally use scikit-learn models (as in lectures):
- sklearn.linear_model.LinearRegression / Ridge / Lasso
- sklearn.pipeline.Pipeline
- sklearn.preprocessing.RobustScaler
- sklearn.compose.TransformedTargetRegressor

Model is saved into resources/ as a single artifact (joblib),
so inference is reproducible and matches training preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


Penalty = Literal["none", "l2", "l1"]


class ModelNotFittedError(RuntimeError):
    """Raised when predict() is called on a model that hasn't been fitted."""


@dataclass
class SalaryRegressor:
    """
    Salary regression wrapper around scikit-learn.

    Parameters
    ----------
    alpha : float
        Regularization strength for Ridge/Lasso. If alpha == 0, LinearRegression is used.
    penalty : {"none", "l2", "l1"}
        Regularization type:
        - "none": LinearRegression (OLS)
        - "l2": Ridge (L2)
        - "l1": Lasso (L1)
    log_target : bool
        If True, fit model on log1p(y) and return predictions via expm1.
        Often helps with heavy-tailed salary distributions.
    """
    alpha: float = 1e-6
    penalty: Penalty = "l2"
    log_target: bool = True

    _pipeline: Pipeline | None = None

    def _build_pipeline(self) -> Pipeline:
        """
        Build sklearn Pipeline:
        X -> RobustScaler -> (LinearRegression/Ridge/Lasso)
        optionally wrapped with TransformedTargetRegressor for log-target.
        """
        if self.penalty == "none" or float(self.alpha) == 0.0:
            base_estimator = LinearRegression()
        elif self.penalty == "l2":
            base_estimator = Ridge(alpha=float(self.alpha))
        elif self.penalty == "l1":
            base_estimator = Lasso(
                alpha=float(self.alpha),
                max_iter=10_000,
                tol=1e-3,
                selection="random",
            )
        else:
            raise ValueError("penalty must be one of: none, l2, l1")

        x_pipe = Pipeline(
            steps=[
                ("scaler", RobustScaler(with_centering=True, with_scaling=True)),
                ("reg", base_estimator),
            ]
        )

        if self.log_target:
            # y' = log1p(y), inverse: expm1(y')
            return Pipeline(
                steps=[
                    (
                        "model",
                        TransformedTargetRegressor(
                            regressor=x_pipe,
                            func=np.log1p,
                            inverse_func=np.expm1,
                            check_inverse=False,
                        ),
                    )
                ]
            )

        return x_pipe

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SalaryRegressor":
        """
        Fit model.

        Raises
        ------
        ValueError
            If shapes are inconsistent or data contains NaN/inf.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)

        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        if x_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got ndim={x_arr.ndim}")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"X and y size mismatch: {x_arr.shape[0]} != {y_arr.shape[0]}")

        if not np.isfinite(x_arr).all():
            raise ValueError("X contains NaN/inf.")
        if not np.isfinite(y_arr).all():
            raise ValueError("y contains NaN/inf.")
        if self.log_target and np.any(y_arr < 0):
            raise ValueError("y contains negative values, cannot use log1p.")

        self._pipeline = self._build_pipeline()
        self._pipeline.fit(x_arr, y_arr)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict salaries (RUB).

        Returns
        -------
        np.ndarray
            1D array of non-negative predictions.
        """
        if self._pipeline is None:
            raise ModelNotFittedError("Model is not fitted.")

        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        if x_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got ndim={x_arr.ndim}")
        if not np.isfinite(x_arr).all():
            raise ValueError("X contains NaN/inf.")

        pred = self._pipeline.predict(x_arr)
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)

        # Salary cannot be negative
        return np.maximum(pred, 0.0)

    def save(self, path: Path) -> None:
        """
        Save fitted model to disk (joblib).
        """
        if self._pipeline is None:
            raise ModelNotFittedError("Nothing to save: model is not fitted.")

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "alpha": float(self.alpha),
                "penalty": str(self.penalty),
                "log_target": bool(self.log_target),
                "pipeline": self._pipeline,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "SalaryRegressor":
        """
        Load model from disk (joblib).
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        obj = joblib.load(path)
        model = cls(
            alpha=float(obj["alpha"]),
            penalty=str(obj.get("penalty", "l2")),
            log_target=bool(obj["log_target"]),
        )
        model._pipeline = obj["pipeline"]
        return model
