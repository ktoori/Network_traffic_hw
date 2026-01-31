"""
Train salary regression using scikit-learn (as in lectures).

Key points:
- train_test_split for hold-out evaluation (lecture requirement)
- L2 regularization: Ridge
- L1 regularization: Lasso
- Optional: no regularization (OLS) via LinearRegression
- RobustScaler for X
- Optional log1p transform for target via TransformedTargetRegressor
- Saves final model to resources/model.joblib

Usage:
    python grid_train.py path/to/x_data.npy path/to/y_data.npy
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import train_test_split

from hh_salary_model.io_utils import DatasetFormatError, load_xy
from hh_salary_model.metrics import regression_report
from hh_salary_model.model import Penalty, SalaryRegressor

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("GridTrain")


@dataclass(frozen=True)
class TrainResult:
    """Container for a single (penalty, alpha) evaluation."""
    penalty: Penalty
    alpha: float
    rmse: float
    mae: float
    r2: float


def parse_args() -> argparse.Namespace:
    """
        Parse command-line arguments for training script.

        Returns
        -------
        argparse.Namespace
            Parsed arguments including:
            - paths to x and y datasets
            - random seed
            - test split size
            - grid of regularization parameters
            - output path for the trained model
        """

    parser = argparse.ArgumentParser(
        description="Train salary regression (Linear/Ridge/Lasso) and save to resources/."
    )
    parser.add_argument("x_path", type=Path, help="Path to x_data.npy")
    parser.add_argument("y_path", type=Path, help="Path to y_data.npy")

    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size (default: 0.2)")
    parser.add_argument(
        "--penalties",
        type=str,
        default="l2,l1,none",
        help='Comma-separated: "l2,l1,none" (default: l2,l1,none)',
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="0,1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,1,10,100",
        help="Comma-separated list of alphas. Use 0 for OLS (or to compare).",
    )
    parser.add_argument(
        "--no-log-target",
        action="store_true",
        help="Disable log1p target transform (enabled by default).",
    )
    parser.add_argument(
        "--model-out`,",
        dest="model_out",
        type=Path,
        default=Path("resources/model.joblib"),
        help="Path to save model (default: resources/model.joblib)",
    )

    return parser.parse_args()


def parse_csv_floats(s: str) -> list[float]:
    """
       Parse comma-separated string into a list of floats.

       Parameters
       ----------
       s : str
           Comma-separated string of float values.

       Returns
       -------
       list[float]
           List of parsed float values.

       Raises
       ------
       ValueError
           If the resulting list is empty or contains invalid values.
       """
    values: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("Alpha grid is empty.")
    return values


def parse_penalties(s: str) -> list[Penalty]:
    """
        Parse comma-separated string into a list of regularization types.

        Parameters
        ----------
        s : str
            Comma-separated string with penalty names.
            Allowed values: "none", "l1", "l2".

        Returns
        -------
        list[Penalty]
            List of parsed penalty identifiers.

        Raises
        ------
        ValueError
            If an unknown penalty value is encountered or the list is empty.
        """
    penalties: list[Penalty] = []
    for part in s.split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p not in {"none", "l1", "l2"}:
            raise ValueError('penalties must be subset of: "none,l1,l2"')
        penalties.append(p)  # type: ignore[arg-type]
    if not penalties:
        raise ValueError("Penalties list is empty.")
    return penalties


def baseline_median(y_true: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evaluate a simple baseline model that always predicts
    the median value of the training target.

    This baseline is used to compare the regression model
    against a trivial constant predictor.

    Parameters
    ----------
    y_true : np.ndarray
        Target values from the training set.
    y_test : np.ndarray
        Target values from the test set.
    """
    y_med = float(np.median(y_true))
    y_pred = np.full_like(y_test, fill_value=y_med, dtype=np.float64)
    rep = regression_report(y_test, y_pred)
    logger.info("Baseline (median): RMSE=%.3f, MAE=%.3f, R^2=%.5f", rep.rmse, rep.mae, rep.r2)


def evaluate_grid(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    penalties: Iterable[Penalty],
    alphas: Iterable[float],
    log_target: bool,
) -> TrainResult:
    """
    Perform grid search over regularization type and strength.

    For each combination of penalty type (none, L1, L2) and
    regularization parameter alpha, the model is trained on the
    training set and evaluated on the test set.

    The best configuration is selected based on minimal RMSE.

    Parameters
    ----------
    x_train : np.ndarray
        Feature matrix for training.
    y_train : np.ndarray
        Target values for training.
    x_test : np.ndarray
        Feature matrix for evaluation.
    y_test : np.ndarray
        Target values for evaluation.
    penalties : Iterable[Penalty]
        Regularization types to evaluate.
    alphas : Iterable[float]
        Regularization strengths.
    log_target : bool
        Whether to apply log1p transformation to the target variable.

    Returns
    -------
    TrainResult
        Best configuration according to RMSE metric.
    """
    best: TrainResult | None = None

    for penalty in penalties:
        for alpha in alphas:
            if penalty == "l1" and alpha < 1e-4 and alpha != 0.0:
                continue

            # If penalty is "none", alpha does not matter (but we allow alpha=0 explicitly)
            model = SalaryRegressor(alpha=float(alpha), penalty=penalty, log_target=log_target)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            rep = regression_report(y_test, y_pred)
            result = TrainResult(
                penalty=penalty,
                alpha=float(alpha),
                rmse=rep.rmse,
                mae=rep.mae,
                r2=rep.r2,
            )

            logger.info(
                "penalty=%s alpha=%g -> RMSE=%.3f MAE=%.3f R^2=%.5f",
                result.penalty,
                result.alpha,
                result.rmse,
                result.mae,
                result.r2,
            )

            if best is None or result.rmse < best.rmse:
                best = result

    if best is None:
        raise RuntimeError("Grid search produced no results.")
    return best


def main() -> int:
    """
        Entry point for model training.

        Loads datasets, performs train/test split, evaluates models
        with different regularization strategies, selects the best
        configuration, retrains the model on the full dataset and
        saves it to disk.

        Returns
        -------
        int
            Exit code:
            - 0 on successful training
            - non-zero value on error
        """
    args = parse_args()

    try:
        x, y = load_xy(args.x_path, args.y_path)
    except (FileNotFoundError, DatasetFormatError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    log_target = not args.no_log_target

    try:
        penalties = parse_penalties(args.penalties)
        alphas = parse_csv_floats(args.alphas)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        shuffle=True,
    )

    logger.info("Split shapes: X_train=%s X_test=%s", x_train.shape, x_test.shape)

    baseline_median(y_train, y_test)

    best = evaluate_grid(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        penalties=penalties,
        alphas=alphas,
        log_target=log_target,
    )

    logger.info(
        "Best config: penalty=%s alpha=%g | RMSE=%.3f MAE=%.3f R^2=%.5f",
        best.penalty,
        best.alpha,
        best.rmse,
        best.mae,
        best.r2,
    )

    # Refit on ALL data with best hyperparams and save
    final_model = SalaryRegressor(alpha=best.alpha, penalty=best.penalty, log_target=log_target)
    final_model.fit(x, y)
    final_model.save(args.model_out)

    logger.info("Model saved to: %s", args.model_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
