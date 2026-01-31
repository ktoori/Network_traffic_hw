"""
I/O helpers for loading NumPy datasets (x_data.npy / y_data.npy) with validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


class DatasetFormatError(ValueError):
    """Raised when x/y datasets have invalid format, shapes or values."""


def load_x(path: Path) -> np.ndarray:
    """
    Load and validate feature matrix X from a .npy file.

    Parameters
    ----------
    path : pathlib.Path
        Path to x_data.npy.

    Returns
    -------
    numpy.ndarray
        2D array of shape (n_samples, n_features), dtype float64.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    DatasetFormatError
        If X is not numeric, has invalid shape, or contains NaN/inf.
    """
    if not path.exists():
        raise FileNotFoundError(f"X file not found: {path}")

    x = np.load(path, allow_pickle=False)

    if x.ndim == 1:
        # Single row case: (n_features,) -> (1, n_features)
        x = x.reshape(1, -1)

    if x.ndim != 2:
        raise DatasetFormatError(f"X must be 2D array, got ndim={x.ndim}")

    try:
        x = x.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise DatasetFormatError("X must be convertible to float64.") from exc

    if not np.isfinite(x).all():
        raise DatasetFormatError("X contains NaN and/or infinite values.")

    if x.shape[0] < 1 or x.shape[1] < 1:
        raise DatasetFormatError(f"X has invalid shape: {x.shape}")

    return x


def load_y(path: Path) -> np.ndarray:
    """
    Load and validate target vector y from a .npy file.

    Parameters
    ----------
    path : pathlib.Path
        Path to y_data.npy.

    Returns
    -------
    numpy.ndarray
        1D array of shape (n_samples,), dtype float64.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    DatasetFormatError
        If y is not numeric, has invalid shape, or contains NaN/inf.
    """
    if not path.exists():
        raise FileNotFoundError(f"y file not found: {path}")

    y = np.load(path, allow_pickle=False)

    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)

    if y.ndim != 1:
        raise DatasetFormatError(f"y must be 1D array, got ndim={y.ndim}")

    try:
        y = y.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise DatasetFormatError("y must be convertible to float64.") from exc

    if not np.isfinite(y).all():
        raise DatasetFormatError("y contains NaN and/or infinite values.")

    if y.shape[0] < 1:
        raise DatasetFormatError("y must have at least 1 element.")

    return y


def load_xy(x_path: Path, y_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and validate X and y together, ensuring compatible shapes.

    Parameters
    ----------
    x_path : pathlib.Path
        Path to x_data.npy.
    y_path : pathlib.Path
        Path to y_data.npy.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (X, y)

    Raises
    ------
    DatasetFormatError
        If X and y have mismatched number of rows.
    """
    x = load_x(x_path)
    y = load_y(y_path)

    if x.shape[0] != y.shape[0]:
        raise DatasetFormatError(
            f"X and y length mismatch: X has {x.shape[0]} rows, y has {y.shape[0]}."
        )

    return x, y
