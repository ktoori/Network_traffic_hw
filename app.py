"""
Inference CLI for salary prediction.

Required interface (adapted):
-----------------------------
python app.py path/to/x_data.npy

Prints
------
JSON array of salaries in RUB (floats).

Model file:
-----------
resources/model.joblib
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from hh_salary_model.io_utils import DatasetFormatError, load_x
from hh_salary_model.model import SalaryRegressor

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("App")


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with:
        - x_path: Path to x_data.npy
        - model_path: Path to saved model
    """
    parser = argparse.ArgumentParser(
        description="Predict salaries from x_data.npy using a trained regression model."
    )
    parser.add_argument(
        "x_path",
        type=Path,
        help="Path to x_data.npy (output of preprocessing pipeline)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("resources/model.joblib"),
        help="Path to trained model (default: resources/model.joblib)",
    )
    return parser.parse_args()


def main() -> int:
    """
    Entry point for inference.

    Returns
    -------
    int
        Exit code:
        - 0 on success
        - 1 on inference error
        - 2 on input/model error
    """
    args = parse_args()

    try:
        model = SalaryRegressor.load(args.model_path)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error(
            "Model not found. Train it first:\n"
            "python grid_train.py path/to/x_data.npy path/to/y_data.npy"
        )
        return 2
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return 2

    try:
        x = load_x(args.x_path)
        predictions = model.predict(x)
    except (FileNotFoundError, DatasetFormatError) as exc:
        logger.error("%s", exc)
        return 2
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        return 1

    # Required output: list of float salaries in RUB
    result = [float(value) for value in predictions.tolist()]
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
