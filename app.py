"""
Entry point for hh.csv preprocessing pipeline.

Usage
----------
    python app.py path/to/hh.csv

The script processes the input CSV file and generates:
    - x_data.npy (feature matrix)
    - y_data.npy (target vector)
in the same directory as the input file.
"""

import argparse
import logging
from pathlib import Path

import config
from handlers import (
    CSVLoadHandler,
    InitialCleanupHandler,
    CleanTextHandler,
    GenderAgeHandler,
    SalaryHandler,
    CityHandler,
    OneHotTextHandler,
    ExperienceHandler,
    EducationHandler,
    CardinalityReducerHandler,
    FillMissingHandler,
    EncodeHandler,
    FeatureTargetSplitHandler,
    SaveNpyHandler,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
logger = logging.getLogger("PipelineRunner")


def build_pipeline() -> CSVLoadHandler:
    """
    Constructs the processing pipeline by chaining handlers.

    Returns
    -------
    CSVLoadHandler
        The first handler in the chain.
    """
    pipeline = CSVLoadHandler()

    (
        pipeline
        .set_next(InitialCleanupHandler())
        .set_next(CleanTextHandler())
        .set_next(GenderAgeHandler())
        .set_next(SalaryHandler())
        .set_next(CityHandler())
        .set_next(OneHotTextHandler("Занятость", config.EMPLOYMENT_PATTERNS))
        .set_next(OneHotTextHandler("График", config.SCHEDULE_PATTERNS))
        .set_next(ExperienceHandler())
        .set_next(EducationHandler())
        .set_next(CardinalityReducerHandler())
        .set_next(FillMissingHandler(target=config.TARGET_COLUMN))
        .set_next(EncodeHandler())
        .set_next(FeatureTargetSplitHandler(target=config.TARGET_COLUMN))
        .set_next(SaveNpyHandler())
    )

    return pipeline


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess hh.csv and generate x_data.npy and y_data.npy"
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the input hh.csv file.",
    )
    return parser.parse_args()


def main() -> int:
    """
    Main execution logic.
    """
    args = parse_arguments()
    csv_path: Path = args.csv_path

    if not csv_path.exists():
        logger.error("Input file not found: %s", csv_path)
        return 2

    pipeline = build_pipeline()

    try:
        logger.info("Starting pipeline for %s.", csv_path)
        # Start pipeline execution from the first handler
        pipeline.handle(csv_path)
        logger.info("Pipeline finished successfully.")
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
