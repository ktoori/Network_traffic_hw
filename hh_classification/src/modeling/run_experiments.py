from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import src.io_utils as io_utils
import src.logging_utils as logging_utils
import src.modeling.constants as mconst
import src.modeling.data_split as data_split
import src.modeling.evaluation as evaluation
import src.modeling.train as train


def parse_args() -> argparse.Namespace:
    """Считывает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="PoC: обучение классификатора уровня junior/middle/senior."
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts_model"))
    return parser.parse_args()


def load_xy(artifacts_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Загружает X_clean.csv и y_level.csv."""
    x_path = artifacts_dir / "X_clean.csv"
    y_path = artifacts_dir / "y_level.csv"

    x = io_utils.read_csv(x_path)
    y_df = io_utils.read_csv(y_path)

    if "level" not in y_df.columns:
        raise ValueError("В файле y_level.csv отсутствует колонка level.")

    y = y_df["level"].astype(str)
    return x, y


def save_metrics_table(summaries: list[evaluation.MetricsSummary], out_dir: Path) -> None:
    """Сохраняет сводную таблицу метрик."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "эксперимент": s.name,
            "accuracy": s.accuracy,
            "f1_macro": s.f1_macro,
            "f1_weighted": s.f1_weighted,
        }
        for s in summaries
    ]
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(out_dir / mconst.METRICS_CSV, index=False, encoding="utf-8-sig")


def main() -> None:
    """Запускает серию экспериментов и сохраняет результаты."""
    args = parse_args()
    logger = logging_utils.setup_logger("hh_level_model")

    x, y = load_xy(args.artifacts_dir)
    split = data_split.split_train_test(x, y)

    experiments = [
        (mconst.MODEL_BASELINE, None, False),
        (mconst.MODEL_BALANCED, "balanced", False),
        (mconst.MODEL_OVERSAMPLE, None, True),
    ]

    summaries: list[evaluation.MetricsSummary] = []

    for exp_name, class_weight, do_oversample in experiments:
        exp_dir = args.out_dir / exp_name
        logger.info("Эксперимент: %s", exp_name)

        trained = train.train_model(
            name=exp_name,
            x_train=split.x_train,
            y_train=split.y_train,
            class_weight=class_weight,
            do_oversample=do_oversample,
        )

        summary = evaluation.evaluate_model(
            name=exp_name,
            model=trained.pipeline,
            x_test=split.x_test,
            y_test=split.y_test,
            out_dir=exp_dir,
        )
        summaries.append(summary)

        logger.info(
            "Готово: accuracy=%.4f, f1_macro=%.4f, f1_weighted=%.4f",
            summary.accuracy,
            summary.f1_macro,
            summary.f1_weighted,
        )

    save_metrics_table(summaries, args.out_dir)
    logger.info("Сводная таблица метрик сохранена: %s", args.out_dir / mconst.METRICS_CSV)


if __name__ == "__main__":
    main()
