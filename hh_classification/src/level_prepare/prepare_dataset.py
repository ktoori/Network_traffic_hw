from __future__ import annotations

import argparse
from pathlib import Path

import src.level_prepare.constants as const
import src.level_prepare.feature_cleaning as cleaning
import src.io_utils as io_utils
import src.level_prepare.level_labeling as labeling
import src.logging_utils as logging_utils
import src.level_prepare.plotting as plotting


def parse_args() -> argparse.Namespace:
    """Считывает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="PoC: разметка уровня junior/middle/senior и подготовка датасета."
    )
    parser.add_argument("x_csv", type=Path, help="Путь к X_data.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    """Запускает подготовку датасета и сохранение артефактов."""
    args = parse_args()
    logger = logging_utils.setup_logger()

    df_raw = io_utils.read_csv(args.x_csv)
    io_utils.validate_columns(df_raw, [const.COL_EXP_MONTHS, const.COL_IT_DIRECTION])

    df_it = labeling.filter_it_direction(df_raw)
    df_labeled = labeling.add_level(df_it)
    df_labeled = labeling.drop_unknown_levels(df_labeled)

    plotting.plot_class_balance(df_labeled[const.LEVEL_COL], args.out_dir)

    x_clean = cleaning.build_x_clean(df_labeled)
    y_level = cleaning.build_y(df_labeled)

    io_utils.save_csv(x_clean, args.out_dir / const.CSV_X_CLEAN)
    io_utils.save_csv(y_level, args.out_dir / const.CSV_Y_LEVEL)

    df_for_debug = df_labeled.drop(columns=[const.TITLE_TEXT_COL], errors="ignore")
    io_utils.save_csv(df_for_debug, args.out_dir / const.CSV_DF_WITH_LEVEL)

    logger.info("Готово. Результаты сохранены в папку: %s", args.out_dir)


if __name__ == "__main__":
    main()
