from __future__ import annotations

import pandas as pd

import src.level_prepare.constants as const
import src.level_prepare.level_labeling as labeling


def build_x_clean(df_with_level: pd.DataFrame) -> pd.DataFrame:
    """Удаляет утечки и шумные группы признаков.

    Parameters
    ----------
    df_with_level
        Датафрейм со столбцом level и техническим текстом должности.

    Returns
    -------
    pd.DataFrame
        Очищенные признаки X.
    """
    leaky_cols = labeling.get_prefixed_columns(df_with_level, const.DROP_PREFIXES_LEAKY_OR_NOISY)
    drop_cols = set(leaky_cols) | {const.LEVEL_COL, const.TITLE_TEXT_COL}
    existing = [col for col in drop_cols if col in df_with_level.columns]
    return df_with_level.drop(columns=existing).copy()


def build_y(df_with_level: pd.DataFrame) -> pd.DataFrame:
    """Возвращает целевую переменную y (level) отдельным датафреймом."""
    return df_with_level[[const.LEVEL_COL]].copy()
