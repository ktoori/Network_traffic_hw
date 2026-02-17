from __future__ import annotations

from typing import Optional

import pandas as pd

import src.level_prepare.constants as const


def filter_it_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только строки с IT_направление == 1.

    Parameters
    ----------
    df
        Исходный датафрейм.

    Returns
    -------
    pd.DataFrame
        Отфильтрованный датафрейм (только IT).
    """
    return df[df[const.COL_IT_DIRECTION] == 1].copy()


def get_prefixed_columns(df: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    """Возвращает колонки, начинающиеся с заданных префиксов."""
    return [col for col in df.columns if col.startswith(prefixes)]


def is_one(value: object) -> bool:
    """Проверяет, что значение можно интерпретировать как 1 (one-hot)."""
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def to_float(value: object) -> Optional[float]:
    """Безопасно приводит значение к float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_title_text(row: pd.Series, title_cols: list[str]) -> str:
    """Собирает строку из активных one-hot колонок должности."""
    active: list[str] = []
    for col in title_cols:
        if is_one(row.get(col, 0)):
            active.append(col)
    return " ".join(active)


def infer_level(title_text: str, exp_months: Optional[float]) -> Optional[str]:
    """Определяет уровень по названию должности и стажу.

    Правила (PoC):
    1) Если в названии есть явные маркеры junior/senior — используем их.
    2) Иначе по стажу (месяцы): <= 24 junior, 25..60 middle, > 60 senior.
    """
    if title_text:
        if const.JUNIOR_TITLE_RE.search(title_text):
            return const.LEVEL_JUNIOR
        if const.SENIOR_TITLE_RE.search(title_text):
            return const.LEVEL_SENIOR

    if exp_months is None:
        return None

    if exp_months <= 24:
        return const.LEVEL_JUNIOR
    if exp_months <= 60:
        return const.LEVEL_MIDDLE
    return const.LEVEL_SENIOR


def add_level(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонку уровня (junior/middle/senior).

    Parameters
    ----------
    df
        Датафрейм с признаками.

    Returns
    -------
    pd.DataFrame
        Датафрейм со столбцами __title_text__ и level.
    """
    title_cols = get_prefixed_columns(df, const.TITLE_PREFIXES)
    if not title_cols:
        raise ValueError(
            "Не найдены колонки должности. Ожидались колонки, начинающиеся с: "
            f"{', '.join(const.TITLE_PREFIXES)}"
        )

    result = df.copy()

    result[const.TITLE_TEXT_COL] = result.apply(
        lambda row: build_title_text(row, title_cols),
        axis=1,
    )

    result[const.LEVEL_COL] = result.apply(
        lambda row: infer_level(
            title_text=str(row.get(const.TITLE_TEXT_COL, "")),
            exp_months=to_float(row.get(const.COL_EXP_MONTHS)),
        ),
        axis=1,
    )

    return result


def drop_unknown_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет строки, где уровень определить не удалось."""
    return df.dropna(subset=[const.LEVEL_COL]).copy()
