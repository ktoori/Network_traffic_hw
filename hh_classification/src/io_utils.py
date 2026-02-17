from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

import src.level_prepare.constants as const


def read_csv(path: Path) -> pd.DataFrame:
    """Читает CSV файл в датафрейм.

    Parameters
    ----------
    path
        Путь к CSV.

    Returns
    -------
    pd.DataFrame
        Датафрейм с данными.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return pd.read_csv(path, encoding=const.ENCODING_UTF8_SIG)


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Проверяет наличие обязательных колонок.

    Parameters
    ----------
    df
        Датафрейм для проверки.
    required
        Список обязательных колонок.

    Returns
    -------
    None
        Ничего не возвращает.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {', '.join(missing)}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Сохраняет датафрейм в CSV.

    Parameters
    ----------
    df
        Датафрейм для сохранения.
    path
        Путь сохранения.

    Returns
    -------
    None
        Ничего не возвращает.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding=const.ENCODING_UTF8_SIG)
