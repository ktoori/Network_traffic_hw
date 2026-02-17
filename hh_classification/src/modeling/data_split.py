from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

import src.modeling.constants as mconst


@dataclass(frozen=True)
class SplitData:
    """Результат разбиения данных на обучающую и тестовую выборки."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_train_test(x: pd.DataFrame, y: pd.Series) -> SplitData:
    """Делит данные на train/test со стратификацией.

    Parameters
    ----------
    x
        Признаки.
    y
        Целевая переменная.

    Returns
    -------
    SplitData
        Разбиение train/test.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=mconst.TEST_SIZE,
        random_state=mconst.RANDOM_STATE,
        stratify=y,
    )
    return SplitData(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
