from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import src.modeling.constants as mconst


@dataclass(frozen=True)
class TrainedModel:
    """Модель и её название для эксперимента."""

    name: str
    pipeline: Pipeline


def build_logreg_pipeline(class_weight: str | None) -> Pipeline:
    """Создаёт пайплайн логистической регрессии.

    Parameters
    ----------
    class_weight
        Веса классов. Например: None или "balanced".

    Returns
    -------
    Pipeline
        Пайплайн (масштабирование + логрег).
    """
    model = LogisticRegression(
        solver="saga",
        max_iter=10_000,
        class_weight=class_weight,
        n_jobs=None,
        random_state=mconst.RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("model", model),
        ]
    )


def oversample_training_set(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Делает oversampling только на train, выравнивая классы до максимума.

    Parameters
    ----------
    x_train
        Обучающие признаки.
    y_train
        Обучающая целевая переменная.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Новый X_train и y_train после oversampling.
    """
    df_train = x_train.copy()
    df_train["_y"] = y_train.values

    class_counts = df_train["_y"].value_counts()
    max_count = int(class_counts.max())

    sampled_parts: list[pd.DataFrame] = []
    for cls, cnt in class_counts.items():
        part = df_train[df_train["_y"] == cls]
        sampled = part.sample(
            n=max_count,
            replace=True,
            random_state=mconst.RANDOM_STATE,
        )
        sampled_parts.append(sampled)

    df_resampled = pd.concat(sampled_parts, axis=0).sample(
        frac=1.0,
        random_state=mconst.RANDOM_STATE,
    )

    y_resampled = df_resampled["_y"].copy()
    x_resampled = df_resampled.drop(columns=["_y"]).copy()
    return x_resampled, y_resampled


def train_model(
    name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight: str | None,
    do_oversample: bool,
) -> TrainedModel:
    """Обучает модель с заданными настройками.

    Parameters
    ----------
    name
        Название эксперимента.
    x_train
        Обучающие признаки.
    y_train
        Обучающая целевая переменная.
    class_weight
        Веса классов (None или "balanced").
    do_oversample
        Делать ли oversampling на train.

    Returns
    -------
    TrainedModel
        Обученная модель.
    """
    x_fit = x_train
    y_fit = y_train

    if do_oversample:
        x_fit, y_fit = oversample_training_set(x_train, y_train)

    pipeline = build_logreg_pipeline(class_weight=class_weight)
    pipeline.fit(x_fit, y_fit)
    return TrainedModel(name=name, pipeline=pipeline)
