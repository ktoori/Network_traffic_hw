from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import src.modeling.constants as mconst


@dataclass(frozen=True)
class MetricsSummary:
    """Сводка метрик для сравнения экспериментов."""

    name: str
    accuracy: float
    f1_macro: float
    f1_weighted: float


def evaluate_model(
    name: str,
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir: Path,
) -> MetricsSummary:
    """Считает метрики и сохраняет отчёт и confusion matrix.

    Parameters
    ----------
    name
        Название эксперимента.
    model
        Обученный пайплайн.
    x_test
        Тестовые признаки.
    y_test
        Тестовая целевая переменная.
    out_dir
        Папка для сохранения артефактов.

    Returns
    -------
    MetricsSummary
        Ключевые метрики эксперимента.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(x_test)
    labels = sorted(y_test.unique().tolist())

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        digits=4,
        zero_division=0,
    )

    report_path = out_dir / mconst.REPORT_TXT
    report_path.write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    _save_confusion_matrix(cm, labels, out_dir / mconst.CONFUSION_MATRIX_PNG)

    # берём значения из отчёта повторно (надёжнее всего парсить dict)
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    accuracy = float(report_dict["accuracy"])
    f1_macro = float(report_dict["macro avg"]["f1-score"])
    f1_weighted = float(report_dict["weighted avg"]["f1-score"])

    return MetricsSummary(name=name, accuracy=accuracy, f1_macro=f1_macro, f1_weighted=f1_weighted)


def _save_confusion_matrix(cm, labels: list[str], path: Path) -> None:
    plt.figure()
    plt.imshow(cm)
    plt.title("Матрица ошибок (confusion matrix)")
    plt.xlabel("Предсказание")
    plt.ylabel("Истина")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
