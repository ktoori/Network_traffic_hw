from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import src.level_prepare.constants as const


def plot_class_balance(y: pd.Series, out_dir: Path) -> None:
    """Строит и сохраняет графики баланса классов."""
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = y.value_counts()
    shares = (counts / counts.sum() * 100).round(2)

    _save_bar(counts, out_dir / const.PLOT_BAR)
    _save_pie(counts, out_dir / const.PLOT_PIE)
    _save_text(counts, shares, out_dir / const.BALANCE_TXT)


def _save_bar(counts: pd.Series, path: Path) -> None:
    plt.figure()
    counts.plot(kind="bar")
    plt.title("Баланс классов: количество резюме по уровням")
    plt.xlabel("Уровень")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _save_pie(counts: pd.Series, path: Path) -> None:
    plt.figure()
    counts.plot(kind="pie", autopct="%1.1f%%")
    plt.title("Баланс классов: доли уровней")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _save_text(counts: pd.Series, shares: pd.Series, path: Path) -> None:
    text = "Количество:\n"
    text += counts.to_string()
    text += "\n\nДоли (%):\n"
    text += shares.to_string()
    path.write_text(text, encoding="utf-8")
