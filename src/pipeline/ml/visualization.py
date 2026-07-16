"""Visualization for the ML module.

Publication-quality evaluation figures under ``reports/ml/figures/``:
per-model ROC / PR / confusion / calibration / lift / gain / feature-
importance plots, plus cross-model comparison bar charts. Uses the same
matplotlib(Agg)+seaborn stack and styling conventions as the EDA module.
Every plotting call is best-effort: a failure is logged and skipped so a bad
figure never kills the pipeline.
"""
from __future__ import annotations

import os
from typing import Callable

import matplotlib
matplotlib.use("Agg")  # noqa: E402 - headless backend before pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, precision_recall_curve, roc_curve)

from ingestion.logging_config import get_logger

log = get_logger("ml.viz")


class MLVisualizer:
    """Save evaluation figures for trained models."""

    def __init__(self, figures_dir: str = "reports/ml/figures",
                 dpi: int = 150, style: str = "whitegrid",
                 palette: str = "deep") -> None:
        self.figures_dir = figures_dir
        self.dpi = dpi
        os.makedirs(self.figures_dir, exist_ok=True)
        sns.set_theme(style=style, palette=palette)
        self.saved: list[str] = []

    # ── plumbing ──────────────────────────────────────────────────────────────
    def _save(self, fig: plt.Figure, name: str) -> str | None:
        path = os.path.join(self.figures_dir, f"{name}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        self.saved.append(path)
        log.info("figure saved: %s", path)
        return path

    def _plot(self, name: str, fn: Callable[[plt.Axes], None],
              figsize: tuple[float, float] = (7, 5)) -> str | None:
        """Run one plotting callback best-effort."""
        try:
            fig, ax = plt.subplots(figsize=figsize)
            fn(ax)
            return self._save(fig, name)
        except Exception as exc:  # noqa: BLE001 - plotting is best-effort
            plt.close("all")
            log.warning("figure '%s' failed: %s", name, exc)
            return None

    # ── per-model figures ─────────────────────────────────────────────────────
    def roc_curve(self, name: str, y_true: np.ndarray,
                  y_proba: np.ndarray) -> str | None:
        def draw(ax: plt.Axes) -> None:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr, tpr):.4f}")
            ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
            ax.set(xlabel="False positive rate", ylabel="True positive rate",
                   title=f"ROC Curve — {name}")
            ax.legend(loc="lower right")
        return self._plot(f"roc_{name}", draw)

    def pr_curve(self, name: str, y_true: np.ndarray,
                 y_proba: np.ndarray) -> str | None:
        def draw(ax: plt.Axes) -> None:
            prec, rec, _ = precision_recall_curve(y_true, y_proba)
            ax.plot(rec, prec, lw=2, label=f"AUC = {auc(rec, prec):.4f}")
            ax.axhline(np.mean(y_true), ls="--", c="grey", lw=1,
                       label="baseline")
            ax.set(xlabel="Recall", ylabel="Precision",
                   title=f"Precision-Recall Curve — {name}")
            ax.legend(loc="upper right")
        return self._plot(f"pr_{name}", draw)

    def confusion(self, name: str,
                  matrix: list[list[int]]) -> str | None:
        def draw(ax: plt.Axes) -> None:
            sns.heatmap(np.asarray(matrix), annot=True, fmt="d",
                        cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["No crisis", "Crisis"],
                        yticklabels=["No crisis", "Crisis"])
            ax.set(xlabel="Predicted", ylabel="Actual",
                   title=f"Confusion Matrix — {name}")
        return self._plot(f"confusion_{name}", draw, figsize=(6, 5))

    def calibration(self, name: str, y_true: np.ndarray,
                    y_proba: np.ndarray) -> str | None:
        def draw(ax: plt.Axes) -> None:
            frac_pos, mean_pred = calibration_curve(y_true, y_proba,
                                                    n_bins=10,
                                                    strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", lw=2, label=name)
            ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1,
                    label="perfectly calibrated")
            ax.set(xlabel="Mean predicted probability",
                   ylabel="Fraction of positives",
                   title=f"Calibration Curve — {name}")
            ax.legend(loc="upper left")
        return self._plot(f"calibration_{name}", draw)

    def lift_chart(self, name: str, y_true: np.ndarray,
                   y_proba: np.ndarray) -> str | None:
        def draw(ax: plt.Axes) -> None:
            deciles, lifts = self._decile_lift(y_true, y_proba)
            ax.bar(deciles, lifts, width=0.07)
            ax.axhline(1.0, ls="--", c="grey", lw=1, label="baseline")
            ax.set(xlabel="Population fraction (sorted by score)",
                   ylabel="Lift", title=f"Lift Chart — {name}")
            ax.legend()
        return self._plot(f"lift_{name}", draw)

    def gain_chart(self, name: str, y_true: np.ndarray,
                   y_proba: np.ndarray) -> str | None:
        def draw(ax: plt.Axes) -> None:
            frac, gains = self._cumulative_gain(y_true, y_proba)
            ax.plot(frac, gains, lw=2, label=name)
            ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1,
                    label="baseline")
            ax.set(xlabel="Population fraction (sorted by score)",
                   ylabel="Fraction of positives captured",
                   title=f"Cumulative Gain Chart — {name}")
            ax.legend(loc="lower right")
        return self._plot(f"gain_{name}", draw)

    def feature_importance(self, name: str, importance: pd.DataFrame,
                           top_n: int = 20) -> str | None:
        if importance is None or importance.empty:
            return None
        top = importance.head(top_n)

        def draw(ax: plt.Axes) -> None:
            sns.barplot(data=top, x="importance", y="feature", ax=ax)
            ax.set(title=f"Feature Importance — {name}",
                   xlabel="Importance", ylabel="")
        return self._plot(f"importance_{name}", draw,
                          figsize=(8, max(4, 0.32 * len(top))))

    # ── comparison figures ────────────────────────────────────────────────────
    def comparison_bars(self, leaderboard: pd.DataFrame,
                        metrics: list[str]) -> list[str]:
        paths = []
        for metric in metrics:
            if metric not in leaderboard.columns:
                continue

            def draw(ax: plt.Axes, m: str = metric) -> None:
                data = leaderboard.sort_values(m, ascending=False)
                sns.barplot(data=data, x=m, y="model", ax=ax)
                ax.set(title=f"Model Comparison — {m}", xlabel=m, ylabel="")
            path = self._plot(f"comparison_{metric}", draw,
                              figsize=(8, max(4, 0.4 * len(leaderboard))))
            if path:
                paths.append(path)
        return paths

    # ── math helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _decile_lift(y_true: np.ndarray,
                     y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(-np.asarray(y_proba))
        y = np.asarray(y_true)[order]
        base = y.mean() if y.mean() > 0 else 1e-9
        fracs = np.linspace(0.1, 1.0, 10)
        lifts = np.array([y[:max(1, int(f * len(y)))].mean() / base
                          for f in fracs])
        return fracs, lifts

    @staticmethod
    def _cumulative_gain(y_true: np.ndarray,
                         y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(-np.asarray(y_proba))
        y = np.asarray(y_true)[order]
        total = y.sum() if y.sum() > 0 else 1
        gains = np.concatenate([[0], np.cumsum(y) / total])
        frac = np.linspace(0, 1, len(gains))
        return frac, gains
