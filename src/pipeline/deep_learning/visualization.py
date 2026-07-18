"""Visualization for the Deep Learning module.

Extends :class:`pipeline.ml.visualization.MLVisualizer` — reusing its ROC /
PR / confusion / calibration / feature-importance plots and best-effort
`_plot` plumbing — with the training-dynamics figures deep models add:
loss curves, accuracy curve, learning-rate schedule, and prediction
distribution. Figures land under ``reports/deep_learning/figures/``.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ingestion.logging_config import get_logger
from pipeline.ml.visualization import MLVisualizer

from .base import TrainingHistory

log = get_logger("dl.viz")


class DLVisualizer(MLVisualizer):
    """Evaluation + training-dynamics figures for deep models."""

    def __init__(self, figures_dir: str = "reports/deep_learning/figures",
                 dpi: int = 150) -> None:
        super().__init__(figures_dir, dpi=dpi)

    # ── training dynamics ─────────────────────────────────────────────────────
    def loss_curves(self, name: str,
                    history: TrainingHistory) -> str | None:
        def draw(ax: plt.Axes) -> None:
            epochs = [e.epoch for e in history.epochs]
            ax.plot(epochs, [e.train_loss for e in history.epochs],
                    lw=2, label="train loss")
            ax.plot(epochs, [e.val_loss for e in history.epochs],
                    lw=2, label="validation loss")
            if history.best_epoch:
                ax.axvline(history.best_epoch, ls="--", c="grey", lw=1,
                           label=f"best epoch ({history.best_epoch})")
            ax.set(xlabel="Epoch", ylabel="Loss",
                   title=f"Training / Validation Loss — {name}")
            ax.legend()
        return self._plot(f"loss_{name}", draw)

    def accuracy_curve(self, name: str,
                       history: TrainingHistory) -> str | None:
        points = [(e.epoch, e.val_metrics["accuracy"])
                  for e in history.epochs if "accuracy" in e.val_metrics]
        if not points:
            return None

        def draw(ax: plt.Axes) -> None:
            xs, ys = zip(*points)
            ax.plot(xs, ys, lw=2, label="validation accuracy")
            ax.set(xlabel="Epoch", ylabel="Accuracy",
                   title=f"Validation Accuracy — {name}")
            ax.legend()
        return self._plot(f"accuracy_{name}", draw)

    def lr_curve(self, name: str,
                 history: TrainingHistory) -> str | None:
        def draw(ax: plt.Axes) -> None:
            ax.plot([e.epoch for e in history.epochs],
                    [e.lr for e in history.epochs], lw=2)
            ax.set(xlabel="Epoch", ylabel="Learning rate",
                   title=f"Learning-Rate Schedule — {name}", yscale="log")
        return self._plot(f"lr_{name}", draw)

    def prediction_distribution(self, name: str, y_true: np.ndarray,
                                y_proba: np.ndarray,
                                threshold: float = 0.5) -> str | None:
        def draw(ax: plt.Axes) -> None:
            y = np.asarray(y_true)
            p = np.asarray(y_proba)
            ax.hist(p[y == 0], bins=40, alpha=0.6, label="healthy",
                    density=True)
            ax.hist(p[y == 1], bins=40, alpha=0.6, label="crisis",
                    density=True)
            ax.axvline(threshold, ls="--", c="grey", lw=1,
                       label=f"threshold {threshold:.3f}")
            ax.set(xlabel="Predicted probability", ylabel="Density",
                   title=f"Prediction Distribution — {name}")
            ax.legend()
        return self._plot(f"pred_dist_{name}", draw)
