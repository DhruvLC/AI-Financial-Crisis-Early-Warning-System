"""Visualization for the Transformer Models module.

Extends :class:`pipeline.deep_learning.visualization.DLVisualizer` —
reusing loss/accuracy/LR curves, ROC / PR / confusion / calibration /
feature-importance / prediction-distribution plots and the best-effort
``_plot`` plumbing — with the attention figures transformers add:
per-model attention heatmaps, attention-by-feature bars, and a
cross-family (ML vs DL vs transformer) model-comparison chart. Figures
land under ``reports/transformers/figures/``.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ingestion.logging_config import get_logger
from pipeline.deep_learning.visualization import DLVisualizer

from .base import AttentionSummary

__all__ = ["TransformerVisualizer"]

log = get_logger("transformers.viz")


class TransformerVisualizer(DLVisualizer):
    """Evaluation + training-dynamics + attention figures."""

    def __init__(self, figures_dir: str = "reports/transformers/figures",
                 dpi: int = 150) -> None:
        super().__init__(figures_dir, dpi=dpi)

    # ── attention figures ─────────────────────────────────────────────────────
    def attention_heatmap(self, name: str, summary: AttentionSummary,
                          max_tokens: int = 24) -> str | None:
        """Mean token×token attention of the last encoder layer."""
        if summary is None or summary.matrix is None:
            return None
        m = np.asarray(summary.matrix)
        labels = list(summary.matrix_labels)
        if len(labels) > max_tokens:                 # keep the plot legible
            m = m[:max_tokens, :max_tokens]
            labels = labels[:max_tokens]

        def draw(ax: plt.Axes) -> None:
            im = ax.imshow(m, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=6)
            ax.set(xlabel="Key token", ylabel="Query token",
                   title=f"Mean Attention (last layer) — {name}")
            plt.colorbar(im, ax=ax, fraction=0.046)
        return self._plot(f"attention_heatmap_{name}", draw)

    def attention_by_feature(self, name: str, summary: AttentionSummary,
                             top_n: int = 20) -> str | None:
        """Horizontal bars of mean attention each feature receives."""
        if summary is None or not summary.feature_attention:
            return None
        items = list(summary.feature_attention.items())[:top_n]

        def draw(ax: plt.Axes) -> None:
            feats, vals = zip(*items)
            ypos = np.arange(len(feats))
            ax.barh(ypos, vals)
            ax.set_yticks(ypos)
            ax.set_yticklabels(feats, fontsize=7)
            ax.invert_yaxis()
            ax.set(xlabel="Mean attention received",
                   title=f"Attention by Feature — {name}")
        return self._plot(f"attention_features_{name}", draw)

    def cross_family_comparison(self, boards: dict[str, pd.DataFrame],
                                metric: str = "roc_auc") -> str | None:
        """One bar per model across families (ML / deep learning /
        transformers), colour-grouped by family."""
        rows = []
        for family, board in boards.items():
            if board is None or board.empty or metric not in board:
                continue
            for _, r in board.iterrows():
                rows.append((family, str(r["model"]), float(r[metric])))
        if not rows:
            return None
        rows.sort(key=lambda t: -t[2])

        def draw(ax: plt.Axes) -> None:
            families = sorted({r[0] for r in rows})
            colors = dict(zip(families,
                              plt.cm.tab10(np.linspace(0, 1, 10))))
            labels = [f"{m}\n({fam})" for fam, m, _ in rows]
            ax.bar(range(len(rows)), [v for _, _, v in rows],
                   color=[colors[fam] for fam, _, _ in rows])
            ax.set_xticks(range(len(rows)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
            ax.set(ylabel=metric,
                   title=f"Model Comparison Across Families — {metric}")
            handles = [plt.Rectangle((0, 0), 1, 1, color=colors[f])
                       for f in families]
            ax.legend(handles, families, fontsize=8)
        return self._plot(f"cross_family_{metric}", draw)
