"""Publication-quality figure management for the EDA module.

Centralises matplotlib/seaborn styling and PNG persistence so every analyzer
produces consistent, high-resolution figures without repeating boilerplate.
Uses the non-interactive ``Agg`` backend so the module runs headless (CI, cron,
servers) exactly like the rest of the pipeline.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")  # headless — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from ingestion.logging_config import get_logger

log = get_logger("eda.figures")


class FigureManager:
    """Apply a consistent house style and save figures as high-res PNGs."""

    def __init__(self, figures_dir: str = "reports/eda/figures",
                 dpi: int = 150, style: str = "whitegrid",
                 palette: str = "deep", enabled: bool = True) -> None:
        self.figures_dir = figures_dir
        self.dpi = int(dpi)
        self.enabled = bool(enabled)
        self.saved: list[str] = []
        os.makedirs(self.figures_dir, exist_ok=True)
        try:
            sns.set_theme(style=style, palette=palette, context="notebook")
            plt.rcParams.update({
                "figure.autolayout": True,
                "axes.titleweight": "bold",
                "axes.titlesize": 12,
                "savefig.bbox": "tight",
                "font.size": 10,
            })
        except Exception as exc:  # noqa: BLE001 - styling must never crash a run
            log.warning("could not apply seaborn theme (%s)", exc)

    @contextmanager
    def figure(self, figsize: tuple[float, float] = (10, 6)):
        """Context manager yielding a fresh Figure that is always closed."""
        fig = plt.figure(figsize=figsize)
        try:
            yield fig
        finally:
            plt.close(fig)

    def save(self, fig, filename: str) -> str:
        """Persist ``fig`` as a PNG under the figures directory; return its path."""
        if not filename.endswith(".png"):
            filename += ".png"
        path = os.path.join(self.figures_dir, filename)
        fig.savefig(path, dpi=self.dpi)
        self.saved.append(path)
        log.debug("saved figure: %s", path)
        return path
