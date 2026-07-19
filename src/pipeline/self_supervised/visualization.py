"""Visualization for the Self-Supervised Learning module.

Extends :class:`pipeline.deep_learning.visualization.DLVisualizer` —
reusing the loss/LR curves, ROC / PR / confusion / calibration plots and
the best-effort ``_plot`` plumbing — with the representation figures SSL
adds: 2-D embedding projections (PCA / t-SNE / UMAP if installed),
cosine-similarity matrices, and embedding-value distributions. Figures
land under ``reports/self_supervised/figures/``.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ingestion.logging_config import get_logger
from pipeline.deep_learning.visualization import DLVisualizer

log = get_logger("ssl.viz")

__all__ = ["SSLVisualizer"]


class SSLVisualizer(DLVisualizer):
    """Pretraining-dynamics + representation figures."""

    def __init__(self,
                 figures_dir: str = "reports/self_supervised/figures",
                 dpi: int = 150, random_state: int = 42) -> None:
        super().__init__(figures_dir, dpi=dpi)
        self.random_state = int(random_state)

    # ── embedding projections ─────────────────────────────────────────────────
    def _project(self, Z: np.ndarray, method: str,
                 max_samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
        """2-D projection of the embeddings; returns (coords, kept idx)."""
        rng = np.random.default_rng(self.random_state)
        idx = np.arange(len(Z))
        if len(Z) > max_samples:
            idx = rng.choice(len(Z), max_samples, replace=False)
        Zs = np.asarray(Z)[idx]
        method = method.lower()
        if method == "pca":
            from sklearn.decomposition import PCA
            return PCA(n_components=2,
                       random_state=self.random_state).fit_transform(Zs), idx
        if method == "tsne":
            from sklearn.manifold import TSNE
            perplexity = min(30.0, max(5.0, (len(Zs) - 1) / 4))
            return TSNE(n_components=2, perplexity=perplexity,
                        random_state=self.random_state,
                        init="pca").fit_transform(Zs), idx
        if method == "umap":
            import umap  # optional dependency; ImportError → skipped plot
            return umap.UMAP(
                n_components=2,
                random_state=self.random_state).fit_transform(Zs), idx
        raise ValueError(f"unknown projection method '{method}'")

    def embedding_projection(self, name: str, Z: np.ndarray,
                             y: np.ndarray, method: str = "pca",
                             max_samples: int = 2000) -> str | None:
        """Scatter of the 2-D projected embeddings, coloured by label."""
        try:
            coords, idx = self._project(Z, method, max_samples)
        except ImportError:
            log.info("%s projection skipped for %s (library not "
                     "installed)", method, name)
            return None
        except Exception as exc:  # noqa: BLE001 - figures are best-effort
            log.warning("%s projection failed for %s: %s", method, name,
                        exc)
            return None
        labels = np.asarray(y)[idx]

        def draw(ax: plt.Axes) -> None:
            for value, label, alpha in ((0, "healthy", 0.4),
                                        (1, "crisis", 0.9)):
                m = labels == value
                ax.scatter(coords[m, 0], coords[m, 1], s=8, alpha=alpha,
                           label=label)
            ax.set(xlabel=f"{method.upper()} 1",
                   ylabel=f"{method.upper()} 2",
                   title=f"Embedding Projection ({method.upper()}) — "
                         f"{name}")
            ax.legend()
        return self._plot(f"projection_{method}_{name}", draw)

    def similarity_matrix(self, name: str, Z: np.ndarray, y: np.ndarray,
                          max_samples: int = 200) -> str | None:
        """Cosine-similarity heatmap of a label-sorted embedding sample."""
        rng = np.random.default_rng(self.random_state)
        idx = np.arange(len(Z))
        if len(Z) > max_samples:
            idx = rng.choice(len(Z), max_samples, replace=False)
        order = np.argsort(np.asarray(y)[idx], kind="stable")
        Zs = np.asarray(Z)[idx][order]
        norms = np.linalg.norm(Zs, axis=1, keepdims=True)
        sim = (Zs / np.clip(norms, 1e-12, None)) @ \
              (Zs / np.clip(norms, 1e-12, None)).T
        n_neg = int((np.asarray(y)[idx] == 0).sum())

        def draw(ax: plt.Axes) -> None:
            im = ax.imshow(sim, cmap="viridis", vmin=-1, vmax=1)
            ax.axhline(n_neg - 0.5, c="white", lw=0.8)
            ax.axvline(n_neg - 0.5, c="white", lw=0.8)
            ax.set(title=f"Cosine Similarity (label-sorted) — {name}",
                   xlabel="sample (healthy | crisis)",
                   ylabel="sample (healthy | crisis)")
            plt.colorbar(im, ax=ax, fraction=0.046)
        return self._plot(f"similarity_{name}", draw)

    def embedding_distribution(self, name: str,
                               Z: np.ndarray) -> str | None:
        """Histograms of embedding values and per-sample L2 norms."""
        Z = np.asarray(Z)

        def draw(ax: plt.Axes) -> None:
            ax.hist(Z.ravel(), bins=60, alpha=0.6, density=True,
                    label="embedding values")
            ax2 = ax.twinx()
            ax2.hist(np.linalg.norm(Z, axis=1), bins=40, alpha=0.4,
                     density=True, color="tab:orange", label="L2 norms")
            ax.set(xlabel="value", ylabel="density (values)",
                   title=f"Representation Distribution — {name}")
            ax2.set_ylabel("density (norms)")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2)
        return self._plot(f"embedding_dist_{name}", draw)
