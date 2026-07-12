"""Dimensionality analysis — PCA variance structure + optional 2-D projection.

Standardizes the numeric feature matrix and fits a PCA to answer: how many
components are needed to explain the bulk of the variance (a proxy for the
intrinsic dimensionality / redundancy of the ratio set), and do the target
classes separate in the leading principal-component plane.

Produces:
* an explained-variance table (per component + cumulative);
* the number of components for 80/90/95/99 % variance;
* a scree/cumulative-variance figure and (optionally) a PC1–PC2 scatter
  coloured by the target.

Requires scikit-learn; skips gracefully if it (or a valid numeric matrix) is
unavailable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import AnalysisResult, EdaAnalyzer

_VARIANCE_TARGETS = (0.80, 0.90, 0.95, 0.99)


class DimensionalityAnalysis(EdaAnalyzer):
    """PCA-based redundancy / intrinsic-dimensionality analysis."""

    name = "dimensionality"

    def _analyze(self, df: pd.DataFrame) -> AnalysisResult:
        result = AnalysisResult(analyzer=self.name)
        num_cols = self.numeric_features(df)
        if len(num_cols) < 2:
            return AnalysisResult(analyzer=self.name, skipped=True,
                                  skip_reason="need >=2 numeric features")
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:  # noqa: BLE001 - sklearn optional
            return AnalysisResult(analyzer=self.name, skipped=True,
                                  skip_reason=f"sklearn unavailable: {exc}")

        X = df[num_cols].fillna(df[num_cols].median(numeric_only=True))
        Xs = StandardScaler().fit_transform(X)
        n_comp = min(len(num_cols), len(df))
        pca = PCA(n_components=n_comp, random_state=42)
        scores = pca.fit_transform(Xs)

        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr)
        var_tbl = pd.DataFrame({
            "component": [f"PC{i + 1}" for i in range(len(evr))],
            "explained_variance_ratio": evr.astype(float),
            "cumulative_variance": cum.astype(float),
        })
        result.tables["pca_explained_variance"] = var_tbl

        n_for = {f"n_components_{int(t * 100)}pct":
                 int(np.searchsorted(cum, t) + 1) for t in _VARIANCE_TARGETS}

        result.summary = {
            "n_features": len(num_cols),
            "n_components": int(n_comp),
            "pc1_variance": float(evr[0]),
            "pc2_variance": float(evr[1]) if len(evr) > 1 else 0.0,
            "top5_cumulative_variance": float(cum[min(4, len(cum) - 1)]),
            **n_for,
            # A crude redundancy index: 1 − (components for 95% / total).
            "redundancy_index": round(
                1 - n_for["n_components_95pct"] / len(num_cols), 4),
        }

        if self.figures is not None:
            self._scree(var_tbl, result)
            if bool(self.cfg.get("projection", True)):
                self._projection(scores, df, result)

        result.note(
            f"{n_for['n_components_95pct']} PCs explain 95% of variance "
            f"({len(num_cols)} features); PC1={evr[0]:.1%}")
        return result

    # ── figures ───────────────────────────────────────────────────────────────
    def _scree(self, var_tbl: pd.DataFrame, result: AnalysisResult) -> None:
        show = var_tbl.head(int(self.cfg.get("scree_components", 30)))
        with self.figures.figure(figsize=(11, 5)) as fig:
            ax1 = fig.add_subplot(111)
            x = np.arange(1, len(show) + 1)
            ax1.bar(x, show["explained_variance_ratio"], color="#2c7fb8",
                    alpha=0.75, label="individual")
            ax1.set_xlabel("principal component")
            ax1.set_ylabel("explained variance ratio")
            ax2 = ax1.twinx()
            ax2.plot(x, show["cumulative_variance"], color="#c0392b",
                     marker="o", ms=3, label="cumulative")
            ax2.set_ylabel("cumulative variance")
            ax2.axhline(0.95, color="grey", ls="--", lw=0.8)
            ax1.set_title("PCA scree & cumulative explained variance")
            path = self.figures.save(fig, "pca_scree")
        result.figures.append(path)

    def _projection(self, scores, df, result: AnalysisResult) -> None:
        if scores.shape[1] < 2:
            return
        with self.figures.figure(figsize=(8, 7)) as fig:
            ax = fig.add_subplot(111)
            if (self.target_col in df.columns
                    and df[self.target_col].nunique(dropna=True) <= 10):
                y = df[self.target_col].to_numpy()
                for cls in pd.unique(y):
                    mask = y == cls
                    ax.scatter(scores[mask, 0], scores[mask, 1], s=8,
                               alpha=0.5, label=f"{self.target_col}={cls}")
                ax.legend(fontsize=8)
            else:
                ax.scatter(scores[:, 0], scores[:, 1], s=8, alpha=0.5,
                           color="#2c7fb8")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("PCA projection (PC1 vs PC2)")
            path = self.figures.save(fig, "pca_projection")
        result.figures.append(path)
