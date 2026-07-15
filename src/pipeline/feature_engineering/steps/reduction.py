"""Dimensionality reduction — project the feature space onto fewer components.

Supports **PCA** and **Truncated SVD** with a configurable target: either an
explicit ``n_components`` or an ``explained_variance`` fraction (PCA picks the
smallest component count reaching it; SVD approximates the same by fitting a
generous basis then truncating at the cumulative-variance cutoff).

Two modes (config ``mode``):

* ``append`` (default) — keep the original features and *add* the component
  scores as ``pca__1..n`` / ``svd__1..n`` columns. Safe default: downstream
  models and the feature-importance step keep interpretable inputs.
* ``replace`` — drop the original numeric features, keep only components.

The fitted projector (and the columns it was fit on) is stored so val/test are
projected with the train-fitted basis — never re-fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD

from ..base import FeatureResult, FeatureStep


class DimensionalityReduction(FeatureStep):
    """PCA / TruncatedSVD projection (fit on train)."""

    name = "reduction"

    def __init__(self, cfg=None, target_col=None, hints=None) -> None:
        super().__init__(cfg, target_col, hints)
        self._model = None
        self._fit_cols: list[str] = []
        self._n_keep = 0

    # ── fit ───────────────────────────────────────────────────────────────────
    def _fit_transform(self, df: pd.DataFrame) -> FeatureResult:
        method = str(self.cfg.get("method", "pca")).lower()
        if method in ("none", "off"):
            return FeatureResult(step=self.name, df=df, skipped=True,
                                 skip_reason="method: none")
        num_cols = self.numeric_features(df)
        if len(num_cols) < 3:
            return FeatureResult(step=self.name, df=df, skipped=True,
                                 skip_reason="fewer than 3 numeric features")

        self._fit_cols = num_cols
        X = df[num_cols].to_numpy(dtype=float)
        target_var = float(self.cfg.get("explained_variance", 0.95))
        n_components = self.cfg.get("n_components")

        if method == "pca":
            self._model = PCA(
                n_components=(int(n_components) if n_components
                              else target_var),
                random_state=int(self.cfg.get("random_state", 42)))
            scores = self._model.fit_transform(X)
            self._n_keep = scores.shape[1]
        elif method in ("svd", "truncated_svd"):
            max_dim = min(len(num_cols) - 1, len(df) - 1)
            fit_n = int(n_components) if n_components else max_dim
            self._model = TruncatedSVD(
                n_components=min(fit_n, max_dim),
                random_state=int(self.cfg.get("random_state", 42)))
            scores = self._model.fit_transform(X)
            if n_components:
                self._n_keep = scores.shape[1]
            else:  # truncate at the explained-variance cutoff
                cum = np.cumsum(self._model.explained_variance_ratio_)
                self._n_keep = int(np.searchsorted(cum, target_var) + 1)
                scores = scores[:, :self._n_keep]
        else:
            return FeatureResult(step=self.name, df=df, skipped=True,
                                 skip_reason=f"unknown method '{method}'")

        prefix = "pca" if method == "pca" else "svd"
        self._prefix = prefix
        out, generated, removed = self._assemble(df, scores)

        evr = self._model.explained_variance_ratio_[:self._n_keep]
        result = FeatureResult(step=self.name, df=out, generated=generated,
                               removed=removed)
        result.params = {"method": method, "mode": self._mode(),
                         "n_components": self._n_keep,
                         "explained_variance_target": target_var}
        result.stats = {
            "n_components": self._n_keep,
            "explained_variance": float(np.sum(evr)),
            "explained_variance_ratio": [round(float(v), 5) for v in evr[:20]],
        }
        result.note(f"{method}: {self._n_keep} components explain "
                    f"{np.sum(evr):.1%} of variance ({self._mode()} mode)")
        return result

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self._fit_cols if c in df.columns]
        if len(cols) != len(self._fit_cols):
            missing = set(self._fit_cols) - set(cols)
            raise ValueError(f"reduction transform: missing columns {missing}")
        scores = self._model.transform(
            df[self._fit_cols].to_numpy(dtype=float))[:, :self._n_keep]
        out, _, _ = self._assemble(df, scores)
        return out

    # ── helpers ───────────────────────────────────────────────────────────────
    def _mode(self) -> str:
        return str(self.cfg.get("mode", "append")).lower()

    def _assemble(self, df: pd.DataFrame, scores: np.ndarray):
        names = [f"{self._prefix}__{i + 1}" for i in range(scores.shape[1])]
        comp = pd.DataFrame(scores, columns=names, index=df.index)
        removed: list[str] = []
        if self._mode() == "replace":
            removed = list(self._fit_cols)
            keep = [c for c in df.columns if c not in removed]
            out = pd.concat([df[keep], comp], axis=1)
        else:
            out = pd.concat([df, comp], axis=1)
        return out, names, removed
