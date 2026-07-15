"""Multicollinearity detection & removal.

Two complementary, fit-on-train techniques (each independently toggleable):

* **Correlation filtering** — for every pair of numeric features whose absolute
  Pearson correlation exceeds ``correlation_threshold``, drop the member of the
  pair with the *weaker* absolute correlation to the target (keeping the more
  predictive one). Seeded from the EDA ``high_correlation_pairs`` hint when
  available, but always recomputed on the engineered train frame since feature
  generation adds new columns the EDA never saw.
* **Variance Inflation Factor (VIF)** — iteratively drops the feature with the
  highest VIF until every remaining feature is below ``vif_threshold``. VIF is
  computed from the diagonal of the inverse correlation matrix
  (``VIF_i = [R^-1]_ii``), which is algebraically identical to the classic
  ``1/(1-R²)`` regression formulation but needs no statsmodels dependency.

The union of dropped columns is remembered and re-applied to val/test via the
base class's kept-column mechanism.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FeatureResult, FeatureStep


class MulticollinearityFilter(FeatureStep):
    """Drop highly-correlated and high-VIF features (fit on train)."""

    name = "multicollinearity"

    # ── fit ───────────────────────────────────────────────────────────────────
    def _fit_transform(self, df: pd.DataFrame) -> FeatureResult:
        num_cols = self.numeric_features(df)
        if len(num_cols) < 2:
            return FeatureResult(step=self.name, df=df, skipped=True,
                                 skip_reason="fewer than 2 numeric features")

        removed_corr = (self._correlation_filter(df, num_cols)
                        if self.cfg.get("correlation_filter", True) else [])
        survivors = [c for c in num_cols if c not in removed_corr]
        removed_vif, vif_table = ([], {})
        if self.cfg.get("vif_filter", True):
            removed_vif, vif_table = self._vif_filter(df, survivors)

        removed = removed_corr + removed_vif
        keep = [c for c in df.columns if c not in removed]
        out = df[keep]

        result = FeatureResult(step=self.name, df=out,
                               selected=[c for c in keep if c != self.target_col],
                               removed=removed)
        result.params = {
            "correlation_threshold": float(
                self.cfg.get("correlation_threshold", 0.95)),
            "vif_threshold": float(self.cfg.get("vif_threshold", 10.0)),
        }
        result.stats = {
            "n_removed_correlation": len(removed_corr),
            "n_removed_vif": len(removed_vif),
            "removed_correlation": removed_corr,
            "removed_vif": removed_vif,
            "final_max_vif": max(vif_table.values()) if vif_table else None,
        }
        result.note(f"dropped {len(removed_corr)} by correlation, "
                    f"{len(removed_vif)} by VIF")
        return result

    # ── correlation filtering ─────────────────────────────────────────────────
    def _correlation_filter(self, df: pd.DataFrame,
                            num_cols: list[str]) -> list[str]:
        thresh = float(self.cfg.get("correlation_threshold", 0.95))
        corr = df[num_cols].corr().abs()
        # Target relevance decides which member of a correlated pair survives.
        if self.target_col in df.columns:
            relevance = df[num_cols].corrwith(df[self.target_col]).abs()
        else:
            relevance = df[num_cols].var()
        relevance = relevance.fillna(0.0)

        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        dropped: set[str] = set()
        pairs = upper.stack()
        for (a, b), r in pairs[pairs > thresh].sort_values(
                ascending=False).items():
            if a in dropped or b in dropped:
                continue
            dropped.add(a if relevance[a] < relevance[b] else b)
        return sorted(dropped)

    # ── VIF filtering ─────────────────────────────────────────────────────────
    def _vif_filter(self, df: pd.DataFrame,
                    cols: list[str]) -> tuple[list[str], dict]:
        thresh = float(self.cfg.get("vif_threshold", 10.0))
        max_iter = int(self.cfg.get("vif_max_iterations", 50))
        active = [c for c in cols
                  if df[c].std() > 0]  # constants have undefined VIF
        removed: list[str] = []
        vifs: dict[str, float] = {}
        for _ in range(max_iter):
            if len(active) < 2:
                break
            vifs = self._compute_vif(df[active])
            worst, worst_v = max(vifs.items(), key=lambda kv: kv[1])
            if worst_v <= thresh:
                break
            active.remove(worst)
            removed.append(worst)
            self.log.debug("VIF drop %s (%.1f)", worst, worst_v)
        return removed, vifs

    @staticmethod
    def _compute_vif(X: pd.DataFrame) -> dict[str, float]:
        """VIF via the inverse correlation matrix (pinv guards singularity)."""
        corr = np.corrcoef(X.to_numpy(dtype=float), rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        inv = np.linalg.pinv(corr)
        diag = np.clip(np.diag(inv), 1.0, 1e12)
        return dict(zip(X.columns, diag.astype(float)))
