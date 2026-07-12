"""Correlation & multicollinearity analysis.

Computes Pearson, Spearman and Kendall correlation matrices, ranks the most
highly-correlated feature pairs, draws a heatmap, and produces a
multicollinearity report via the Variance Inflation Factor (VIF).

VIF is derived from the diagonal of the inverse Pearson correlation matrix
(``VIF_i = (R^-1)_ii``), which needs only numpy — avoiding a hard dependency on
statsmodels. A pseudo-inverse is used so a singular matrix degrades gracefully.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import AnalysisResult, EdaAnalyzer


class CorrelationAnalysis(EdaAnalyzer):
    """Pearson/Spearman/Kendall correlations + multicollinearity (VIF)."""

    name = "correlation"

    def _analyze(self, df: pd.DataFrame) -> AnalysisResult:
        result = AnalysisResult(analyzer=self.name)
        num_cols = self.numeric_features(df)
        if len(num_cols) < 2:
            return AnalysisResult(analyzer=self.name, skipped=True,
                                  skip_reason="need >=2 numeric features")

        X = df[num_cols]
        high_thresh = float(self.cfg.get("high_correlation_threshold", 0.8))
        vif_thresh = float(self.cfg.get("vif_threshold", 10.0))
        kendall_max = int(self.cfg.get("kendall_max_features", 40))

        pearson = X.corr(method="pearson")
        spearman = X.corr(method="spearman")
        # Kendall is O(n^2) per pair — cap width to stay tractable on wide tables.
        kcols = num_cols[:kendall_max]
        kendall = X[kcols].corr(method="kendall")

        result.tables["correlation_pearson"] = pearson.reset_index()
        result.tables["correlation_spearman"] = spearman.reset_index()
        result.tables["correlation_kendall"] = kendall.reset_index()

        pairs = self._high_pairs(pearson, high_thresh)
        result.tables["high_correlation_pairs"] = pairs

        vif_tbl = self._vif(pearson, vif_thresh)
        result.tables["multicollinearity_vif"] = vif_tbl

        result.summary = {
            "n_features": len(num_cols),
            "high_correlation_threshold": high_thresh,
            "n_high_correlation_pairs": int(len(pairs)),
            "top_pairs": pairs.head(15).to_dict("records"),
            "vif_threshold": vif_thresh,
            "n_high_vif": int((vif_tbl["vif"] > vif_thresh).sum()),
            "high_vif_features": vif_tbl[vif_tbl["vif"] > vif_thresh]
            ["feature"].head(20).tolist(),
        }

        if self.figures is not None:
            self._heatmap(pearson, result)

        result.note(f"{len(pairs)} pair(s) |r|>={high_thresh}; "
                    f"{result.summary['n_high_vif']} feature(s) VIF>{vif_thresh}")
        return result

    @staticmethod
    def _high_pairs(corr: pd.DataFrame, thresh: float) -> pd.DataFrame:
        m = corr.to_numpy()
        cols = corr.columns.tolist()
        out = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = m[i, j]
                if np.isfinite(r) and abs(r) >= thresh:
                    out.append({"feature_a": cols[i], "feature_b": cols[j],
                                "pearson_r": float(r)})
        pairs = pd.DataFrame(out)
        if not pairs.empty:
            pairs = pairs.reindex(
                pairs["pearson_r"].abs().sort_values(ascending=False).index
            ).reset_index(drop=True)
        return pairs

    @staticmethod
    def _vif(corr: pd.DataFrame, thresh: float) -> pd.DataFrame:
        cols = corr.columns.tolist()
        c = corr.to_numpy()
        c = np.nan_to_num(c, nan=0.0)
        try:
            inv = np.linalg.pinv(c)
            vif = np.clip(np.diag(inv), 0, None)
        except Exception:  # noqa: BLE001 - never crash on numerical issues
            vif = np.full(len(cols), np.nan)
        tbl = pd.DataFrame({"feature": cols, "vif": vif.astype(float)})
        return tbl.sort_values("vif", ascending=False).reset_index(drop=True)

    def _heatmap(self, corr: pd.DataFrame, result: AnalysisResult) -> None:
        import seaborn as sns
        n = corr.shape[0]
        annot = n <= 20
        with self.figures.figure(figsize=(min(0.35 * n + 4, 22),
                                          min(0.32 * n + 3, 20))) as fig:
            ax = fig.add_subplot(111)
            sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0,
                        vmin=-1, vmax=1, annot=annot, fmt=".2f",
                        square=False, cbar_kws={"shrink": 0.6},
                        xticklabels=False if n > 40 else True)
            ax.set_title("Pearson correlation heatmap")
            ax.tick_params(labelsize=6)
            path = self.figures.save(fig, "correlation_heatmap")
        result.figures.append(path)
