"""Feature distribution analysis — histograms, density, boxplots, violins.

Generates distribution figures for numeric features and automatically flags
highly-skewed and heavy-tailed variables (via skewness and excess kurtosis).
To keep the figure count bounded on wide tables, per-feature histogram/boxplot
grids are drawn in batches, and the more expensive density/violin plots are
limited to the ``top_n`` most non-normal features (configurable).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from ..base import AnalysisResult, EdaAnalyzer


class FeatureDistributionAnalysis(EdaAnalyzer):
    """Distribution plots + automatic skew / heavy-tail detection."""

    name = "distributions"

    def _analyze(self, df: pd.DataFrame) -> AnalysisResult:
        result = AnalysisResult(analyzer=self.name)
        num_cols = self.numeric_features(df)
        if not num_cols:
            return AnalysisResult(analyzer=self.name, skipped=True,
                                  skip_reason="no numeric features")

        skew_thresh = float(self.cfg.get("skew_threshold", 1.0))
        kurt_thresh = float(self.cfg.get("kurtosis_threshold", 3.0))
        top_n = int(self.cfg.get("top_n", 12))
        max_grid = int(self.cfg.get("max_grid_features", 36))

        rows = []
        for c in num_cols:
            s = df[c].dropna()
            sk = float(stats.skew(s)) if len(s) > 2 and s.std() > 0 else 0.0
            ku = float(stats.kurtosis(s)) if len(s) > 2 and s.std() > 0 else 0.0
            rows.append({
                "feature": c, "skewness": sk, "excess_kurtosis": ku,
                "highly_skewed": bool(abs(sk) >= skew_thresh),
                "heavy_tailed": bool(ku >= kurt_thresh),
            })
        shape_tbl = pd.DataFrame(rows)
        result.tables["distribution_shape"] = shape_tbl

        skewed = shape_tbl[shape_tbl["highly_skewed"]]["feature"].tolist()
        heavy = shape_tbl[shape_tbl["heavy_tailed"]]["feature"].tolist()
        result.summary = {
            "n_features": len(num_cols),
            "n_highly_skewed": len(skewed),
            "n_heavy_tailed": len(heavy),
            "highly_skewed": skewed[:25],
            "heavy_tailed": heavy[:25],
        }

        if self.figures is not None:
            # Most non-normal features (by |skew| + |kurtosis|) get detailed plots.
            ranked = shape_tbl.assign(
                nonnorm=shape_tbl["skewness"].abs()
                + shape_tbl["excess_kurtosis"].abs()
            ).sort_values("nonnorm", ascending=False)["feature"].tolist()
            focus = ranked[:top_n]
            self._hist_grid(df, ranked[:max_grid], result, "histograms")
            self._box_grid(df, ranked[:max_grid], result, "boxplots")
            self._density_grid(df, focus, result)
            self._violin_grid(df, focus, result)

        result.note(f"{len(skewed)} highly-skewed, {len(heavy)} heavy-tailed "
                    f"of {len(num_cols)} feature(s)")
        return result

    # ── figure grids ─────────────────────────────────────────────────────────
    @staticmethod
    def _grid_shape(n: int) -> tuple[int, int]:
        cols = min(4, n) or 1
        rows = math.ceil(n / cols)
        return rows, cols

    def _hist_grid(self, df, cols, result, tag):
        if not cols:
            return
        rows, ncols = self._grid_shape(len(cols))
        with self.figures.figure(figsize=(ncols * 3.2, rows * 2.4)) as fig:
            for i, c in enumerate(cols, 1):
                ax = fig.add_subplot(rows, ncols, i)
                ax.hist(df[c].dropna(), bins=30, color="#2c7fb8",
                        edgecolor="white")
                ax.set_title(str(c)[:32], fontsize=8)
                ax.tick_params(labelsize=6)
            fig.suptitle("Feature histograms", fontsize=13, weight="bold")
            path = self.figures.save(fig, "feature_histograms")
        result.figures.append(path)

    def _box_grid(self, df, cols, result, tag):
        if not cols:
            return
        import seaborn as sns
        rows, ncols = self._grid_shape(len(cols))
        with self.figures.figure(figsize=(ncols * 3.2, rows * 2.4)) as fig:
            for i, c in enumerate(cols, 1):
                ax = fig.add_subplot(rows, ncols, i)
                sns.boxplot(y=df[c].dropna(), ax=ax, color="#7fcdbb")
                ax.set_title(str(c)[:32], fontsize=8)
                ax.set_ylabel("")
                ax.tick_params(labelsize=6)
            fig.suptitle("Feature boxplots", fontsize=13, weight="bold")
            path = self.figures.save(fig, "feature_boxplots")
        result.figures.append(path)

    def _density_grid(self, df, cols, result):
        if not cols:
            return
        import seaborn as sns
        rows, ncols = self._grid_shape(len(cols))
        with self.figures.figure(figsize=(ncols * 3.2, rows * 2.4)) as fig:
            for i, c in enumerate(cols, 1):
                ax = fig.add_subplot(rows, ncols, i)
                try:
                    sns.kdeplot(df[c].dropna(), ax=ax, fill=True,
                                color="#756bb1")
                except Exception:  # noqa: BLE001 - kde can fail on degenerate cols
                    ax.hist(df[c].dropna(), bins=30, color="#756bb1")
                ax.set_title(str(c)[:32], fontsize=8)
                ax.set_ylabel("")
                ax.tick_params(labelsize=6)
            fig.suptitle("Density plots (most non-normal features)",
                         fontsize=13, weight="bold")
            path = self.figures.save(fig, "feature_density")
        result.figures.append(path)

    def _violin_grid(self, df, cols, result):
        if not cols:
            return
        import seaborn as sns
        rows, ncols = self._grid_shape(len(cols))
        with self.figures.figure(figsize=(ncols * 3.2, rows * 2.6)) as fig:
            for i, c in enumerate(cols, 1):
                ax = fig.add_subplot(rows, ncols, i)
                sns.violinplot(y=df[c].dropna(), ax=ax, color="#fdae6b")
                ax.set_title(str(c)[:32], fontsize=8)
                ax.set_ylabel("")
                ax.tick_params(labelsize=6)
            fig.suptitle("Violin plots (most non-normal features)",
                         fontsize=13, weight="bold")
            path = self.figures.save(fig, "feature_violins")
        result.figures.append(path)
