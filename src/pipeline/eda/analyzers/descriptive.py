"""Descriptive statistics — central tendency, dispersion, shape.

Computes mean, median, mode, std, variance, min, max, quartiles, configurable
percentiles, skewness and kurtosis for every numeric feature. The resulting
table is exported to CSV/Markdown/JSON by the report writer via
``AnalysisResult.tables``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..base import AnalysisResult, EdaAnalyzer

_DEFAULT_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


class DescriptiveStatistics(EdaAnalyzer):
    """Full descriptive-statistics table for numeric features."""

    name = "descriptive"

    def _analyze(self, df: pd.DataFrame) -> AnalysisResult:
        result = AnalysisResult(analyzer=self.name)
        num_cols = self.numeric_features(df)
        if not num_cols:
            return AnalysisResult(analyzer=self.name, skipped=True,
                                  skip_reason="no numeric features")

        pctiles = self.cfg.get("percentiles", _DEFAULT_PERCENTILES)
        rows = []
        for c in num_cols:
            s = df[c].dropna()
            mode = s.mode()
            row = {
                "feature": c,
                "count": int(s.shape[0]),
                "mean": float(s.mean()) if len(s) else np.nan,
                "median": float(s.median()) if len(s) else np.nan,
                "mode": float(mode.iloc[0]) if len(mode) else np.nan,
                "std": float(s.std()) if len(s) else np.nan,
                "variance": float(s.var()) if len(s) else np.nan,
                "min": float(s.min()) if len(s) else np.nan,
                "max": float(s.max()) if len(s) else np.nan,
                "range": float(s.max() - s.min()) if len(s) else np.nan,
                "q1": float(s.quantile(0.25)) if len(s) else np.nan,
                "q3": float(s.quantile(0.75)) if len(s) else np.nan,
                "iqr": float(s.quantile(0.75) - s.quantile(0.25)) if len(s) else np.nan,
                # scipy skew/kurtosis need >0 variance; guard tiny/constant cols
                "skewness": float(stats.skew(s)) if len(s) > 2 and s.std() > 0 else 0.0,
                "kurtosis": float(stats.kurtosis(s)) if len(s) > 2 and s.std() > 0 else 0.0,
            }
            for p in pctiles:
                row[f"p{p}"] = float(s.quantile(p / 100.0)) if len(s) else np.nan
            rows.append(row)

        stats_tbl = pd.DataFrame(rows).set_index("feature")
        result.tables["descriptive_statistics"] = stats_tbl.reset_index()
        result.summary = {
            "n_features_described": len(num_cols),
            "percentiles": pctiles,
            "mean_abs_skew": float(np.nanmean(np.abs(stats_tbl["skewness"]))),
            "mean_kurtosis": float(np.nanmean(stats_tbl["kurtosis"])),
        }
        result.note(f"described {len(num_cols)} numeric feature(s)")
        return result
