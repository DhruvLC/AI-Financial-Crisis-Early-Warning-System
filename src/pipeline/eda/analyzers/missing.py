"""Missing-value analysis — counts, percentages, ranking, heatmap."""
from __future__ import annotations

import pandas as pd

from ..base import AnalysisResult, EdaAnalyzer


class MissingValueAnalysis(EdaAnalyzer):
    """Per-feature missing counts/percentages, ranking, and a nullity heatmap."""

    name = "missing"

    def _analyze(self, df: pd.DataFrame) -> AnalysisResult:
        result = AnalysisResult(analyzer=self.name)
        n = len(df)
        counts = df.isna().sum()
        pct = (counts / n * 100.0).round(4)

        tbl = pd.DataFrame({
            "feature": counts.index,
            "missing_count": counts.values.astype(int),
            "missing_pct": pct.values,
        }).sort_values("missing_count", ascending=False).reset_index(drop=True)
        result.tables["missing_values"] = tbl

        total_missing = int(counts.sum())
        cols_with_missing = tbl[tbl["missing_count"] > 0]
        result.summary = {
            "total_missing_cells": total_missing,
            "overall_missing_pct": round(total_missing / (n * df.shape[1]) * 100, 4),
            "n_features_with_missing": int(len(cols_with_missing)),
            "top_missing": cols_with_missing.head(10).to_dict("records"),
        }

        if self.figures is not None and total_missing > 0:
            self._heatmap(df, result)
        elif total_missing == 0:
            result.note("no missing values in the processed dataset")

        result.note(f"{total_missing} missing cell(s) across "
                    f"{len(cols_with_missing)} feature(s)")
        return result

    def _heatmap(self, df: pd.DataFrame, result: AnalysisResult) -> None:
        import seaborn as sns
        with self.figures.figure(figsize=(12, 6)) as fig:
            ax = fig.add_subplot(111)
            sns.heatmap(df.isna(), cbar=False, ax=ax,
                        cmap=["#f0f0f0", "#c0392b"])
            ax.set_title("Missing-value heatmap (rows × features)")
            ax.set_xlabel("features")
            ax.set_ylabel("rows")
            path = self.figures.save(fig, "missing_values_heatmap")
        result.figures.append(path)
