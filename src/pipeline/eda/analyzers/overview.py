"""Dataset overview — dimensions, feature inventory, target information."""
from __future__ import annotations

import pandas as pd

from ..base import AnalysisResult, EdaAnalyzer


class DatasetOverview(EdaAnalyzer):
    """Summarise shape, memory, feature types, and the target variable."""

    name = "overview"

    def _analyze(self, df: pd.DataFrame) -> AnalysisResult:
        result = AnalysisResult(analyzer=self.name)

        numeric = self.numeric_features(df)
        categorical = self.categorical_features(df)
        mem_bytes = int(df.memory_usage(deep=True).sum())

        dtypes_tbl = pd.DataFrame({
            "feature": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "non_null": df.notna().sum().values,
            "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        })
        result.tables["feature_types"] = dtypes_tbl

        summary = {
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "n_features": len(self.feature_columns(df)),
            "n_numeric_features": len(numeric),
            "n_categorical_features": len(categorical),
            "memory_bytes": mem_bytes,
            "memory_mb": round(mem_bytes / (1024 ** 2), 3),
            "feature_names": [str(c) for c in self.feature_columns(df)],
            "target_col": self.target_col,
        }

        if self.target_col in df.columns:
            tgt = df[self.target_col]
            counts = tgt.value_counts(dropna=False)
            summary["target"] = {
                "dtype": str(tgt.dtype),
                "n_classes": int(tgt.nunique(dropna=True)),
                "class_counts": {str(k): int(v) for k, v in counts.items()},
                "is_binary": bool(tgt.nunique(dropna=True) == 2),
            }

        result.summary = summary
        result.note(f"{summary['n_rows']} rows x {summary['n_cols']} cols; "
                    f"{summary['n_numeric_features']} numeric / "
                    f"{summary['n_categorical_features']} categorical; "
                    f"{summary['memory_mb']} MB")
        return result
