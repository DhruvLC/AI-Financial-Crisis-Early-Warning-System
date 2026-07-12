"""Missing-value analysis: per-column counts, percentages, overall report."""
from __future__ import annotations

import pandas as pd

from ..base import BaseCheck, CheckOutcome, Severity
from ..schemas import SourceSchema


class MissingValueAnalyzer(BaseCheck):
    """Quantify missingness overall and per column."""

    name = "missing_values"

    def _run(self, df: pd.DataFrame, spec: SourceSchema, ctx: dict) -> CheckOutcome:
        out = CheckOutcome(check=self.name)
        # Warn threshold per column; error threshold for the whole table.
        col_warn = float(self.cfg.get("column_missing_warn_pct", 0.30))
        col_error = float(self.cfg.get("column_missing_error_pct", 0.90))

        n_rows = max(len(df), 1)
        miss_counts = df.isna().sum()
        miss_pct = (miss_counts / n_rows).round(4)
        total_cells = max(df.shape[0] * df.shape[1], 1)
        overall_pct = float(miss_counts.sum()) / total_cells

        by_col = {c: {"count": int(miss_counts[c]), "pct": float(miss_pct[c])}
                  for c in df.columns if miss_counts[c] > 0}

        # A required column that is fully/mostly empty is fatal.
        for col in spec.required_columns:
            if col in miss_pct and miss_pct[col] >= col_error:
                out.add("required_column_mostly_missing", Severity.ERROR,
                        f"required column '{col}' is {miss_pct[col]:.1%} missing",
                        column=col, pct=float(miss_pct[col]))

        heavy = {c: float(miss_pct[c]) for c in df.columns
                 if col_warn <= miss_pct[c] < col_error}
        if heavy:
            out.add("columns_with_high_missingness", Severity.WARN,
                    f"{len(heavy)} column(s) >= {col_warn:.0%} missing: "
                    f"{list(heavy)[:10]}", columns=heavy)

        if overall_pct > 0 and not out.findings:
            out.add("some_missing", Severity.INFO,
                    f"{overall_pct:.2%} of cells missing")

        out.metrics = {
            "overall_missing_pct": round(overall_pct, 4),
            "columns_with_missing": len(by_col),
            "missing_by_col": by_col,
            "completeness_score": round(1.0 - overall_pct, 4),
        }
        return out
