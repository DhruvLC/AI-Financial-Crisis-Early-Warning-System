"""Data validation: schema checks, missing-value detection, sanity ranges."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


@dataclass
class ValidationReport:
    """Outcome of validating a single dataset."""
    source: str
    n_rows: int
    n_cols: int
    missing_by_col: dict = field(default_factory=dict)
    missing_total_pct: float = 0.0
    schema_ok: bool = True
    missing_columns: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.schema_ok and not self.errors

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "missing_total_pct": round(self.missing_total_pct, 4),
            "missing_by_col": {k: v for k, v in self.missing_by_col.items() if v},
            "schema_ok": self.schema_ok,
            "missing_columns": self.missing_columns,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class DataValidator:
    """Reusable validator applied by every ingestor before storage."""

    def __init__(self, max_missing_pct: float = 0.5) -> None:
        # Fraction of cells allowed to be null before it's an error, not a warning.
        self.max_missing_pct = max_missing_pct

    def validate(
        self,
        df: pd.DataFrame,
        source: str,
        required_columns: Iterable[str] | None = None,
        non_empty: bool = True,
    ) -> ValidationReport:
        report = ValidationReport(source=source, n_rows=len(df), n_cols=df.shape[1])

        if non_empty and df.empty:
            report.errors.append("dataset is empty")
            report.schema_ok = False
            return report

        # Schema check
        if required_columns:
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                report.missing_columns = missing
                report.schema_ok = False
                report.errors.append(f"missing required columns: {missing}")

        # Missing-value detection
        missing_counts = df.isna().sum()
        report.missing_by_col = missing_counts.to_dict()
        total_cells = max(df.shape[0] * df.shape[1], 1)
        report.missing_total_pct = float(missing_counts.sum()) / total_cells

        if report.missing_total_pct > self.max_missing_pct:
            report.errors.append(
                f"missing ratio {report.missing_total_pct:.2%} exceeds "
                f"{self.max_missing_pct:.0%} threshold"
            )
        elif report.missing_total_pct > 0:
            report.warnings.append(
                f"{report.missing_total_pct:.2%} of cells missing"
            )

        # Duplicate rows are a warning, not fatal
        dupes = int(df.duplicated().sum())
        if dupes:
            report.warnings.append(f"{dupes} duplicate rows")

        return report
