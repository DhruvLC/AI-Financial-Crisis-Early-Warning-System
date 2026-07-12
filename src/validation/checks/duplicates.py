"""Duplicate detection: full-row, per-entity records, and duplicate timestamps."""
from __future__ import annotations

import pandas as pd

from ..base import BaseCheck, CheckOutcome, Severity
from ..schemas import SourceSchema


class DuplicateDetector(BaseCheck):
    """Detect duplicate rows, duplicate company/entity records, and duplicate
    timestamps within an entity's series."""

    name = "duplicates"

    def _run(self, df: pd.DataFrame, spec: SourceSchema, ctx: dict) -> CheckOutcome:
        out = CheckOutcome(check=self.name)
        n_rows = max(len(df), 1)

        # 1. Full-row duplicates ---------------------------------------------
        dup_rows = int(df.duplicated().sum())
        dup_row_pct = dup_rows / n_rows
        if dup_rows:
            out.add("duplicate_rows", Severity.WARN,
                    f"{dup_rows} fully duplicated row(s) ({dup_row_pct:.2%})",
                    count=dup_rows, pct=round(dup_row_pct, 4))

        # 2. Duplicate entity records / timestamps ---------------------------
        # The natural key is (entity, date) for a time series, or just the
        # entity for a cross-section. Duplicates on that key are suspicious.
        entity = spec.entity_column if spec.entity_column in df.columns else None
        date_cols = [c for c in (spec.date_columns + spec.year_columns)
                     if c in df.columns]
        dup_ts = 0
        key = [c for c in ([entity] if entity else []) + date_cols if c]
        if entity and date_cols:
            dup_ts = int(df.duplicated(subset=key).sum())
            if dup_ts:
                out.add("duplicate_entity_timestamps", Severity.WARN,
                        f"{dup_ts} duplicate (entity, date) record(s) on {key}",
                        key=key, count=dup_ts)
        elif entity:
            dup_ent = int(df.duplicated(subset=[entity]).sum())
            if dup_ent:
                out.add("duplicate_entity_records", Severity.INFO,
                        f"{dup_ent} repeated '{entity}' value(s) "
                        "(expected for panel data)", count=dup_ent)

        # 3. Duplicate timestamps globally (single-series feeds like FRED) ----
        if not entity and date_cols:
            for dc in date_cols:
                dts = int(df.duplicated(subset=[dc]).sum())
                if dts:
                    dup_ts += dts
                    out.add("duplicate_timestamps", Severity.WARN,
                            f"{dts} duplicate value(s) in timestamp column '{dc}'",
                            column=dc, count=dts)

        out.metrics = {
            "duplicate_rows": dup_rows,
            "duplicate_row_pct": round(dup_row_pct, 4),
            "duplicate_timestamps": dup_ts,
            "uniqueness_score": round(1.0 - dup_row_pct, 4),
        }
        if not out.findings:
            out.add("no_duplicates", Severity.INFO, "no duplicate rows detected")
        return out
