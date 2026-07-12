"""Time-series validation: chronological order, duplicate/missing timestamps, gaps.

Works both for single-series feeds (e.g. FRED — one global timeline) and panel
feeds (e.g. Yahoo Finance, World Bank — one timeline per ``entity_id``). For
panel data every check is applied within each entity group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import BaseCheck, CheckOutcome, Severity
from ..schemas import SourceSchema


class TimeSeriesValidator(BaseCheck):
    """Validate temporal structure of dated datasets."""

    name = "time_series"

    def applicable(self, df: pd.DataFrame, spec: SourceSchema) -> bool:
        cols = set(df.columns)
        return any(c in cols for c in (spec.date_columns + spec.year_columns))

    def _time_column(self, df: pd.DataFrame, spec: SourceSchema):
        """Return (column, as_series_sortable) for the primary time axis."""
        for dc in spec.date_columns:
            if dc in df.columns:
                return dc, pd.to_datetime(df[dc], errors="coerce")
        for yc in spec.year_columns:
            if yc in df.columns:
                return yc, pd.to_numeric(df[yc], errors="coerce")
        return None, None

    def _run(self, df: pd.DataFrame, spec: SourceSchema, ctx: dict) -> CheckOutcome:
        out = CheckOutcome(check=self.name)
        tcol, tvals = self._time_column(df, spec)
        if tcol is None:
            return out.skip("no usable time column")

        work = pd.DataFrame({"_t": tvals})
        entity = spec.entity_column if spec.entity_column in df.columns else None
        if entity:
            work[entity] = df[entity].to_numpy()

        n_unparseable = int(work["_t"].isna().sum())
        if n_unparseable:
            out.add("unparseable_timestamps", Severity.WARN,
                    f"{n_unparseable} row(s) have unparseable '{tcol}' values",
                    column=tcol, count=n_unparseable)
        work = work.dropna(subset=["_t"])
        if work.empty:
            return out.skip("no parseable timestamps")

        groups = work.groupby(entity) if entity else [(None, work)]
        unordered_groups, dup_ts_total, gap_groups = 0, 0, 0
        gap_examples: list[dict] = []

        for key, g in groups:
            t = g["_t"]
            # Chronological ordering (as stored) -----------------------------
            if not t.is_monotonic_increasing:
                unordered_groups += 1
            # Duplicate timestamps -------------------------------------------
            dup_ts_total += int(t.duplicated().sum())
            # Gap detection on the sorted, de-duplicated series ---------------
            ts = t.sort_values().drop_duplicates()
            if len(ts) >= 3:
                if pd.api.types.is_datetime64_any_dtype(ts):
                    deltas = ts.diff().dropna().dt.total_seconds()
                else:
                    deltas = ts.diff().dropna().astype(float)
                if len(deltas) and deltas.median() > 0:
                    # A gap is >3x the typical cadence.
                    big = deltas[deltas > 3 * deltas.median()]
                    if len(big):
                        gap_groups += 1
                        if len(gap_examples) < 5:
                            gap_examples.append({
                                "entity": str(key) if key is not None else None,
                                "n_gaps": int(len(big)),
                            })

        n_groups = max(work[entity].nunique() if entity else 1, 1)
        if unordered_groups:
            scope = f"{unordered_groups}/{n_groups} series" if entity else "series"
            out.add("not_chronological", Severity.WARN,
                    f"{scope} not in chronological order (as stored)",
                    n_unordered=unordered_groups)
        if dup_ts_total:
            out.add("duplicate_timestamps", Severity.WARN,
                    f"{dup_ts_total} duplicate timestamp(s) within series",
                    count=dup_ts_total)
        if gap_groups:
            out.add("time_gaps", Severity.INFO,
                    f"{gap_groups} series contain large time gaps "
                    "(> 3x typical cadence)", examples=gap_examples)

        ordered_frac = 1.0 - unordered_groups / n_groups
        out.metrics = {
            "n_series": int(n_groups),
            "unordered_series": unordered_groups,
            "duplicate_timestamps": dup_ts_total,
            "series_with_gaps": gap_groups,
            "unparseable_timestamps": n_unparseable,
            "timeliness_score": round(max(0.0, ordered_frac), 4),
        }
        if not out.findings:
            out.add("time_series_ok", Severity.INFO,
                    "timestamps ordered, unique, and evenly spaced")
        return out
