"""Duplicate handling — full-row, per-entity, and per-timestamp de-duplication.

The *detection* logic mirrors ``validation.checks.duplicates.DuplicateDetector``
(same natural-key derivation), but this step **removes** duplicates rather than
only flagging them. It runs pre-split on the full frame so the train/val/test
partitions never share duplicated records.

Key derivation reuses the source schema (``spec.entity_column`` /
``spec.date_columns`` / ``spec.year_columns``) when one is available, exactly
like the validator; for the plain bankruptcy modelling table (no schema) it
falls back to full-row de-duplication only.

Config (``preprocessing.duplicates``)::

    duplicates:
      enabled: true
      drop_rows: true               # exact full-row duplicates
      drop_entity_timestamps: true  # duplicate (entity, date) records
      drop_timestamps: true         # duplicate timestamps in single-series feeds
      keep: last                    # which duplicate to keep (first | last)
"""
from __future__ import annotations

import pandas as pd

from ..base import PreprocessStep, StepResult


class DuplicateRemover(PreprocessStep):
    """Remove duplicate rows, duplicate entity records, and duplicate
    timestamps (stateless — applied identically to any frame)."""

    name = "duplicates"

    def _keys(self, df: pd.DataFrame) -> tuple[str | None, list[str]]:
        """Return (entity_column, date_columns) present in ``df`` per the schema."""
        entity, date_cols = None, []
        spec = self.spec
        if spec is not None:
            ent = getattr(spec, "entity_column", None)
            if ent and ent in df.columns:
                entity = ent
            dcols = list(getattr(spec, "date_columns", []) or []) + \
                list(getattr(spec, "year_columns", []) or [])
            date_cols = [c for c in dcols if c in df.columns]
        return entity, date_cols

    def _fit_transform(self, df: pd.DataFrame) -> StepResult:
        keep = self.cfg.get("keep", "last")
        keep = keep if keep in ("first", "last") else "last"
        before = len(df)
        entity, date_cols = self._keys(df)
        removed = {"rows": 0, "entity_timestamps": 0, "timestamps": 0}

        # 1. Exact full-row duplicates ---------------------------------------
        if self.cfg.get("drop_rows", True):
            mask = df.duplicated(keep=keep)
            removed["rows"] = int(mask.sum())
            if removed["rows"]:
                df = df[~mask]

        # 2. Duplicate (entity, timestamp) records ---------------------------
        if self.cfg.get("drop_entity_timestamps", True) and entity and date_cols:
            key = [entity, *date_cols]
            mask = df.duplicated(subset=key, keep=keep)
            removed["entity_timestamps"] = int(mask.sum())
            if removed["entity_timestamps"]:
                df = df[~mask]

        # 3. Duplicate timestamps in a single-series feed --------------------
        elif self.cfg.get("drop_timestamps", True) and not entity and date_cols:
            for dc in date_cols:
                mask = df.duplicated(subset=[dc], keep=keep)
                n = int(mask.sum())
                if n:
                    removed["timestamps"] += n
                    df = df[~mask]

        df = df.reset_index(drop=True)
        result = StepResult(step=self.name, df=df)
        result.params = {
            "keep": keep,
            "entity_column": entity,
            "date_columns": date_cols,
        }
        result.stats = {
            "rows_before": before,
            "rows_after": len(df),
            "removed_total": before - len(df),
            **{f"removed_{k}": v for k, v in removed.items()},
        }
        result.note(f"removed {before - len(df)} duplicate row(s)")
        return result
