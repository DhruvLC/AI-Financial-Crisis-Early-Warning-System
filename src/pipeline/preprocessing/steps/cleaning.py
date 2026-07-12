"""Data cleaning — invalid-record removal, value/format standardization,
and consistency checks.

Runs pre-split on the full frame. Four sub-tasks, each independently toggleable:

* **Invalid record removal** — drop rows with a missing/blank target, and
  (config-driven) rows carrying ``inf``/``-inf`` in numeric feature columns
  that would silently break scaling and tree splits downstream.
* **Standardize categorical values** — trim/collapse whitespace, optionally
  case-fold, and normalise common null-like tokens (``"na"``, ``"none"``,
  ``"null"``, ``"?"``, ``""``) to real ``NaN`` so the imputer handles them.
* **Standardize date formats** — parse the schema's date columns with
  ``pd.to_datetime`` (ISO-normalised, invalid -> ``NaT``).
* **Data consistency checks** — non-fatal audits recorded in the report
  (e.g. columns that are entirely null after cleaning).

Config (``preprocessing.cleaning``)::

    cleaning:
      enabled: true
      drop_invalid_target: true
      drop_inf_rows: true
      standardize_categoricals: true
      lowercase_categoricals: false
      standardize_dates: true
      null_like_tokens: ["", "na", "n/a", "none", "null", "?", "nan"]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import PreprocessStep, StepResult

_DEFAULT_NULL_TOKENS = ["", "na", "n/a", "none", "null", "?", "nan", "-"]


class DataCleaner(PreprocessStep):
    """Remove invalid records and standardize categorical/date values."""

    name = "cleaning"

    def _date_columns(self, df: pd.DataFrame) -> list[str]:
        spec = self.spec
        if spec is None:
            return []
        cols = list(getattr(spec, "date_columns", []) or [])
        return [c for c in cols if c in df.columns]

    def _fit_transform(self, df: pd.DataFrame) -> StepResult:
        df = df.copy()
        before = len(df)
        result = StepResult(step=self.name, df=df)
        stats: dict = {"rows_before": before}
        consistency: list[str] = []

        feats = self.feature_columns(df)

        # 1. Invalid record removal — bad target -----------------------------
        dropped_target = 0
        if self.cfg.get("drop_invalid_target", True) and \
                self.target_col in df.columns:
            bad = df[self.target_col].isna()
            dropped_target = int(bad.sum())
            if dropped_target:
                df = df[~bad]
        stats["dropped_invalid_target"] = dropped_target

        # 2. Standardize categorical values ----------------------------------
        cat_cols = [c for c in feats
                    if not pd.api.types.is_numeric_dtype(df[c])
                    and not pd.api.types.is_datetime64_any_dtype(df[c])]
        null_tokens = {str(t).lower() for t in
                       self.cfg.get("null_like_tokens", _DEFAULT_NULL_TOKENS)}
        lower = bool(self.cfg.get("lowercase_categoricals", False))
        standardized_cols = 0
        if self.cfg.get("standardize_categoricals", True) and cat_cols:
            for c in cat_cols:
                s = df[c].astype("string")
                # collapse internal whitespace + strip
                s = s.str.strip().str.replace(r"\s+", " ", regex=True)
                if lower:
                    s = s.str.lower()
                # normalise null-like tokens -> real NaN
                s = s.mask(s.str.lower().isin(null_tokens), other=pd.NA)
                df[c] = s
            standardized_cols = len(cat_cols)
        stats["standardized_categorical_cols"] = standardized_cols

        # 3. Standardize date formats ----------------------------------------
        date_cols = self._date_columns(df)
        parsed_dates = 0
        for c in date_cols:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
            parsed_dates += 1
        stats["standardized_date_cols"] = parsed_dates

        # 4. Invalid record removal — inf rows -------------------------------
        dropped_inf = 0
        numeric = df[[c for c in feats if c in df.columns]].select_dtypes(
            include=[np.number])
        if self.cfg.get("drop_inf_rows", True) and not numeric.empty:
            inf_mask = np.isinf(numeric.to_numpy()).any(axis=1)
            dropped_inf = int(inf_mask.sum())
            if dropped_inf:
                df = df[~inf_mask]
        stats["dropped_inf_rows"] = dropped_inf

        # 5. Consistency checks (non-fatal, recorded for the report) ---------
        df = df.reset_index(drop=True)
        for c in self.feature_columns(df):
            if len(df) and df[c].isna().all():
                consistency.append(f"column '{c}' is entirely null after cleaning")
        if self.target_col in df.columns:
            n_classes = df[self.target_col].nunique(dropna=True)
            if n_classes < 2:
                consistency.append(
                    f"target '{self.target_col}' has <2 classes after cleaning "
                    f"({n_classes})")

        stats["rows_after"] = len(df)
        stats["removed_total"] = before - len(df)
        stats["consistency_findings"] = consistency

        result.df = df
        result.params = {
            "date_columns": date_cols,
            "categorical_columns": cat_cols,
            "lowercase_categoricals": lower,
        }
        result.stats = stats
        result.note(
            f"removed {before - len(df)} invalid record(s); standardized "
            f"{standardized_cols} categorical + {parsed_dates} date column(s)")
        for c in consistency:
            result.note(f"consistency: {c}")
        return result
