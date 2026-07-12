"""Schema validation: required columns, dtypes, missing/unexpected columns."""
from __future__ import annotations

import pandas as pd

from ..base import BaseCheck, CheckOutcome, Severity
from ..schemas import (
    ANY, BOOL, DATETIME, FLOAT, INTEGER, NUMERIC, STRING, SourceSchema,
)


def _dtype_matches(series: pd.Series, kind: str) -> bool:
    """True if ``series`` satisfies the expected dtype ``kind``."""
    if kind == ANY:
        return True
    if kind == NUMERIC:
        return pd.api.types.is_numeric_dtype(series)
    if kind == INTEGER:
        return pd.api.types.is_integer_dtype(series)
    if kind == FLOAT:
        return pd.api.types.is_float_dtype(series)
    if kind == DATETIME:
        # Accept real datetimes, or something coercible for the bulk of values.
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        coerced = pd.to_datetime(series, errors="coerce")
        return coerced.notna().mean() >= 0.9
    if kind == BOOL:
        return pd.api.types.is_bool_dtype(series)
    if kind == STRING:
        return pd.api.types.is_object_dtype(series) or \
            pd.api.types.is_string_dtype(series)
    return True


class SchemaValidator(BaseCheck):
    """Validate columns and dtypes against the source's registered contract."""

    name = "schema"

    def _run(self, df: pd.DataFrame, spec: SourceSchema, ctx: dict) -> CheckOutcome:
        out = CheckOutcome(check=self.name)
        cols = set(df.columns)

        # 1. Required columns present -----------------------------------------
        missing_required = [c for c in spec.required_columns if c not in cols]
        if missing_required:
            out.add("missing_required_columns", Severity.ERROR,
                    f"missing required column(s): {missing_required}",
                    columns=missing_required)

        # 2. Optional-but-known columns that are absent (informational) -------
        missing_optional = [c.name for c in spec.columns
                            if not c.required and c.name not in cols]
        if missing_optional:
            out.add("missing_optional_columns", Severity.INFO,
                    f"optional column(s) absent: {missing_optional}",
                    columns=missing_optional)

        # 3. Unexpected columns -----------------------------------------------
        # For dynamic sources (config-driven series) extras are expected, so we
        # only note them; for fixed-schema sources they are a warning.
        unexpected = sorted(cols - spec.known_columns)
        if unexpected:
            level = Severity.INFO if spec.dynamic_columns else Severity.WARN
            out.add("unexpected_columns", level,
                    f"{len(unexpected)} unexpected column(s): {unexpected[:15]}",
                    columns=unexpected)

        # 4. Dtype validation for known, present columns ----------------------
        dtype_mismatches = {}
        for c in spec.columns:
            if c.name in cols and c.dtype != ANY:
                if not _dtype_matches(df[c.name], c.dtype):
                    dtype_mismatches[c.name] = {
                        "expected": c.dtype, "actual": str(df[c.name].dtype),
                    }
        if dtype_mismatches:
            out.add("dtype_mismatch", Severity.WARN,
                    f"{len(dtype_mismatches)} column(s) have unexpected dtype: "
                    f"{list(dtype_mismatches)[:10]}",
                    mismatches=dtype_mismatches)

        # Metrics feeding the quality score -----------------------------------
        n_required = max(len(spec.required_columns), 1)
        out.metrics = {
            "n_required": len(spec.required_columns),
            "n_missing_required": len(missing_required),
            "n_unexpected": len(unexpected),
            "n_dtype_mismatch": len(dtype_mismatches),
            # 1.0 = perfect schema; penalise missing required + dtype problems.
            "schema_score": max(
                0.0,
                1.0
                - len(missing_required) / n_required
                - 0.25 * len(dtype_mismatches) / max(len(spec.columns), 1),
            ),
        }
        if not out.findings:
            out.add("schema_ok", Severity.INFO, "schema matches contract")
        return out
