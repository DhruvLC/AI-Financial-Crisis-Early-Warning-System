"""Financial-data validation: signs, ranges, ratios, fiscal years, future dates.

Handles both wide feeds (explicit revenue/asset/price columns) and long-format
feeds like SEC EDGAR, where a ``concept`` column selects the semantic rule to
apply to a single ``value`` column.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..base import BaseCheck, CheckOutcome, Severity
from ..schemas import MIN_FISCAL_YEAR, FinancialSpec, SourceSchema


class FinancialValidator(BaseCheck):
    """Domain sanity checks for financial datasets."""

    name = "financial"

    def applicable(self, df: pd.DataFrame, spec: SourceSchema) -> bool:
        # Applies if there's a financial spec, or any date column to future-check.
        return spec.financial is not None or bool(spec.date_columns)

    def _run(self, df: pd.DataFrame, spec: SourceSchema, ctx: dict) -> CheckOutcome:
        out = CheckOutcome(check=self.name)
        n_rows = max(len(df), 1)
        fin: FinancialSpec = spec.financial or FinancialSpec()
        invalid_rows = 0

        # 1. Non-negative columns (revenue, volume, …) ------------------------
        for col in fin.nonneg_columns:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                bad = int((s < 0).sum())
                if bad:
                    invalid_rows += bad
                    out.add("negative_value", Severity.ERROR,
                            f"'{col}' has {bad} negative value(s) (must be >= 0)",
                            column=col, count=bad)

        # 2. Strictly-positive columns (assets, prices) -----------------------
        for col in fin.positive_columns:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                bad = int((s <= 0).sum())
                if bad:
                    invalid_rows += bad
                    out.add("nonpositive_value", Severity.ERROR,
                            f"'{col}' has {bad} value(s) <= 0 (must be > 0)",
                            column=col, count=bad)

        # 3. Ratio bounds -----------------------------------------------------
        for col, (lo, hi) in (fin.ratio_bounds or {}).items():
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                bad = int(((s < lo) | (s > hi)).sum())
                if bad:
                    invalid_rows += bad
                    out.add("impossible_ratio", Severity.WARN,
                            f"'{col}' has {bad} value(s) outside [{lo}, {hi}]",
                            column=col, count=bad)

        # 4. Long-format concept rules (SEC EDGAR) ----------------------------
        invalid_rows += self._check_concepts(df, fin, out)

        # 5. Fiscal-year sanity ----------------------------------------------
        cur_year = datetime.now(timezone.utc).year
        fy_col = fin.fiscal_year_column
        if fy_col and fy_col in df.columns:
            fy = pd.to_numeric(df[fy_col], errors="coerce").dropna()
            bad = int(((fy < MIN_FISCAL_YEAR) | (fy > cur_year + 1)).sum())
            if bad:
                out.add("invalid_fiscal_year", Severity.WARN,
                        f"{bad} row(s) have fiscal year outside "
                        f"[{MIN_FISCAL_YEAR}, {cur_year + 1}]",
                        column=fy_col, count=bad)

        # 6. Future dates -----------------------------------------------------
        grace = int((self.cfg.get("future_date_grace_days", 2)
                     if isinstance(self.cfg, dict) else 2))
        now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
        cutoff = now + pd.Timedelta(days=grace)
        for dc in spec.date_columns:
            if dc in df.columns:
                dts = pd.to_datetime(df[dc], errors="coerce")
                bad = int((dts > cutoff).sum())
                if bad:
                    invalid_rows += bad
                    out.add("future_date", Severity.ERROR,
                            f"'{dc}' has {bad} date(s) in the future",
                            column=dc, count=bad)
        # Integer-year future check
        for yc in spec.year_columns:
            if yc in df.columns:
                yr = pd.to_numeric(df[yc], errors="coerce")
                bad = int((yr > cur_year + 1).sum())
                if bad:
                    out.add("future_year", Severity.WARN,
                            f"'{yc}' has {bad} year(s) beyond {cur_year + 1}",
                            column=yc, count=bad)

        out.metrics = {
            "invalid_financial_rows": invalid_rows,
            "invalid_financial_pct": round(invalid_rows / n_rows, 4),
            "validity_score": round(max(0.0, 1.0 - invalid_rows / n_rows), 4),
        }
        if not out.findings:
            out.add("financials_ok", Severity.INFO, "financial sanity checks passed")
        return out

    def _check_concepts(self, df: pd.DataFrame, fin: FinancialSpec,
                        out: CheckOutcome) -> int:
        """Apply per-concept sign rules to a long-format value column."""
        if not (fin.concept_column and fin.value_column and fin.concept_rules):
            return 0
        if fin.concept_column not in df.columns or fin.value_column not in df.columns:
            return 0
        invalid = 0
        val = pd.to_numeric(df[fin.value_column], errors="coerce")
        for concept, rule in fin.concept_rules.items():
            mask = df[fin.concept_column] == concept
            if not mask.any():
                continue
            vals = val[mask]
            if rule == "positive":
                bad = int((vals <= 0).sum())
            elif rule == "nonneg":
                bad = int((vals < 0).sum())
            else:
                bad = 0
            if bad:
                invalid += bad
                out.add("invalid_concept_value", Severity.ERROR,
                        f"concept '{concept}' has {bad} value(s) violating "
                        f"'{rule}' rule", concept=concept, count=bad)
        return invalid
