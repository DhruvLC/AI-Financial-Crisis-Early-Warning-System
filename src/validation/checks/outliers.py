"""Outlier detection using multiple techniques: IQR, Z-Score, Isolation Forest.

Isolation Forest requires scikit-learn. Following the project's convention for
optional dependencies (see the yfinance/parquet fallbacks), it is skipped with
an informational finding when scikit-learn is not installed rather than failing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import BaseCheck, CheckOutcome, Severity
from ..schemas import SourceSchema


class OutlierDetector(BaseCheck):
    """Flag numeric columns / rows with an unusually high outlier fraction."""

    name = "outliers"

    def _numeric_frame(self, df: pd.DataFrame, spec: SourceSchema) -> pd.DataFrame:
        """Numeric columns worth scanning (exclude id/key/year columns)."""
        skip = set(spec.time_series_keys) | set(spec.year_columns)
        if spec.entity_column:
            skip.add(spec.entity_column)
        num = df.select_dtypes(include=[np.number]).replace(
            [np.inf, -np.inf], np.nan
        )
        keep = [c for c in num.columns if c not in skip and num[c].nunique() >= 5]
        return num[keep]

    def _run(self, df: pd.DataFrame, spec: SourceSchema, ctx: dict) -> CheckOutcome:
        out = CheckOutcome(check=self.name)
        ocfg = self.cfg.get("outliers", {}) if isinstance(self.cfg, dict) else {}
        z_thresh = float(ocfg.get("zscore_threshold", 3.0))
        iqr_mult = float(ocfg.get("iqr_multiplier", 1.5))
        flag_pct = float(ocfg.get("column_flag_pct", 0.10))

        num = self._numeric_frame(df, spec)
        if num.empty or len(num) < 20:
            return out.skip("too few numeric rows/columns for outlier analysis")

        iqr_by_col: dict[str, float] = {}
        z_by_col: dict[str, float] = {}
        for col in num.columns:
            s = num[col].dropna()
            if len(s) < 20:
                continue
            # IQR fence
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
                iqr_by_col[col] = round(float(((s < lo) | (s > hi)).mean()), 4)
            # Z-score
            std = s.std(ddof=0)
            if std > 0:
                z = (s - s.mean()).abs() / std
                z_by_col[col] = round(float((z > z_thresh).mean()), 4)

        iqr_flagged = {c: p for c, p in iqr_by_col.items() if p > flag_pct}
        z_flagged = {c: p for c, p in z_by_col.items() if p > flag_pct}
        if iqr_flagged:
            out.add("iqr_outliers", Severity.WARN,
                    f"{len(iqr_flagged)} column(s) exceed {flag_pct:.0%} IQR "
                    f"outliers: {list(iqr_flagged)[:10]}", columns=iqr_flagged)
        if z_flagged:
            out.add("zscore_outliers", Severity.WARN,
                    f"{len(z_flagged)} column(s) exceed {flag_pct:.0%} z-score "
                    f"outliers (|z|>{z_thresh}): {list(z_flagged)[:10]}",
                    columns=z_flagged)

        # Isolation Forest (multivariate, optional dependency) ----------------
        iforest_pct = None
        if ocfg.get("isolation_forest", True):
            iforest_pct = self._isolation_forest(num, ocfg, out)

        # Column-averaged outlier rate feeds the quality score.
        mean_iqr = float(np.mean(list(iqr_by_col.values()))) if iqr_by_col else 0.0
        out.metrics = {
            "iqr_outlier_pct_by_col": iqr_by_col,
            "zscore_outlier_pct_by_col": z_by_col,
            "isolation_forest_outlier_pct": iforest_pct,
            "mean_iqr_outlier_pct": round(mean_iqr, 4),
        }
        if not out.findings:
            out.add("no_significant_outliers", Severity.INFO,
                    "no column exceeds the outlier threshold")
        return out

    def _isolation_forest(self, num: pd.DataFrame, ocfg: dict,
                          out: CheckOutcome) -> float | None:
        try:
            from sklearn.ensemble import IsolationForest
        except Exception:  # noqa: BLE001 - optional dependency
            out.add("isolation_forest_unavailable", Severity.INFO,
                    "scikit-learn not installed; skipping Isolation Forest")
            return None

        X = num.dropna()
        if len(X) < 50:
            out.add("isolation_forest_skipped", Severity.INFO,
                    "too few complete rows for Isolation Forest")
            return None
        contamination = ocfg.get("contamination", "auto")
        try:
            model = IsolationForest(
                contamination=contamination, random_state=42, n_estimators=100,
            )
            preds = model.fit_predict(X.to_numpy())
        except Exception as exc:  # noqa: BLE001
            out.add("isolation_forest_failed", Severity.INFO,
                    f"Isolation Forest failed: {exc}")
            return None
        frac = float((preds == -1).mean())
        flag_pct = float(ocfg.get("column_flag_pct", 0.10))
        if frac > flag_pct:
            out.add("isolation_forest_outliers", Severity.WARN,
                    f"Isolation Forest flags {frac:.2%} of rows as anomalous",
                    pct=round(frac, 4), n_rows=len(X))
        return round(frac, 4)
