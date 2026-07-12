"""Outlier analysis — IQR, z-score and (optional) Isolation-Forest detection.

Quantifies univariate outliers per numeric feature via two complementary
rules — the Tukey IQR fence and the |z|-score threshold — and (optionally) a
multivariate Isolation-Forest pass that flags whole *rows* as anomalies. The
output feeds both the report (which features are outlier-heavy) and the
business-insights engine (data-quality / treatment recommendations).

Note the processed dataset has usually already been winsorized/clipped by the
preprocessing stage; residual outliers here therefore describe the *post-
treatment* distribution and are expected to be modest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import AnalysisResult, EdaAnalyzer


class OutlierAnalysis(EdaAnalyzer):
    """Per-feature IQR/z-score outlier rates + optional multivariate anomalies."""

    name = "outliers"

    def _analyze(self, df: pd.DataFrame) -> AnalysisResult:
        result = AnalysisResult(analyzer=self.name)
        num_cols = self.numeric_features(df)
        if not num_cols:
            return AnalysisResult(analyzer=self.name, skipped=True,
                                  skip_reason="no numeric features")

        iqr_mult = float(self.cfg.get("iqr_multiplier", 1.5))
        z_thresh = float(self.cfg.get("zscore_threshold", 3.0))
        n = len(df)

        rows = []
        for c in num_cols:
            s = df[c].dropna()
            if s.empty:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
            iqr_out = int(((s < lo) | (s > hi)).sum())
            std = s.std()
            if std and std > 0:
                z_out = int((np.abs((s - s.mean()) / std) >= z_thresh).sum())
            else:
                z_out = 0
            rows.append({
                "feature": c,
                "iqr_outliers": iqr_out,
                "iqr_outlier_pct": round(iqr_out / n * 100, 4),
                "zscore_outliers": z_out,
                "zscore_outlier_pct": round(z_out / n * 100, 4),
                "lower_fence": float(lo),
                "upper_fence": float(hi),
            })

        tbl = pd.DataFrame(rows).sort_values(
            "iqr_outlier_pct", ascending=False).reset_index(drop=True)
        result.tables["outlier_summary"] = tbl

        summary = {
            "iqr_multiplier": iqr_mult,
            "zscore_threshold": z_thresh,
            "n_features": len(tbl),
            "total_iqr_outliers": int(tbl["iqr_outliers"].sum()),
            "total_zscore_outliers": int(tbl["zscore_outliers"].sum()),
            "mean_iqr_outlier_pct": float(tbl["iqr_outlier_pct"].mean()),
            "top_outlier_features": tbl.head(15)[
                ["feature", "iqr_outlier_pct", "zscore_outlier_pct"]
            ].to_dict("records"),
        }

        # Optional multivariate anomaly detection (whole rows).
        if bool(self.cfg.get("isolation_forest", True)):
            summary["isolation_forest"] = self._isolation_forest(df, num_cols,
                                                                 result)

        result.summary = summary

        if self.figures is not None:
            self._chart(tbl, result)

        result.note(f"{summary['total_iqr_outliers']} IQR outlier cell(s); "
                    f"mean {summary['mean_iqr_outlier_pct']:.2f}% per feature")
        return result

    # ── multivariate anomalies ───────────────────────────────────────────────
    def _isolation_forest(self, df, num_cols, result) -> dict:
        try:
            from sklearn.ensemble import IsolationForest
        except Exception as exc:  # noqa: BLE001 - sklearn optional
            result.note(f"IsolationForest unavailable ({exc})")
            return {"available": False}

        contamination = self.cfg.get("contamination", "auto")
        X = df[num_cols].fillna(df[num_cols].median(numeric_only=True))
        try:
            iso = IsolationForest(
                n_estimators=int(self.cfg.get("if_n_estimators", 200)),
                contamination=contamination, random_state=42, n_jobs=-1)
            labels = iso.fit_predict(X)          # -1 == anomaly
            scores = iso.score_samples(X)
        except Exception as exc:  # noqa: BLE001 - never crash the analyzer
            result.note(f"IsolationForest failed ({exc})")
            return {"available": False}

        n_anom = int((labels == -1).sum())
        return {
            "available": True,
            "n_anomalies": n_anom,
            "anomaly_pct": round(n_anom / len(df) * 100, 4),
            "min_score": float(scores.min()),
            "mean_score": float(scores.mean()),
        }

    # ── figure ────────────────────────────────────────────────────────────────
    def _chart(self, tbl: pd.DataFrame, result: AnalysisResult) -> None:
        import seaborn as sns
        top = tbl.head(20)
        if top.empty:
            return
        with self.figures.figure(figsize=(10, max(4, 0.4 * len(top)))) as fig:
            ax = fig.add_subplot(111)
            sns.barplot(data=top, y="feature", x="iqr_outlier_pct", ax=ax,
                        hue="feature", legend=False, palette="flare")
            ax.set_title("Top features by IQR outlier rate")
            ax.set_xlabel("% rows flagged as outliers (IQR fence)")
            ax.set_ylabel("")
            ax.tick_params(labelsize=7)
            path = self.figures.save(fig, "outlier_rates")
        result.figures.append(path)
