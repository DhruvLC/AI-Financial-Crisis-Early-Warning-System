"""Business-insights engine.

Reads the machine-readable ``summary`` block of every analyzer's
:class:`~pipeline.eda.base.AnalysisResult` and distills a small set of
prioritized, plain-English findings and modelling recommendations for the
Financial Crisis Early-Warning System. This is deliberately *not* an
``EdaAnalyzer`` — it consumes the other analyzers' output rather than the raw
DataFrame, so it always runs last.

Each insight is a dict ``{severity, category, message, recommendation}``.
Severity is one of ``critical | warning | info`` and drives ordering in the
reports. The engine never raises: a missing/oddly-shaped summary yields fewer
insights, not a crash.
"""
from __future__ import annotations

from typing import Any

from ingestion.logging_config import get_logger

_SEVERITY_RANK = {"critical": 0, "warning": 1, "info": 2}


class BusinessInsightsEngine:
    """Synthesize prioritized business insights from analyzer summaries."""

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}
        self.log = get_logger("eda.insights")

    def generate(self, results: dict) -> dict[str, Any]:
        """``results`` maps analyzer name → :class:`AnalysisResult`."""
        insights: list[dict] = []

        def summ(name: str) -> dict:
            r = results.get(name)
            if r is None or getattr(r, "skipped", False):
                return {}
            return getattr(r, "summary", {}) or {}

        for fn in (self._target, self._missing, self._distributions,
                   self._correlation, self._outliers, self._ratios,
                   self._relationships, self._dimensionality):
            try:
                fn(summ, insights)
            except Exception as exc:  # noqa: BLE001 - one bad rule never aborts
                self.log.warning("insight rule %s failed: %s",
                                 fn.__name__, exc)

        insights.sort(key=lambda d: _SEVERITY_RANK.get(d["severity"], 9))
        counts = {s: sum(1 for i in insights if i["severity"] == s)
                  for s in ("critical", "warning", "info")}
        return {
            "n_insights": len(insights),
            "severity_counts": counts,
            "insights": insights,
            "headline": self._headline(insights, counts),
        }

    # ── individual rule blocks ────────────────────────────────────────────────
    def _add(self, out, severity, category, message, recommendation):
        out.append({"severity": severity, "category": category,
                    "message": message, "recommendation": recommendation})

    def _target(self, summ, out):
        s = summ("target")
        if not s:
            return
        ratio = s.get("imbalance_ratio")
        pos_pct = s.get("positive_pct")
        if s.get("is_imbalanced"):
            self._add(out, "critical", "class_imbalance",
                      f"Severe class imbalance: {ratio}:1 "
                      f"(positive class only {pos_pct}% of samples).",
                      "Use class weighting or resampling (SMOTE/undersampling) "
                      "and evaluate with PR-AUC / recall, not accuracy.")
        elif ratio and ratio >= 2:
            self._add(out, "warning", "class_imbalance",
                      f"Moderate class imbalance ({ratio}:1).",
                      "Prefer stratified splits and threshold tuning; monitor "
                      "minority-class recall.")

    def _missing(self, summ, out):
        s = summ("missing")
        if not s:
            return
        n = s.get("n_features_with_missing", 0)
        pct = s.get("overall_missing_pct", 0)
        if n == 0:
            self._add(out, "info", "data_quality",
                      "No missing values in the processed dataset.",
                      "No imputation needed downstream.")
        elif pct >= 5:
            self._add(out, "warning", "data_quality",
                      f"{n} feature(s) contain missing values "
                      f"({pct}% of cells overall).",
                      "Confirm the imputation strategy is appropriate before "
                      "modelling; consider missingness indicators.")

    def _distributions(self, summ, out):
        s = summ("distributions")
        if not s:
            return
        skew = s.get("n_highly_skewed", 0)
        heavy = s.get("n_heavy_tailed", 0)
        if skew:
            self._add(out, "warning", "feature_distribution",
                      f"{skew} feature(s) are highly skewed and {heavy} are "
                      f"heavy-tailed.",
                      "Apply log/Box-Cox/Yeo-Johnson transforms or robust "
                      "scaling for linear/distance-based models.")

    def _correlation(self, summ, out):
        s = summ("correlation")
        if not s:
            return
        pairs = s.get("n_high_correlation_pairs", 0)
        vif = s.get("n_high_vif", 0)
        if pairs or vif:
            self._add(out, "warning", "multicollinearity",
                      f"{pairs} highly-correlated feature pair(s) and {vif} "
                      f"feature(s) with high VIF detected.",
                      "Drop/combine redundant ratios or use regularized / "
                      "tree-based models robust to collinearity.")

    def _outliers(self, summ, out):
        s = summ("outliers")
        if not s:
            return
        mean_pct = s.get("mean_iqr_outlier_pct", 0)
        iso = s.get("isolation_forest", {}) or {}
        if mean_pct and mean_pct >= 5:
            self._add(out, "warning", "outliers",
                      f"Features average {mean_pct:.1f}% IQR outliers after "
                      f"preprocessing.",
                      "Revisit winsorization limits or use outlier-robust "
                      "models; extreme ratios may be economically meaningful.")
        if iso.get("available") and iso.get("anomaly_pct", 0) >= 1:
            self._add(out, "info", "outliers",
                      f"Isolation-Forest flags {iso['anomaly_pct']}% of rows "
                      f"as multivariate anomalies.",
                      "Inspect flagged firms — they may be genuine distress "
                      "cases or data-entry errors.")

    def _ratios(self, summ, out):
        s = summ("ratios")
        if not s:
            return
        top = s.get("top_discriminative") or []
        if top:
            names = ", ".join(str(t["feature"])[:40] for t in top[:5])
            self._add(out, "info", "financial_signal",
                      f"Most bankruptcy-discriminative ratios: {names}.",
                      "Prioritize these (and their categories) in feature "
                      "selection and in the model-explainability narrative.")
        strength = s.get("category_strength") or {}
        if strength:
            best = max(strength, key=strength.get)
            self._add(out, "info", "financial_signal",
                      f"'{best}' is the strongest financial category for "
                      f"separating bankrupt vs solvent firms.",
                      "Ensure this category is well represented after any "
                      "feature pruning.")

    def _relationships(self, summ, out):
        s = summ("relationships")
        if not s:
            return
        top = s.get("top_by_separability") or []
        if top:
            best = top[0]
            self._add(out, "info", "predictive_power",
                      f"Strongest univariate predictor: "
                      f"'{str(best['feature'])[:40]}' "
                      f"(AUC {best.get('univariate_auc', 0):.3f}).",
                      "Even single ratios carry signal; expect tree ensembles "
                      "to exploit their interactions.")
        if s.get("mean_separability", 0) < 0.05:
            self._add(out, "warning", "predictive_power",
                      "Univariate feature-target separability is generally weak.",
                      "Signal likely lives in feature interactions — favour "
                      "non-linear models (gradient boosting).")

    def _dimensionality(self, summ, out):
        s = summ("dimensionality")
        if not s:
            return
        n95 = s.get("n_components_95pct")
        nfeat = s.get("n_features")
        red = s.get("redundancy_index")
        if n95 and nfeat and n95 < 0.5 * nfeat:
            self._add(out, "info", "dimensionality",
                      f"Only {n95} of {nfeat} components explain 95% of "
                      f"variance (redundancy index {red}).",
                      "Consider PCA/feature selection to cut dimensionality "
                      "and speed up training without losing much signal.")

    @staticmethod
    def _headline(insights, counts) -> str:
        if counts["critical"]:
            crit = next(i for i in insights if i["severity"] == "critical")
            return crit["message"]
        if counts["warning"]:
            warn = next(i for i in insights if i["severity"] == "warning")
            return warn["message"]
        return "No critical data issues detected; dataset looks model-ready."
