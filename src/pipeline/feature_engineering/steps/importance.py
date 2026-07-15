"""Feature importance scoring — Random Forest, XGBoost, and SHAP.

Unlike the other steps, this one is **analysis-only**: it never changes the
frame (the returned ``df`` is the input untouched). It fits importance models
on the final engineered train split and reports per-feature scores that feed
the reports and the feature store's metadata:

* **Random Forest** — impurity-based ``feature_importances_``;
* **XGBoost** — gain-based importances (skipped gracefully if xgboost is not
  installed);
* **SHAP** — mean |SHAP value| per feature over a sampled subset (skipped
  gracefully if shap is not installed), computed against the XGBoost model
  when available, else the random forest.

Scores from each method are normalised to sum to 1 so they are comparable; a
consensus ``mean_rank`` orders the final table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ..base import FeatureEngineeringError, FeatureResult, FeatureStep


class FeatureImportance(FeatureStep):
    """Score features with RF / XGBoost / SHAP (analysis-only, no transform)."""

    name = "importance"

    def __init__(self, cfg=None, target_col=None, hints=None) -> None:
        super().__init__(cfg, target_col, hints)
        self.importances: pd.DataFrame | None = None   # feature x method

    def _fit_transform(self, df: pd.DataFrame) -> FeatureResult:
        X, y = self.split_xy(df)
        if y is None:
            raise FeatureEngineeringError(
                "feature importance requires the target column")
        if X.shape[1] == 0:
            return FeatureResult(step=self.name, df=df, skipped=True,
                                 skip_reason="no numeric features")

        scores: dict[str, pd.Series] = {}
        rf_model = xgb_model = None

        if self.cfg.get("random_forest", True):
            rf_model, s = self._random_forest(X, y)
            scores["random_forest"] = s
        if self.cfg.get("xgboost", True):
            xgb_model, s = self._xgboost(X, y)
            if s is not None:
                scores["xgboost"] = s
        if self.cfg.get("shap", True):
            s = self._shap(xgb_model or rf_model, X)
            if s is not None:
                scores["shap"] = s

        if not scores:
            return FeatureResult(step=self.name, df=df, skipped=True,
                                 skip_reason="no importance method available")

        table = pd.DataFrame(scores).fillna(0.0)
        table = table.div(table.sum(axis=0).replace(0, 1.0), axis=1)  # normalise
        table["mean_rank"] = table.rank(ascending=False).mean(axis=1)
        table = table.sort_values("mean_rank")
        self.importances = table

        top_n = int(self.cfg.get("report_top_n", 25))
        result = FeatureResult(step=self.name, df=df)   # frame unchanged
        result.params = {"methods": list(scores)}
        result.stats = {
            "methods": list(scores),
            "n_features_scored": int(table.shape[0]),
            "top_features": [
                {"feature": f, **{m: round(float(row[m]), 5)
                                  for m in scores}}
                for f, row in table.head(top_n).iterrows()],
        }
        result.note(f"scored {table.shape[0]} features via "
                    f"{', '.join(scores)}")
        return result

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df   # analysis-only step: identity on held-out splits

    # ── scorers ───────────────────────────────────────────────────────────────
    def _random_forest(self, X, y):
        rf = RandomForestClassifier(
            n_estimators=int(self.cfg.get("n_estimators", 300)),
            class_weight="balanced",
            random_state=int(self.cfg.get("random_state", 42)),
            n_jobs=-1).fit(X, y)
        return rf, pd.Series(rf.feature_importances_, index=X.columns)

    def _xgboost(self, X, y):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            self.log.info("xgboost not installed — skipping")
            return None, None
        pos = max(int((y == 1).sum()), 1)
        model = XGBClassifier(
            n_estimators=int(self.cfg.get("n_estimators", 300)),
            max_depth=6, learning_rate=0.05,
            scale_pos_weight=(len(y) - pos) / pos,
            random_state=int(self.cfg.get("random_state", 42)),
            n_jobs=-1, eval_metric="logloss").fit(X, y)
        return model, pd.Series(model.feature_importances_, index=X.columns)

    def _shap(self, model, X):
        if model is None:
            return None
        try:
            import shap
        except ImportError:
            self.log.info("shap not installed — skipping")
            return None
        try:
            n = min(int(self.cfg.get("shap_max_samples", 500)), len(X))
            sample = X.sample(n, random_state=int(
                self.cfg.get("random_state", 42)))
            values = shap.TreeExplainer(model).shap_values(sample)
            if isinstance(values, list):          # per-class list (RF)
                values = values[-1]
            if values.ndim == 3:                  # (rows, feats, classes)
                values = values[:, :, -1]
            return pd.Series(np.abs(values).mean(axis=0), index=X.columns)
        except Exception as exc:  # noqa: BLE001 - SHAP is best-effort
            self.log.warning("SHAP failed (%s) — skipping", exc)
            return None
