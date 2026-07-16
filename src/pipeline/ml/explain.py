"""Explainability for the ML module.

Three complementary views of a trained model, each degrading gracefully:

* native feature importance (tree importances / |coefficients|),
* permutation importance (model-agnostic, on the validation split),
* SHAP values (if the optional ``shap`` package is installed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ingestion.logging_config import get_logger

from .base import BaseModel

try:  # optional back-end
    import shap
    _HAS_SHAP, _SHAP_REASON = True, None
except Exception as exc:  # noqa: BLE001 - any import failure counts
    _HAS_SHAP, _SHAP_REASON = False, f"shap unavailable: {exc}"

log = get_logger("ml.explain")


class ModelExplainer:
    """Produce importance tables + SHAP summaries for a trained model."""

    def __init__(self, cfg: dict | None = None,
                 random_state: int = 42) -> None:
        cfg = cfg or {}
        self.random_state = random_state
        self.permutation_repeats = int(cfg.get("permutation_repeats", 5))
        self.shap_enabled = bool(cfg.get("shap", True))
        self.shap_max_samples = int(cfg.get("shap_max_samples", 500))
        self.top_n = int(cfg.get("report_top_n", 25))

    @staticmethod
    def shap_available() -> tuple[bool, str | None]:
        return _HAS_SHAP, _SHAP_REASON

    def native_importance(self, model: BaseModel) -> pd.DataFrame | None:
        return model.native_importance()

    def permutation(self, model: BaseModel, X: pd.DataFrame,
                    y: pd.Series) -> pd.DataFrame | None:
        """Permutation importance on held-out data (None on failure)."""
        try:
            result = permutation_importance(
                model.estimator, X, y, scoring="roc_auc",
                n_repeats=self.permutation_repeats,
                random_state=self.random_state, n_jobs=-1)
            return (pd.DataFrame({"feature": list(X.columns),
                                  "importance": result.importances_mean,
                                  "std": result.importances_std})
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True))
        except Exception as exc:  # noqa: BLE001 - explainability is best-effort
            log.warning("permutation importance failed for %s: %s",
                        model.name, exc)
            return None

    def shap_summary(self, model: BaseModel,
                     X: pd.DataFrame) -> dict | None:
        """Mean-|SHAP| per feature (None if shap missing or unsupported)."""
        if not self.shap_enabled:
            return None
        if not _HAS_SHAP:
            log.warning("SHAP skipped: %s", _SHAP_REASON)
            return None
        sample = X.sample(min(len(X), self.shap_max_samples),
                          random_state=self.random_state)
        try:
            try:  # fast exact path for tree models
                explainer = shap.TreeExplainer(model.estimator)
                values = explainer.shap_values(sample)
            except Exception:  # noqa: BLE001 - fall back to model-agnostic
                explainer = shap.Explainer(model.predict_proba, sample)
                values = explainer(sample).values
            if isinstance(values, list):             # per-class list → class 1
                values = values[-1]
            values = np.asarray(values)
            if values.ndim == 3:                     # (n, features, classes)
                values = values[:, :, -1]
            mean_abs = np.abs(values).mean(axis=0)
            order = np.argsort(mean_abs)[::-1]
            cols = list(sample.columns)
            return {"n_samples": len(sample),
                    "mean_abs_shap": {cols[i]: float(mean_abs[i])
                                      for i in order}}
        except Exception as exc:  # noqa: BLE001
            log.warning("SHAP failed for %s: %s", model.name, exc)
            return None

    def explain(self, model: BaseModel, X_val: pd.DataFrame,
                y_val: pd.Series) -> dict:
        """All three views in one call — keys may map to None."""
        log.info("explaining %s (permutation repeats=%d, shap=%s)",
                 model.name, self.permutation_repeats,
                 self.shap_enabled and _HAS_SHAP)
        return {
            "native": self.native_importance(model),
            "permutation": self.permutation(model, X_val, y_val),
            "shap": self.shap_summary(model, X_val),
        }
