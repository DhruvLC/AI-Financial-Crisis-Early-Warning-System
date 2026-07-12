"""Outlier treatment — winsorization, IQR clipping, z-score filtering,
Isolation Forest filtering, selected by config.

The audit found ``pipeline.data_prep.clip_outliers`` implemented only std-based
*clipping* under a "winsorize" misnomer. This step provides the four requested
methods with the correct semantics and leak-safe fitting:

* **clip** methods (``winsorize``, ``iqr_clip``, ``zscore``) cap extreme values.
  Bounds are learned on the **train** split and re-applied to val/test, so the
  transformation is deterministic and leak-free. The legacy
  ``data_prep.clip_outliers`` is reused verbatim as the ``zscore_clip`` method
  (no logic duplicated).
* **filter** methods (``zscore_filter``, ``isolation_forest``) *remove* rows.
  Rows are only ever dropped from the **train** split — val/test are returned
  untouched by :meth:`_transform` so the held-out evaluation set is preserved.

Isolation Forest reuses scikit-learn (already a hard dependency here) and, like
``validation.checks.outliers``, degrades gracefully if a fit fails.

Config (``preprocessing.outliers``)::

    outliers:
      enabled: true
      method: winsorize        # winsorize | iqr_clip | zscore_clip
                               #   | zscore_filter | isolation_forest | none
      winsor_limits: [0.01, 0.99]   # lower/upper quantiles for winsorize
      iqr_multiplier: 1.5           # IQR fence for iqr_clip
      zscore_threshold: 5.0         # |z| cap (clip) or cut (filter)
      contamination: auto           # isolation_forest anomaly fraction
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import data_prep  # reuse the existing z-score clipper

from ..base import PreprocessingError, PreprocessStep, StepResult

_CLIP_METHODS = {"winsorize", "iqr_clip", "zscore_clip"}
_FILTER_METHODS = {"zscore_filter", "isolation_forest"}
_METHODS = _CLIP_METHODS | _FILTER_METHODS | {"none"}


class OutlierTreatment(PreprocessStep):
    """Cap or remove outliers using a configurable method (fit on train)."""

    name = "outliers"

    def __init__(self, cfg=None, target_col=None, spec=None) -> None:
        super().__init__(cfg, target_col, spec)
        self.method = str(self.cfg.get("method", "winsorize")).lower()
        if self.method not in _METHODS:
            raise PreprocessingError(
                f"unknown outlier method '{self.method}'; "
                f"choose one of {sorted(_METHODS)}")
        self._num_cols: list[str] = []
        self._lower: pd.Series | None = None   # fitted clip bounds
        self._upper: pd.Series | None = None

    def _numeric_features(self, df: pd.DataFrame) -> list[str]:
        feats = self.feature_columns(df)
        return df[feats].select_dtypes(include=[np.number]).columns.tolist()

    # ── bound fitting for clip methods ───────────────────────────────────────
    def _fit_bounds(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        num = df[self._num_cols]
        if self.method == "winsorize":
            lo_q, hi_q = self.cfg.get("winsor_limits", [0.01, 0.99])
            lower = num.quantile(lo_q)
            upper = num.quantile(hi_q)
        elif self.method == "iqr_clip":
            mult = float(self.cfg.get("iqr_multiplier", 1.5))
            q1, q3 = num.quantile(0.25), num.quantile(0.75)
            iqr = (q3 - q1)
            lower, upper = q1 - mult * iqr, q3 + mult * iqr
        else:  # zscore_clip — mirror data_prep.clip_outliers bounds
            z = float(self.cfg.get("zscore_threshold", 5.0))
            mean, std = num.mean(), num.std().replace(0, 1)
            lower, upper = mean - z * std, mean + z * std
        return lower, upper

    def _fit_transform(self, df: pd.DataFrame) -> StepResult:
        df = df.copy()
        result = StepResult(step=self.name, df=df)
        self._num_cols = self._numeric_features(df)

        if self.method == "none" or not self._num_cols:
            return result.__class__(step=self.name, df=df, skipped=True,
                                    skip_reason="no numeric features or method=none")

        before = len(df)
        if self.method in _CLIP_METHODS:
            if self.method == "zscore_clip":
                # Reuse the existing implementation wholesale.
                z = float(self.cfg.get("zscore_threshold", 5.0))
                df = data_prep.clip_outliers(df, self.target_col, z=z)
            # Record fitted bounds (also used to transform val/test).
            self._lower, self._upper = self._fit_bounds(df if False else result.df
                                                        if False else df)
            self._lower, self._upper = self._fit_bounds(
                result.df if False else df)
            if self.method != "zscore_clip":
                df[self._num_cols] = df[self._num_cols].clip(
                    lower=self._lower, upper=self._upper, axis=1)
            n_clipped = int(((df[self._num_cols].to_numpy() ==
                              self._upper.to_numpy()) |
                             (df[self._num_cols].to_numpy() ==
                              self._lower.to_numpy())).sum())
            result.stats = {
                "method": self.method,
                "rows": before,
                "numeric_cols": len(self._num_cols),
                "cells_at_bound": n_clipped,
            }
            result.note(f"clipped outliers via '{self.method}' on "
                        f"{len(self._num_cols)} numeric column(s)")
        else:  # filter methods — drop train rows only
            mask = self._outlier_mask(df)
            removed = int(mask.sum())
            df = df[~mask].reset_index(drop=True)
            result.stats = {
                "method": self.method,
                "rows_before": before,
                "rows_after": len(df),
                "rows_removed": removed,
                "removed_pct": round(removed / max(before, 1), 4),
            }
            result.note(f"removed {removed} outlier row(s) via '{self.method}'")

        result.df = df
        result.params = {
            "method": self.method,
            "numeric_columns": self._num_cols,
            "lower_bounds": None if self._lower is None
            else {k: float(v) for k, v in self._lower.items()},
            "upper_bounds": None if self._upper is None
            else {k: float(v) for k, v in self._upper.items()},
        }
        return result

    def _outlier_mask(self, df: pd.DataFrame) -> np.ndarray:
        """Boolean row mask of outliers for the filter methods."""
        num = df[self._num_cols].replace([np.inf, -np.inf], np.nan)
        if self.method == "zscore_filter":
            z_thresh = float(self.cfg.get("zscore_threshold", 5.0))
            std = num.std(ddof=0).replace(0, np.nan)
            z = (num - num.mean()).abs() / std
            return (z > z_thresh).any(axis=1).to_numpy()
        # isolation_forest
        return self._isolation_forest_mask(num)

    def _isolation_forest_mask(self, num: pd.DataFrame) -> np.ndarray:
        try:
            from sklearn.ensemble import IsolationForest
        except Exception:  # noqa: BLE001 - optional at runtime
            self.log.warning("scikit-learn unavailable; skipping Isolation Forest")
            return np.zeros(len(num), dtype=bool)

        X = num.fillna(num.median())
        if len(X) < 50:
            self.log.info("too few rows (%d) for Isolation Forest — skipping",
                          len(X))
            return np.zeros(len(num), dtype=bool)
        contamination = self.cfg.get("contamination", "auto")
        try:
            model = IsolationForest(
                contamination=contamination, random_state=42, n_estimators=100)
            preds = model.fit_predict(X.to_numpy())
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Isolation Forest failed (%s); no rows removed", exc)
            return np.zeros(len(num), dtype=bool)
        return preds == -1

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Clip methods re-apply fitted bounds; filter methods must NOT drop
        # rows from held-out val/test, so they are a no-op here.
        if self.method in _CLIP_METHODS and self._lower is not None:
            df = df.copy()
            cols = [c for c in self._num_cols if c in df.columns]
            df[cols] = df[cols].clip(lower=self._lower[cols],
                                     upper=self._upper[cols], axis=1)
            return df
        return df
