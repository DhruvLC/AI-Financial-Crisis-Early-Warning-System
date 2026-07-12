"""Feature scaling / normalization — standard, min-max, robust (fit on train).

Standardizes numeric feature columns so scale-sensitive models (logistic
regression, SVM, neural nets, KNN) see comparable ranges, and so SHAP/coefficient
magnitudes are interpretable. Tree ensembles are scale-invariant, so this step
is optional and toggleable, but it is fit-on-train and re-applied to val/test to
stay leak-free — the same discipline as :class:`~.imputation.Imputer`.

The scaler statistics (mean/std, min/max, median/IQR) are captured in
``StepResult.params`` so the transformation is reproducible and auditable, and
so the fitted scaler can be persisted alongside the model.

Config (``preprocessing.scaling``)::

    scaling:
      enabled: true
      method: standard         # standard | minmax | robust | none
      feature_range: [0, 1]     # only used when method == minmax
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler, RobustScaler, StandardScaler)

from ..base import PreprocessingError, PreprocessStep, StepResult

_METHODS = {"standard", "minmax", "robust", "none"}


class FeatureScaler(PreprocessStep):
    """Scale numeric features with a configurable method (fit on train)."""

    name = "scaling"

    def __init__(self, cfg=None, target_col=None, spec=None) -> None:
        super().__init__(cfg, target_col, spec)
        self.method = str(self.cfg.get("method", "standard")).lower()
        if self.method not in _METHODS:
            raise PreprocessingError(
                f"unknown scaling method '{self.method}'; "
                f"choose one of {sorted(_METHODS)}")
        self.feature_range = tuple(self.cfg.get("feature_range", [0, 1]))
        self._num_cols: list[str] = []
        self._scaler = None

    def _numeric_features(self, df: pd.DataFrame) -> list[str]:
        feats = self.feature_columns(df)
        return df[feats].select_dtypes(include=[np.number]).columns.tolist()

    def _make_scaler(self):
        if self.method == "standard":
            return StandardScaler()
        if self.method == "minmax":
            return MinMaxScaler(feature_range=self.feature_range)
        return RobustScaler()   # "robust"

    def _fit_transform(self, df: pd.DataFrame) -> StepResult:
        df = df.copy()
        self._num_cols = self._numeric_features(df)

        if self.method == "none" or not self._num_cols:
            return StepResult(step=self.name, df=df, skipped=True,
                              skip_reason="no numeric features or method=none")

        self._scaler = self._make_scaler()
        df[self._num_cols] = self._scaler.fit_transform(df[self._num_cols])

        result = StepResult(step=self.name, df=df)
        result.params = {
            "method": self.method,
            "numeric_columns": self._num_cols,
            "statistics": self._scaler_stats(),
        }
        result.stats = {
            "method": self.method,
            "numeric_cols_scaled": len(self._num_cols),
        }
        result.note(f"scaled {len(self._num_cols)} numeric column(s) via "
                    f"'{self.method}'")
        return result

    def _scaler_stats(self) -> dict:
        """Human/report-facing per-column fitted statistics."""
        cols = self._num_cols
        s = self._scaler
        if self.method == "standard":
            return {"mean": self._pairs(cols, s.mean_),
                    "scale": self._pairs(cols, s.scale_)}
        if self.method == "minmax":
            return {"data_min": self._pairs(cols, s.data_min_),
                    "data_max": self._pairs(cols, s.data_max_)}
        # robust
        return {"center": self._pairs(cols, s.center_),
                "scale": self._pairs(cols, s.scale_)}

    @staticmethod
    def _pairs(cols: list[str], values) -> dict:
        return {c: float(v) for c, v in zip(cols, np.asarray(values).ravel())}

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._scaler is None:
            return df
        df = df.copy()
        cols = [c for c in self._num_cols if c in df.columns]
        if len(cols) != len(self._num_cols):
            # Reorder/guard: sklearn scalers need the exact fitted column set.
            missing = set(self._num_cols) - set(cols)
            if missing:
                raise PreprocessingError(
                    f"scaling.transform missing fitted columns: {sorted(missing)}")
        df[self._num_cols] = self._scaler.transform(df[self._num_cols])
        return df
