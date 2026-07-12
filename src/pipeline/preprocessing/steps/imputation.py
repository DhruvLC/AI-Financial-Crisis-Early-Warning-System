"""Missing-value handling: mean / median / KNN imputation.

Extends the median-only ``fillna`` in ``pipeline.data_prep.clean`` into a
configurable, fit-on-train imputer. Numeric columns are imputed with the
configured strategy; categorical columns are always imputed with their most
frequent value (imputing a category with a mean is undefined). Fitting on the
train split and re-applying to val/test keeps the transformation leak-free.

Config (``preprocessing.imputation``)::

    imputation:
      enabled: true
      strategy: median        # mean | median | knn
      knn_neighbors: 5        # only used when strategy == knn
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

from ..base import PreprocessingError, PreprocessStep, StepResult

_NUMERIC_STRATEGIES = {"mean", "median", "knn"}


class Imputer(PreprocessStep):
    """Fill missing values with a configurable strategy (fit on train)."""

    name = "imputation"

    def __init__(self, cfg=None, target_col=None, spec=None) -> None:
        super().__init__(cfg, target_col, spec)
        self.strategy = str(self.cfg.get("strategy", "median")).lower()
        if self.strategy not in _NUMERIC_STRATEGIES:
            raise PreprocessingError(
                f"unknown imputation strategy '{self.strategy}'; "
                f"choose one of {sorted(_NUMERIC_STRATEGIES)}")
        self.knn_neighbors = int(self.cfg.get("knn_neighbors", 5))
        self._numeric_cols: list[str] = []
        self._categorical_cols: list[str] = []
        self._numeric_imputer = None
        self._categorical_imputer = None

    def _split_columns(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        feats = self.feature_columns(df)
        numeric = df[feats].select_dtypes(include=[np.number]).columns.tolist()
        categorical = [c for c in feats if c not in numeric]
        return numeric, categorical

    def _fit_transform(self, df: pd.DataFrame) -> StepResult:
        df = df.copy()
        result = StepResult(step=self.name, df=df)
        self._numeric_cols, self._categorical_cols = self._split_columns(df)

        missing_before = int(df[self.feature_columns(df)].isna().sum().sum())

        # ── numeric columns ──────────────────────────────────────────────────
        if self._numeric_cols:
            if self.strategy == "knn":
                self._numeric_imputer = KNNImputer(n_neighbors=self.knn_neighbors)
            else:
                self._numeric_imputer = SimpleImputer(strategy=self.strategy)
            df[self._numeric_cols] = self._numeric_imputer.fit_transform(
                df[self._numeric_cols])

        # ── categorical columns (always most-frequent) ──────────────────────
        if self._categorical_cols:
            self._categorical_imputer = SimpleImputer(strategy="most_frequent")
            df[self._categorical_cols] = self._categorical_imputer.fit_transform(
                df[self._categorical_cols])

        missing_after = int(df[self.feature_columns(df)].isna().sum().sum())
        result.df = df
        result.params = {
            "strategy": self.strategy,
            "knn_neighbors": self.knn_neighbors if self.strategy == "knn" else None,
            "numeric_columns": self._numeric_cols,
            "categorical_columns": self._categorical_cols,
        }
        result.stats = {
            "missing_before": missing_before,
            "missing_after": missing_after,
            "cells_imputed": missing_before - missing_after,
            "numeric_cols_imputed": len(self._numeric_cols),
            "categorical_cols_imputed": len(self._categorical_cols),
        }
        result.note(
            f"imputed {missing_before - missing_after} cell(s) "
            f"via '{self.strategy}' (numeric) + most_frequent (categorical)")
        return result

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Guard against columns that vanished/renamed between splits.
        num = [c for c in self._numeric_cols if c in df.columns]
        cat = [c for c in self._categorical_cols if c in df.columns]
        if self._numeric_imputer is not None and num:
            df[num] = self._numeric_imputer.transform(df[num])
        if self._categorical_imputer is not None and cat:
            df[cat] = self._categorical_imputer.transform(df[cat])
        return df
