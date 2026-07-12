"""Categorical encoding — one-hot and ordinal encoding (fit on train).

The bankruptcy modelling table is fully numeric, but the broader ingested
sources (SEC EDGAR, news, macro panels) carry categorical/text columns
(``entity_id``, sector labels, tickers). This step turns those into numeric
features so every downstream model sees a purely numeric matrix.

It is **stateful and leak-safe**: the category vocabulary is learned on the
train split and re-applied to val/test. Categories unseen at fit time map to
an all-zero one-hot block (``handle_unknown="ignore"``) or to a configurable
sentinel for ordinal encoding, so held-out frames never crash the transform.

High-cardinality columns (more distinct values than ``max_cardinality``) are
left untouched and recorded in the report, to avoid exploding the feature space
with one-hot columns for free-text/identifier fields.

Config (``preprocessing.encoding``)::

    encoding:
      enabled: true
      method: onehot          # onehot | ordinal
      max_cardinality: 50      # skip columns with more distinct values
      drop_first: false        # drop one level (onehot) to avoid collinearity
      unknown_value: -1        # ordinal code for categories unseen at fit
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import PreprocessingError, PreprocessStep, StepResult

_METHODS = {"onehot", "ordinal"}


class CategoricalEncoder(PreprocessStep):
    """Encode categorical feature columns into numeric form (fit on train)."""

    name = "encoding"

    def __init__(self, cfg=None, target_col=None, spec=None) -> None:
        super().__init__(cfg, target_col, spec)
        self.method = str(self.cfg.get("method", "onehot")).lower()
        if self.method not in _METHODS:
            raise PreprocessingError(
                f"unknown encoding method '{self.method}'; "
                f"choose one of {sorted(_METHODS)}")
        self.max_cardinality = int(self.cfg.get("max_cardinality", 50))
        self.drop_first = bool(self.cfg.get("drop_first", False))
        self.unknown_value = self.cfg.get("unknown_value", -1)
        # fitted state
        self._encoded_cols: list[str] = []
        self._skipped_cols: list[str] = []
        self._categories: dict[str, list] = {}     # col -> ordered category list
        self._onehot_columns: list[str] = []        # full output column order

    # ── column discovery ─────────────────────────────────────────────────────
    def _categorical_features(self, df: pd.DataFrame) -> list[str]:
        feats = self.feature_columns(df)
        return [c for c in feats
                if not pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_datetime64_any_dtype(df[c])]

    def _fit_transform(self, df: pd.DataFrame) -> StepResult:
        df = df.copy()
        result = StepResult(step=self.name, df=df)
        cat_cols = self._categorical_features(df)

        if not cat_cols:
            return StepResult(step=self.name, df=df, skipped=True,
                              skip_reason="no categorical feature columns")

        # Partition by cardinality so identifier/free-text columns don't explode.
        for c in cat_cols:
            n_unique = int(df[c].astype("string").nunique(dropna=True))
            if n_unique > self.max_cardinality:
                self._skipped_cols.append(c)
            else:
                self._encoded_cols.append(c)
                cats = sorted(
                    df[c].astype("string").dropna().unique().tolist())
                self._categories[c] = cats

        new_cols = 0
        if self._encoded_cols:
            if self.method == "onehot":
                df = self._onehot_fit(df)
                new_cols = len(self._onehot_columns)
            else:
                df = self._ordinal_apply(df)
                new_cols = len(self._encoded_cols)

        result.df = df
        result.params = {
            "method": self.method,
            "encoded_columns": self._encoded_cols,
            "skipped_high_cardinality": self._skipped_cols,
            "max_cardinality": self.max_cardinality,
            "categories": {k: list(map(str, v))
                           for k, v in self._categories.items()},
        }
        result.stats = {
            "categorical_cols": len(cat_cols),
            "encoded_cols": len(self._encoded_cols),
            "skipped_high_cardinality_cols": len(self._skipped_cols),
            "new_feature_cols": new_cols,
        }
        result.note(
            f"encoded {len(self._encoded_cols)} categorical column(s) via "
            f"'{self.method}'"
            + (f"; skipped {len(self._skipped_cols)} high-cardinality column(s)"
               if self._skipped_cols else ""))
        return result

    # ── one-hot ──────────────────────────────────────────────────────────────
    def _onehot_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._onehot_encode(df)
        # Freeze the fitted dummy columns: everything the encode step added
        # on top of the untouched base columns.
        base_cols = [c for c in df.columns if c not in self._encoded_cols]
        self._onehot_columns = [c for c in out.columns if c not in base_cols]
        return out

    def _onehot_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        pieces = []
        for c in self._encoded_cols:
            s = df[c].astype("string")
            cats = self._categories[c]
            cat_dtype = pd.CategoricalDtype(categories=cats)
            dummies = pd.get_dummies(
                s.astype(cat_dtype), prefix=c, prefix_sep="=",
                dummy_na=False, drop_first=self.drop_first, dtype=np.int8)
            pieces.append(dummies)
        base = df.drop(columns=self._encoded_cols)
        if pieces:
            encoded = pd.concat([base] + pieces, axis=1)
        else:
            encoded = base
        return encoded

    # ── ordinal ──────────────────────────────────────────────────────────────
    def _ordinal_apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self._encoded_cols:
            mapping = {cat: i for i, cat in enumerate(self._categories[c])}
            s = df[c].astype("string")
            df[c] = s.map(mapping).fillna(self.unknown_value).astype(float)
        return df

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._encoded_cols:
            return df
        df = df.copy()
        if self.method == "onehot":
            encoded = self._onehot_encode(df)
            # Align to the fitted column set (add missing dummy cols as 0,
            # drop any spurious ones — categories unseen at fit are ignored).
            for col in self._onehot_columns:
                if col not in encoded.columns:
                    encoded[col] = np.int8(0)
            base_cols = [c for c in encoded.columns
                         if c not in self._onehot_columns]
            ordered = base_cols + self._onehot_columns
            return encoded[ordered]
        return self._ordinal_apply(df)
