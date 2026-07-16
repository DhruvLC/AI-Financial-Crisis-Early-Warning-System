"""Data loading for the Machine Learning module.

Loads engineered train/val/test splits from the versioned
:class:`~pipeline.feature_engineering.store.FeatureStore` and verifies —
before any training starts — that the splits are usable: schema consistency
across splits, feature alignment with the stored metadata, target
availability, no NaN/inf values, and metadata (content-hash) integrity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ingestion.logging_config import get_logger
from pipeline.feature_engineering.store import FeatureStore

from .base import MLError

log = get_logger("ml.data")

REQUIRED_SPLITS = ("train", "val", "test")


@dataclass
class MLDataset:
    """The verified, ready-to-train view of one feature-store version."""

    version: str
    target_col: str
    features: list[str]
    splits: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)

    def xy(self, split: str) -> tuple[pd.DataFrame, pd.Series]:
        df = self.splits[split]
        return df[self.features], df[self.target_col]


class MLDataLoader:
    """Load + verify engineered splits from the feature store."""

    def __init__(self, store_dir: str = "data/features",
                 target_col: str = "Bankrupt?") -> None:
        self.store = FeatureStore(store_dir)
        self.target_col = target_col

    def load(self, version: str | None = None) -> MLDataset:
        """Return a verified :class:`MLDataset` (raises :class:`MLError`)."""
        try:
            splits, metadata = self.store.load(version)
        except FileNotFoundError as exc:
            raise MLError(f"feature store empty/unreadable: {exc}") from exc

        target = metadata.get("target_col", self.target_col)
        checks = self._verify(splits, metadata, target)
        features = [c for c in splits["train"].columns if c != target]
        ds = MLDataset(version=metadata.get("version", "unknown"),
                       target_col=target, features=features,
                       splits=splits, metadata=metadata, checks=checks)
        log.info("loaded feature-store %s: %s | %d features | target=%r",
                 ds.version,
                 ", ".join(f"{n}={len(d)}" for n, d in splits.items()),
                 len(features), target)
        return ds

    # ── verification ──────────────────────────────────────────────────────────
    def _verify(self, splits: dict[str, pd.DataFrame], metadata: dict,
                target: str) -> dict:
        checks: dict = {}

        missing = [s for s in REQUIRED_SPLITS if s not in splits]
        if missing:
            raise MLError(f"feature store missing split(s): {missing}")
        empty = [n for n, d in splits.items() if d.empty]
        if empty:
            raise MLError(f"empty split(s) in feature store: {empty}")
        checks["splits_present"] = True

        # target availability (must be present and binary in every split)
        for name, df in splits.items():
            if target not in df.columns:
                raise MLError(f"target '{target}' missing from split '{name}'")
            classes = set(pd.unique(df[target].dropna()))
            if not classes <= {0, 1}:
                raise MLError(
                    f"target '{target}' in split '{name}' is not binary "
                    f"(values: {sorted(classes)[:5]})")
        checks["target_available"] = True

        # schema consistency / feature alignment across splits
        ref = list(splits["train"].columns)
        for name, df in splits.items():
            if list(df.columns) != ref:
                raise MLError(
                    f"schema mismatch: split '{name}' columns differ from "
                    f"train (train={len(ref)} cols, {name}={df.shape[1]})")
        checks["schema_consistent"] = True

        stored = metadata.get("features")
        if stored is not None:
            current = [c for c in ref if c != target]
            if list(stored) != current:
                raise MLError(
                    "feature alignment failed: stored feature list differs "
                    "from the loaded splits")
        checks["features_aligned"] = stored is not None

        # data sanity: no NaN / inf in features or target
        for name, df in splits.items():
            numeric = df.select_dtypes(include=[np.number])
            n_nan = int(df.isna().sum().sum())
            n_inf = int(np.isinf(numeric.to_numpy()).sum())
            if n_nan or n_inf:
                raise MLError(
                    f"split '{name}' contains {n_nan} NaN / {n_inf} inf "
                    "values — rerun preprocessing/feature engineering")
        checks["no_nan_inf"] = True

        # metadata integrity: recorded content hashes must match the frames
        hashes = metadata.get("hashes", {})
        mismatched = [n for n, expect in hashes.items()
                      if n in splits and
                      FeatureStore._hash(splits[n]) != expect]
        if mismatched:
            raise MLError(
                f"metadata integrity failed: content hash mismatch for "
                f"split(s) {mismatched}")
        checks["hashes_verified"] = bool(hashes)

        log.info("feature-store verification passed: %s", checks)
        return checks
