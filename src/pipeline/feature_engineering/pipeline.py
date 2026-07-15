"""Feature-engineering orchestrator.

Wires the individual :mod:`~pipeline.feature_engineering.steps` into one
leak-safe pipeline over the already-preprocessed train/val/test splits:

    load EDA hints  →  fit on TRAIN
        (generate → drop multicollinear → select → reduce → score importance)
    →  apply the fitted steps to VAL and TEST
    →  align columns  →  persist to the feature store

Every step is **fit on the train split only** and re-applied to val/test.
Per-step :class:`~pipeline.feature_engineering.base.FeatureResult` records feed
a :class:`~pipeline.feature_engineering.lineage.FeatureLineageTracker`, and the
run is summarised by
:class:`~pipeline.feature_engineering.report.FeatureEngineeringReport`.

Design deliberately parallels
:class:`pipeline.preprocessing.pipeline.PreprocessingPipeline`: config-driven
construction, per-step exception isolation honouring ``fail_fast``, uniform
logging.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ingestion.logging_config import get_logger

from .base import FeatureEngineeringError
from .eda_insights import EdaInsightLoader
from .lineage import FeatureLineageTracker
from .steps import FEATURE_STEPS
from .store import FeatureStore


@dataclass
class FeatureEngineeringResult:
    """Everything a feature-engineering run produces."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    step_results: list = field(default_factory=list)   # list[FeatureResult]
    lineage: dict = field(default_factory=dict)
    hints: dict = field(default_factory=dict)
    importances: pd.DataFrame | None = None
    store_record: dict | None = None
    report: dict = field(default_factory=dict)


class FeatureEngineeringPipeline:
    """Run the full generate → filter → select → reduce → score flow."""

    def __init__(self, cfg: dict, target_col: str) -> None:
        self.cfg = cfg or {}
        self.fe_cfg = dict(self.cfg.get("feature_engineering", {}))
        self.target_col = target_col
        self.log = get_logger("features.pipeline")
        self.fail_fast = bool(self.fe_cfg.get("fail_fast", True))
        self.lineage = FeatureLineageTracker()
        self.step_results: list = []

    # ── step construction ────────────────────────────────────────────────────
    def _build(self, cls, hints: dict):
        step_cfg = self.fe_cfg.get(cls.name, {}) or {}
        return cls(cfg=step_cfg, target_col=self.target_col, hints=hints)

    def _run_step(self, step, df: pd.DataFrame) -> pd.DataFrame:
        """Fit+transform one step, recording lineage; honour fail_fast."""
        before = df
        try:
            result = step.fit_transform(df)
        except FeatureEngineeringError:
            if self.fail_fast:
                raise
            self.log.error("step '%s' failed; continuing (fail_fast off)",
                           step.name)
            return df
        self.step_results.append(result)
        self.lineage.record(result, before, result.df)
        return result.df

    # ── orchestration ────────────────────────────────────────────────────────
    def run(self, train: pd.DataFrame, val: pd.DataFrame,
            test: pd.DataFrame) -> FeatureEngineeringResult:
        for name, df in (("train", train), ("val", val), ("test", test)):
            if df is None or len(df) == 0:
                raise FeatureEngineeringError(
                    f"feature engineering received an empty '{name}' split")
        if self.target_col not in train.columns:
            raise FeatureEngineeringError(
                f"target_col '{self.target_col}' not present in train")

        self.log.info("feature engineering start: train=%d x %d, "
                      "val=%d, test=%d", len(train), train.shape[1],
                      len(val), len(test))
        self.lineage.start(train)

        # EDA hints (best-effort; empty when the report is absent).
        hints = EdaInsightLoader(
            self.fe_cfg.get("eda_report_path",
                            "reports/eda/eda_report.json")).load()

        # Fit each step on TRAIN, then apply the fitted steps to VAL/TEST.
        importances = None
        fitted = []
        for cls in FEATURE_STEPS:
            step = self._build(cls, hints)
            train = self._run_step(step, train)
            fitted.append(step)
            if step.name == "importance":
                importances = getattr(step, "importances", None)
        for step in fitted:
            val = step.transform(val)
            test = step.transform(test)

        # Keep column sets identical across splits (generation on val/test
        # can only differ if source columns were missing — align to train).
        val = self._align_columns(train, val)
        test = self._align_columns(train, test)

        self.lineage.finish(train)
        self.log.info("feature engineering done: train=%d val=%d test=%d, "
                      "%d cols", len(train), len(val), len(test),
                      train.shape[1])

        result = FeatureEngineeringResult(
            train=train, val=val, test=test,
            step_results=self.step_results,
            lineage=self.lineage.as_dict(),
            hints=hints, importances=importances,
        )

        # Persist to the feature store unless disabled.
        store_cfg = self.fe_cfg.get("store", {}) or {}
        if store_cfg.get("enabled", True):
            store = FeatureStore(store_cfg.get("dir", "data/features"))
            result.store_record = store.save(
                {"train": train, "val": val, "test": test},
                metadata={
                    "target_col": self.target_col,
                    "n_features": train.shape[1] - 1,
                    "features": [c for c in train.columns
                                 if c != self.target_col],
                    "lineage": result.lineage,
                    "config": self.fe_cfg,
                })
        return result

    @staticmethod
    def _align_columns(train: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
        """Ensure ``other`` has exactly train's columns, in the same order."""
        other = other.copy()
        for col in train.columns:
            if col not in other.columns:
                other[col] = 0.0
        return other[list(train.columns)]
