"""Preprocessing orchestrator.

Wires the individual :mod:`~pipeline.preprocessing.steps` into one leak-safe
pipeline that mirrors the diagram's Data Preparation stage:

    pre-split (clean → de-duplicate)  →  train/val/test split
        →  post-split fit on TRAIN (impute → outliers → encode → scale)
        →  apply fitted steps to VAL and TEST

Row-level operations (cleaning, de-duplication) run **before** the split on the
full frame so partitions never share records. Every learned transformation
(imputation, outlier bounds, encoders, scalers) is **fit on the train split
only** and re-applied to val/test — the standard guard against leakage.

Each step's :class:`~pipeline.preprocessing.base.StepResult` is threaded into a
:class:`~pipeline.preprocessing.lineage.LineageTracker`, and the whole run is
summarised by :class:`~pipeline.preprocessing.report.PreprocessingReport`.

Design deliberately parallels :class:`validation.runner.DataValidationRunner`:
config-driven construction, per-step exception isolation honouring ``fail_fast``,
uniform logging.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ingestion.logging_config import get_logger
from pipeline import data_prep

from .base import PreprocessingError
from .lineage import LineageTracker
from .steps import POST_SPLIT_STEPS, PRE_SPLIT_STEPS


@dataclass
class PreprocessResult:
    """Everything a preprocessing run produces."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    step_results: list = field(default_factory=list)   # list[StepResult]
    lineage: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)


class PreprocessingPipeline:
    """Run the full clean → split → fit-on-train → apply preprocessing flow."""

    def __init__(self, cfg: dict, target_col: str, spec=None) -> None:
        self.cfg = cfg or {}
        self.pp_cfg = dict(self.cfg.get("preprocessing", {}))
        self.target_col = target_col
        self.spec = spec
        self.log = get_logger("preprocessing.pipeline")
        self.fail_fast = bool(self.pp_cfg.get("fail_fast", True))
        self.lineage = LineageTracker()
        self.step_results: list = []

    # ── step construction ────────────────────────────────────────────────────
    def _build(self, cls):
        step_cfg = self.pp_cfg.get(cls.name, {}) or {}
        return cls(cfg=step_cfg, target_col=self.target_col, spec=self.spec)

    def _run_step(self, step, df: pd.DataFrame) -> pd.DataFrame:
        """Fit+transform one step, recording lineage; honour fail_fast."""
        before = df
        try:
            result = step.fit_transform(df)
        except PreprocessingError:
            if self.fail_fast:
                raise
            self.log.error("step '%s' failed; continuing (fail_fast off)",
                           step.name)
            return df
        self.step_results.append(result)
        self.lineage.record(result, before, result.df)
        return result.df

    def _apply_fitted(self, steps: list, df: pd.DataFrame) -> pd.DataFrame:
        """Apply already-fitted post-split steps to a held-out frame."""
        for step in steps:
            df = step.transform(df)
        return df

    # ── orchestration ────────────────────────────────────────────────────────
    def run(self, df: pd.DataFrame) -> PreprocessResult:
        if df is None or len(df) == 0:
            raise PreprocessingError("preprocessing received an empty dataset")
        if self.target_col not in df.columns:
            raise PreprocessingError(
                f"target_col '{self.target_col}' not present in the data")

        self.log.info("preprocessing start: %d rows x %d cols",
                      len(df), df.shape[1])
        self.lineage.start(df)

        # 1. Pre-split, row-level steps on the full frame --------------------
        for cls in PRE_SPLIT_STEPS:
            df = self._run_step(self._build(cls), df)

        # 2. Train / val / test split (reuse the existing splitter) ----------
        train, val, test = data_prep.split(df, self.cfg)

        # 3. Post-split steps: fit on TRAIN, apply to VAL/TEST ---------------
        fitted = []
        for cls in POST_SPLIT_STEPS:
            step = self._build(cls)
            train = self._run_step(step, train)
            fitted.append(step)
        val = self._apply_fitted(fitted, val)
        test = self._apply_fitted(fitted, test)

        # 4. Keep column sets consistent across splits (encoding may add
        #    train-only dummy columns; align val/test to train). ------------
        val = self._align_columns(train, val)
        test = self._align_columns(train, test)

        self.lineage.finish(train)
        self.log.info("preprocessing done: train=%d val=%d test=%d, %d cols",
                      len(train), len(val), len(test), train.shape[1])

        result = PreprocessResult(
            train=train, val=val, test=test,
            step_results=self.step_results,
            lineage=self.lineage.as_dict(),
        )
        # report built lazily by the caller/report module to avoid import cycle
        return result

    @staticmethod
    def _align_columns(train: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
        """Ensure ``other`` has exactly train's columns, in the same order."""
        other = other.copy()
        for col in train.columns:
            if col not in other.columns:
                other[col] = 0
        return other[list(train.columns)]
