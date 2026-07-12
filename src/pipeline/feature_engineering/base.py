"""Core datatypes for the Feature Engineering module.

Mirrors :mod:`pipeline.preprocessing.base` and :mod:`pipeline.eda.base`: a
small set of shared containers plus a :class:`FeatureStep` template that every
concrete step (generation, multicollinearity handling, selection, reduction)
subclasses. Keeping the plumbing here lets the individual steps under
``pipeline.feature_engineering.steps`` stay small, uniform, and independently
testable.

Like the preprocessing steps, a feature step is **stateful and leak-safe**: it
is ``fit`` on the training split (learning which columns to keep/drop/create and
any fitted transformers) and then ``apply``-ed to the validation/test splits so
no information from held-out data ever leaks into the fitted feature set. The
:class:`FeatureResult` it emits carries the transformed frame plus a
machine-readable record — which features were *generated*, *selected*, and
*removed*, and why — feeding the reports, the lineage trail, and the metadata
store.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ingestion.logging_config import get_logger


class FeatureEngineeringError(RuntimeError):
    """Raised for fatal feature-engineering problems (empty data, missing
    target, invalid columns, failed transforms) when the pipeline is configured
    to fail fast."""


@dataclass
class FeatureResult:
    """Everything one feature step reports after transforming a frame.

    ``df`` is the transformed data (target column preserved). ``params``
    captures fitted state needed to reproduce the transform on val/test
    (kept columns, fitted estimators' selections, PCA components, …).
    ``stats`` carries report-facing numbers. The ``generated`` / ``selected`` /
    ``removed`` lists document the step's effect on the feature set.
    """

    step: str
    df: pd.DataFrame
    generated: list[str] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None

    def note(self, message: str) -> None:
        self.notes.append(message)

    @property
    def transformation(self) -> dict:
        """Compact record for the lineage trail."""
        return {
            "step": self.step,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "n_generated": len(self.generated),
            "n_removed": len(self.removed),
            "params": self.params,
            "stats": self.stats,
        }

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "status": "skipped" if self.skipped else "applied",
            "skip_reason": self.skip_reason,
            "n_generated": len(self.generated),
            "n_selected": len(self.selected),
            "n_removed": len(self.removed),
            "generated": self.generated,
            "removed": self.removed,
            "stats": self.stats,
            "notes": self.notes,
        }


class FeatureStep(abc.ABC):
    """Template for a fit/apply feature-engineering step.

    Subclasses set :attr:`name` and implement :meth:`_fit_transform` (learn
    which features to create/keep/drop on the train frame and return its
    transformed version). When the step carries column state across splits it
    overrides :meth:`_transform` to re-apply the fitted decision to val/test.

    The public :meth:`fit_transform` / :meth:`transform` wrap execution in
    uniform logging and exception isolation exactly like
    :class:`pipeline.preprocessing.base.PreprocessStep`: a crashing step raises
    :class:`FeatureEngineeringError` so the orchestrator can honour
    ``fail_fast`` rather than leaking an arbitrary exception type.
    """

    #: short machine name, e.g. "selection"; used for logging + report keys
    name: str = "base"

    def __init__(self, cfg: dict | None = None, target_col: str | None = None,
                 hints: dict | None = None) -> None:
        self.cfg = cfg or {}
        self.target_col = target_col
        #: EDA-derived hints (see :mod:`pipeline.feature_engineering.eda_insights`)
        self.hints = hints or {}
        self.log = get_logger(f"features.{self.name}")
        self._fitted = False

    # ── overridable hooks ────────────────────────────────────────────────────
    def enabled(self) -> bool:
        """Steps are on by default; skipped when config sets ``enabled: false``."""
        return bool(self.cfg.get("enabled", True))

    def feature_columns(self, df: pd.DataFrame) -> list[str]:
        """All columns except the target (which is never transformed)."""
        return [c for c in df.columns if c != self.target_col]

    def numeric_features(self, df: pd.DataFrame) -> list[str]:
        feats = self.feature_columns(df)
        return df[feats].select_dtypes(include=[np.number]).columns.tolist()

    def split_xy(self, df: pd.DataFrame):
        """Return ``(X_numeric_features, y)`` for supervised steps."""
        X = df[self.numeric_features(df)]
        y = df[self.target_col] if self.target_col in df.columns else None
        return X, y

    @abc.abstractmethod
    def _fit_transform(self, df: pd.DataFrame) -> FeatureResult:
        """Learn on ``df`` (the train split) and return its transformed version."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply already-fitted parameters to another frame (val/test).

        Default keeps only the columns retained on train (plus the target),
        which is the correct behaviour for selection/removal steps. Steps that
        *create* columns (generation, PCA) override this.
        """
        keep = [c for c in self._kept_columns if c in df.columns]
        return df[keep]

    #: set by :meth:`fit_transform`; the exact column set present after fit.
    _kept_columns: list[str] = []

    # ── public API with logging + exception isolation ────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> FeatureResult:
        if not self.enabled():
            self.log.info("step disabled via config — skipping")
            return FeatureResult(step=self.name, df=df, skipped=True,
                                 skip_reason="disabled via config")
        if df is None or len(df) == 0:
            raise FeatureEngineeringError(
                f"step '{self.name}' received an empty dataset")
        if self.target_col is not None and self.target_col not in df.columns:
            raise FeatureEngineeringError(
                f"step '{self.name}': target '{self.target_col}' not present")
        try:
            result = self._fit_transform(df)
            self._kept_columns = list(result.df.columns)
            self._fitted = True
            if result.skipped:
                self.log.info("skipped: %s", result.skip_reason)
            else:
                self.log.info(
                    "applied on %d x %d -> %d x %d "
                    "(+%d generated, -%d removed)",
                    len(df), df.shape[1], len(result.df), result.df.shape[1],
                    len(result.generated), len(result.removed))
            return result
        except FeatureEngineeringError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise to our error type
            self.log.exception("step '%s' crashed during fit: %s",
                               self.name, exc)
            raise FeatureEngineeringError(
                f"step '{self.name}' failed during fit: "
                f"{type(exc).__name__}: {exc}") from exc

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted step to a held-out frame."""
        if not self.enabled() or not self._fitted:
            return df
        if df is None or len(df) == 0:
            self.log.warning("transform received an empty frame — returning as-is")
            return df
        try:
            return self._transform(df)
        except Exception as exc:  # noqa: BLE001
            self.log.exception("step '%s' crashed during transform: %s",
                               self.name, exc)
            raise FeatureEngineeringError(
                f"step '{self.name}' failed during transform: "
                f"{type(exc).__name__}: {exc}") from exc
