"""Core datatypes for the Data Preparation (preprocessing) module.

This mirrors the design of ``src/validation/base.py``: a small set of shared
containers plus a :class:`PreprocessStep` template that every concrete step
subclasses. Keeping the plumbing here lets the individual steps under
``pipeline.preprocessing.steps`` stay small, uniform, and independently
testable.

Key differences from the validation framework: a preprocessing step *transforms*
data (it returns a new DataFrame) and is **stateful** — it is ``fit`` on the
training split and then ``apply``-ed to validation/test to avoid leakage. The
:class:`StepResult` it emits carries both the transformed frame and a
machine-readable record of what was done, which feeds the reports and the data
lineage.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ingestion.logging_config import get_logger


class PreprocessingError(RuntimeError):
    """Raised for fatal preprocessing problems (empty data, invalid columns,
    failed transformations) when the pipeline is configured to fail fast."""


@dataclass
class StepResult:
    """Everything one step reports after transforming a frame.

    ``df`` is the transformed data. ``params`` captures the fitted parameters
    (e.g. imputation fill values, scaler statistics) so the transformation is
    reproducible and auditable. ``stats`` carries human/report-facing numbers
    (rows removed, columns encoded). ``transformation`` is the compact lineage
    record appended to the :class:`~pipeline.preprocessing.lineage.LineageTracker`.
    """

    step: str
    df: pd.DataFrame
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
            "params": self.params,
            "stats": self.stats,
        }

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "status": "skipped" if self.skipped else "applied",
            "skip_reason": self.skip_reason,
            "params": self.params,
            "stats": self.stats,
            "notes": self.notes,
        }


class PreprocessStep(abc.ABC):
    """Template for a fit/apply preprocessing step.

    Subclasses set :attr:`name` and implement :meth:`_fit_transform` (learn
    parameters on the training frame and return its transformed version) and,
    when the step learns state, :meth:`_transform` (apply the already-fitted
    parameters to another frame). Steps that do not carry state across splits
    (row-level cleaning, duplicate removal) only need :meth:`_fit_transform`
    and inherit an identity :meth:`_transform`.

    The public :meth:`fit_transform` / :meth:`transform` wrap execution in
    uniform logging and exception isolation, exactly like
    :class:`validation.base.BaseCheck.run`. A crashing step raises
    :class:`PreprocessingError` (so the orchestrator can honour ``fail_fast``)
    rather than leaking an arbitrary exception type.
    """

    #: short machine name, e.g. "imputation"; used for logging + report keys
    name: str = "base"

    def __init__(self, cfg: dict | None = None, target_col: str | None = None,
                 spec: Any = None) -> None:
        self.cfg = cfg or {}
        self.target_col = target_col
        self.spec = spec
        self.log = get_logger(f"preprocessing.{self.name}")
        self._fitted = False

    # ── overridable hooks ────────────────────────────────────────────────────
    def enabled(self) -> bool:
        """Steps are on by default; a step is skipped when its config block
        sets ``enabled: false``."""
        return bool(self.cfg.get("enabled", True))

    @abc.abstractmethod
    def _fit_transform(self, df: pd.DataFrame) -> StepResult:
        """Learn parameters on ``df`` (the train split) and return its
        transformed version wrapped in a :class:`StepResult`."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply already-fitted parameters to another frame (val/test).

        Stateless steps (row filters) keep the identity default; stateful steps
        (imputation, scaling, encoding) override this.
        """
        return df

    # ── feature-column helper shared by every step ───────────────────────────
    def feature_columns(self, df: pd.DataFrame) -> list[str]:
        """All columns except the target (which is never transformed)."""
        return [c for c in df.columns if c != self.target_col]

    # ── public API with logging + exception isolation ────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> StepResult:
        if not self.enabled():
            self.log.info("step disabled via config — skipping")
            return StepResult(step=self.name, df=df, skipped=True,
                              skip_reason="disabled via config")
        if df is None or len(df) == 0:
            raise PreprocessingError(
                f"step '{self.name}' received an empty dataset")
        try:
            result = self._fit_transform(df)
            self._fitted = True
            if result.skipped:
                self.log.info("skipped: %s", result.skip_reason)
            else:
                self.log.info("applied on %d rows x %d cols -> %d rows x %d cols",
                              len(df), df.shape[1],
                              len(result.df), result.df.shape[1])
            return result
        except PreprocessingError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise to our error type
            self.log.exception("step '%s' crashed during fit: %s", self.name, exc)
            raise PreprocessingError(
                f"step '{self.name}' failed during fit: "
                f"{type(exc).__name__}: {exc}") from exc

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted step to a held-out frame."""
        if not self.enabled() or not self._fitted:
            return df
        if df is None or len(df) == 0:
            # An empty val/test split is unusual but not fatal on its own.
            self.log.warning("transform received an empty frame — returning as-is")
            return df
        try:
            return self._transform(df)
        except Exception as exc:  # noqa: BLE001
            self.log.exception("step '%s' crashed during transform: %s",
                               self.name, exc)
            raise PreprocessingError(
                f"step '{self.name}' failed during transform: "
                f"{type(exc).__name__}: {exc}") from exc
