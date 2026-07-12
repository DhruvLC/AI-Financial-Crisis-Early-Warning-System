"""Core datatypes for the Exploratory Data Analysis (EDA) module.

Mirrors the design of ``src/pipeline/preprocessing/base.py`` and
``src/validation/base.py``: a small set of shared containers plus an
:class:`EdaAnalyzer` template that every concrete analyzer subclasses. Keeping
the plumbing here lets the individual analyzers under
``pipeline.eda.analyzers`` stay small, uniform, and independently testable.

An analyzer *reads* a (already preprocessed) DataFrame and *produces* an
:class:`AnalysisResult` carrying tables (for CSV/MD/JSON export), figure paths,
and a machine-readable summary that feeds the reports and the business-insights
engine. Analyzers never mutate their input.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ingestion.logging_config import get_logger


class EdaError(RuntimeError):
    """Raised for fatal EDA problems (empty data, missing target, invalid
    columns, corrupted datasets) when the runner is configured to fail fast."""


@dataclass
class AnalysisResult:
    """Everything one analyzer reports.

    ``tables`` maps a short name to a DataFrame the report/statistics writers
    persist (CSV/MD/JSON). ``figures`` lists PNG paths produced by the analyzer.
    ``summary`` is the compact, machine-readable record consumed by the report
    and the :class:`~pipeline.eda.analyzers.insights.BusinessInsightsEngine`.
    """

    analyzer: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None

    def note(self, message: str) -> None:
        self.notes.append(message)

    def as_dict(self) -> dict:
        return {
            "analyzer": self.analyzer,
            "status": "skipped" if self.skipped else "completed",
            "skip_reason": self.skip_reason,
            "figures": list(self.figures),
            "tables": sorted(self.tables.keys()),
            "summary": self.summary,
            "notes": self.notes,
        }


class EdaAnalyzer(abc.ABC):
    """Template for one EDA analyzer.

    Subclasses set :attr:`name` and implement :meth:`_analyze`. The public
    :meth:`run` wraps execution in uniform logging and exception isolation
    exactly like :class:`validation.base.BaseCheck.run` and
    :class:`pipeline.preprocessing.base.PreprocessStep.fit_transform`: a crashing
    analyzer raises :class:`EdaError` so the runner can honour ``fail_fast``
    rather than leaking an arbitrary exception type.
    """

    #: short machine name, e.g. "descriptive"; used for logging + report keys
    name: str = "base"

    def __init__(self, cfg: dict | None = None, target_col: str | None = None,
                 figures: Any = None) -> None:
        self.cfg = cfg or {}
        self.target_col = target_col
        self.figures = figures            # a FigureManager (or None for no plots)
        self.log = get_logger(f"eda.{self.name}")

    # ── overridable hooks ────────────────────────────────────────────────────
    def enabled(self) -> bool:
        """Analyzers are on by default; skipped when config sets ``enabled: false``."""
        return bool(self.cfg.get("enabled", True))

    def feature_columns(self, df: pd.DataFrame) -> list[str]:
        """All columns except the target (never analyzed as a feature)."""
        return [c for c in df.columns if c != self.target_col]

    def numeric_features(self, df: pd.DataFrame) -> list[str]:
        feats = self.feature_columns(df)
        return df[feats].select_dtypes(include="number").columns.tolist()

    def categorical_features(self, df: pd.DataFrame) -> list[str]:
        feats = self.feature_columns(df)
        return [c for c in feats
                if not pd.api.types.is_numeric_dtype(df[c])]

    @abc.abstractmethod
    def _analyze(self, df: pd.DataFrame) -> AnalysisResult:
        """Produce the analysis for ``df`` (never mutates ``df``)."""

    # ── public API with logging + exception isolation ────────────────────────
    def run(self, df: pd.DataFrame) -> AnalysisResult:
        if not self.enabled():
            self.log.info("analyzer disabled via config — skipping")
            return AnalysisResult(analyzer=self.name, skipped=True,
                                  skip_reason="disabled via config")
        if df is None or len(df) == 0:
            raise EdaError(f"analyzer '{self.name}' received an empty dataset")
        if self.target_col is not None and self.target_col not in df.columns:
            raise EdaError(
                f"analyzer '{self.name}': target '{self.target_col}' not present")
        try:
            result = self._analyze(df)
            n_fig = len(result.figures)
            if result.skipped:
                self.log.info("skipped: %s", result.skip_reason)
            else:
                self.log.info("completed — %d table(s), %d figure(s)",
                              len(result.tables), n_fig)
            return result
        except EdaError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise to our error type
            self.log.exception("analyzer '%s' crashed: %s", self.name, exc)
            raise EdaError(
                f"analyzer '{self.name}' failed: "
                f"{type(exc).__name__}: {exc}") from exc
