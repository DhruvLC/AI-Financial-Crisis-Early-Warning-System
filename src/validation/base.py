"""Core datatypes for the Data Validation module.

Defines the severity model, the finding/outcome containers each check emits,
the per-dataset report, and the :class:`BaseCheck` template every concrete
check subclasses. Keeping these here means the individual checks under
``validation.checks`` stay small and uniform.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import pandas as pd

from ingestion.logging_config import get_logger


class Severity(IntEnum):
    """Ordered severity levels (higher = worse)."""
    INFO = 0
    WARN = 1
    ERROR = 2

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass
class Finding:
    """A single validation observation produced by a check."""
    code: str
    level: Severity
    message: str
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {"code": self.code, "level": self.level.label, "message": self.message}
        if self.details:
            d["details"] = self.details
        return d


@dataclass
class CheckOutcome:
    """Everything one check reports for one dataset.

    ``metrics`` carries machine-readable numbers (fractions, counts) that the
    :class:`~validation.quality.QualityScorer` consumes; ``findings`` carries
    the human-facing observations.
    """
    check: str
    findings: list[Finding] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str | None = None

    # ── ergonomic helpers used by the checks ────────────────────────────────
    def add(self, code: str, level: Severity, message: str, **details: Any) -> None:
        self.findings.append(Finding(code, level, message, details))

    def skip(self, reason: str) -> "CheckOutcome":
        self.skipped = True
        self.skip_reason = reason
        return self

    @property
    def worst(self) -> Severity:
        return max((f.level for f in self.findings), default=Severity.INFO)

    def as_dict(self) -> dict:
        return {
            "check": self.check,
            "status": "skipped" if self.skipped else self.worst.label,
            "skip_reason": self.skip_reason,
            "metrics": self.metrics,
            "findings": [f.as_dict() for f in self.findings],
        }


@dataclass
class DatasetReport:
    """Aggregated validation result for a single dataset/source."""
    source: str
    present: bool
    interim_path: str | None = None
    n_rows: int = 0
    n_cols: int = 0
    load_error: str | None = None
    outcomes: list[CheckOutcome] = field(default_factory=list)
    quality_score: float = 0.0
    quality_grade: str = "F"
    quality_components: dict = field(default_factory=dict)

    @property
    def n_errors(self) -> int:
        return sum(f.level == Severity.ERROR
                   for o in self.outcomes for f in o.findings)

    @property
    def n_warnings(self) -> int:
        return sum(f.level == Severity.WARN
                   for o in self.outcomes for f in o.findings)

    @property
    def is_valid(self) -> bool:
        return self.present and self.load_error is None and self.n_errors == 0

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "present": self.present,
            "interim_path": self.interim_path,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "load_error": self.load_error,
            "is_valid": self.is_valid,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "quality": {
                "score": round(self.quality_score, 2),
                "grade": self.quality_grade,
                "components": {k: round(v, 4)
                               for k, v in self.quality_components.items()},
            },
            "checks": [o.as_dict() for o in self.outcomes],
        }


class BaseCheck(abc.ABC):
    """Template for a validation check.

    Subclasses set :attr:`name` and implement :meth:`_run`. The public
    :meth:`run` wraps execution in uniform logging + exception isolation so a
    single crashing check never aborts a dataset's whole validation.
    """

    #: short machine name, e.g. "schema"; used for logging + report keys
    name: str = "base"

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}
        self.log = get_logger(f"validation.{self.name}")

    def applicable(self, df: pd.DataFrame, spec: Any) -> bool:
        """Override to skip a check when it doesn't apply to this dataset."""
        return True

    @abc.abstractmethod
    def _run(self, df: pd.DataFrame, spec: Any, ctx: dict) -> CheckOutcome:
        """Do the actual work; return a populated :class:`CheckOutcome`."""

    def run(self, df: pd.DataFrame, spec: Any, ctx: dict) -> CheckOutcome:
        outcome = CheckOutcome(check=self.name)
        if not self.applicable(df, spec):
            return outcome.skip("not applicable to this source")
        try:
            return self._run(df, spec, ctx)
        except Exception as exc:  # noqa: BLE001 - isolate a check failure
            self.log.exception("check '%s' crashed: %s", self.name, exc)
            outcome.add("check_crashed", Severity.ERROR,
                        f"check raised {type(exc).__name__}: {exc}")
            return outcome
