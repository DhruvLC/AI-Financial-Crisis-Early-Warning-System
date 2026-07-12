"""Concrete validation checks, each a :class:`~validation.base.BaseCheck`."""
from .schema import SchemaValidator
from .missing import MissingValueAnalyzer
from .duplicates import DuplicateDetector
from .outliers import OutlierDetector
from .financial import FinancialValidator
from .timeseries import TimeSeriesValidator

#: default check order (schema first so downstream findings have context)
DEFAULT_CHECKS = [
    SchemaValidator,
    MissingValueAnalyzer,
    DuplicateDetector,
    OutlierDetector,
    FinancialValidator,
    TimeSeriesValidator,
]

__all__ = [
    "SchemaValidator",
    "MissingValueAnalyzer",
    "DuplicateDetector",
    "OutlierDetector",
    "FinancialValidator",
    "TimeSeriesValidator",
    "DEFAULT_CHECKS",
]
