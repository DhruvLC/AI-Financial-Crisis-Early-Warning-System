"""Data Validation module for the AI Financial Crisis Early Warning System.

A deep, per-dataset validation framework that runs over the ingested interim
datasets (``data/interim/``) and reports schema, missing-value, duplicate,
outlier, financial-sanity, and time-series findings plus a 0-100 quality score.

It complements the two lighter validation layers already in the project:
    * ``ingestion.validation.DataValidator``   — per-source inline gate
    * ``ingestion.cross_validation``            — corpus coverage/consistency

Entry points:
    * :class:`~validation.runner.DataValidationRunner`  (programmatic / CLI)
    * individual checks under :mod:`validation.checks`  (composable)
"""
from .base import (
    BaseCheck, CheckOutcome, DatasetReport, Finding, Severity,
)
from .runner import DataValidationRunner, DataValidationError
from .quality import QualityScorer
from .report import ReportGenerator
from .schemas import SOURCE_SCHEMAS, SourceSchema, schema_for

__all__ = [
    "BaseCheck",
    "CheckOutcome",
    "DatasetReport",
    "Finding",
    "Severity",
    "DataValidationRunner",
    "DataValidationError",
    "QualityScorer",
    "ReportGenerator",
    "SOURCE_SCHEMAS",
    "SourceSchema",
    "schema_for",
]
