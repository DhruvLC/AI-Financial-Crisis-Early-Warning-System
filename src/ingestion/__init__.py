"""Data ingestion module for the AI Financial Crisis Early Warning System.

Every data source is an :class:`~ingestion.base.BaseIngestor` subclass that
fetches → validates → stores raw + interim → writes metadata. The
:class:`~ingestion.runner.IngestionRunner` orchestrates all enabled sources
with per-source failure isolation.
"""
from .base import BaseIngestor, IngestionResult
from .validation import DataValidator, ValidationReport
from .cross_validation import (
    CrossSourceValidator,
    CrossSourceReport,
    CrossSourceValidationError,
    validate_corpus,
)
from .metadata import MetadataWriter

__all__ = [
    "BaseIngestor",
    "IngestionResult",
    "DataValidator",
    "ValidationReport",
    "CrossSourceValidator",
    "CrossSourceReport",
    "CrossSourceValidationError",
    "validate_corpus",
    "MetadataWriter",
]
