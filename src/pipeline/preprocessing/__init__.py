"""Data Preparation (preprocessing) module.

A modular, fit-on-train, leak-safe preprocessing framework mirroring the design
of ``src/validation``. One :class:`~.base.PreprocessStep` subclass per concern
(cleaning, de-duplication, imputation, outliers, encoding, scaling), an
orchestrator that sequences them around the train/val/test split, a lineage
tracker, and a report writer.
"""
from .base import PreprocessingError, PreprocessStep, StepResult
from .lineage import LineageTracker
from .pipeline import PreprocessingPipeline, PreprocessResult
from .report import PreprocessingReport
from .steps import (
    CategoricalEncoder, DataCleaner, DuplicateRemover, FeatureScaler,
    Imputer, OutlierTreatment, POST_SPLIT_STEPS, PRE_SPLIT_STEPS, STEP_REGISTRY,
)

__all__ = [
    "PreprocessingError", "PreprocessStep", "StepResult",
    "LineageTracker", "PreprocessingPipeline", "PreprocessResult",
    "PreprocessingReport",
    "DataCleaner", "DuplicateRemover", "Imputer", "OutlierTreatment",
    "CategoricalEncoder", "FeatureScaler",
    "PRE_SPLIT_STEPS", "POST_SPLIT_STEPS", "STEP_REGISTRY",
]
