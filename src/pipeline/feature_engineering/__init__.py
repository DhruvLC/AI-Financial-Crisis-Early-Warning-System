"""Feature Engineering module.

A modular, fit-on-train, leak-safe feature-engineering framework mirroring the
design of ``pipeline.preprocessing``. One :class:`~.base.FeatureStep` subclass
per concern (generation, multicollinearity filtering, selection, dimensionality
reduction, importance scoring), an orchestrator that fits them on the train
split and applies them to val/test, EDA-hint integration, a lineage tracker, a
versioned feature store, and a report writer.
"""
from .base import FeatureEngineeringError, FeatureResult, FeatureStep
from .eda_insights import EdaInsightLoader
from .lineage import FeatureLineageTracker
from .pipeline import FeatureEngineeringPipeline, FeatureEngineeringResult
from .report import FeatureEngineeringReport
from .store import FeatureStore
from .steps import (
    DimensionalityReduction, FeatureGeneration, FeatureImportance,
    FeatureSelection, FEATURE_STEPS, MulticollinearityFilter, STEP_REGISTRY,
)

__all__ = [
    "FeatureEngineeringError", "FeatureResult", "FeatureStep",
    "EdaInsightLoader", "FeatureLineageTracker",
    "FeatureEngineeringPipeline", "FeatureEngineeringResult",
    "FeatureEngineeringReport", "FeatureStore",
    "FeatureGeneration", "MulticollinearityFilter", "FeatureSelection",
    "DimensionalityReduction", "FeatureImportance",
    "FEATURE_STEPS", "STEP_REGISTRY",
]
