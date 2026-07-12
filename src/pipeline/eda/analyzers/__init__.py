"""Concrete EDA analyzers + a name→class registry.

``DEFAULT_ORDER`` fixes the sequence the runner executes (and the report
renders). Overview/target/descriptive first (context), then quality
(missing/outliers), then structure (distributions/correlation), then the
domain and modelling lenses (ratios/relationships/dimensionality).
"""
from .correlation import CorrelationAnalysis
from .descriptive import DescriptiveStatistics
from .dimensionality import DimensionalityAnalysis
from .distributions import FeatureDistributionAnalysis
from .missing import MissingValueAnalysis
from .outliers import OutlierAnalysis
from .overview import DatasetOverview
from .ratios import FinancialRatioAnalysis
from .relationships import FeatureRelationshipAnalysis
from .target import TargetAnalysis

ANALYZER_REGISTRY: dict = {
    "overview": DatasetOverview,
    "target": TargetAnalysis,
    "descriptive": DescriptiveStatistics,
    "missing": MissingValueAnalysis,
    "outliers": OutlierAnalysis,
    "distributions": FeatureDistributionAnalysis,
    "correlation": CorrelationAnalysis,
    "ratios": FinancialRatioAnalysis,
    "relationships": FeatureRelationshipAnalysis,
    "dimensionality": DimensionalityAnalysis,
}

DEFAULT_ORDER: list = [
    "overview", "target", "descriptive", "missing", "outliers",
    "distributions", "correlation", "ratios", "relationships",
    "dimensionality",
]

__all__ = [
    "ANALYZER_REGISTRY", "DEFAULT_ORDER",
    "DatasetOverview", "TargetAnalysis", "DescriptiveStatistics",
    "MissingValueAnalysis", "OutlierAnalysis", "FeatureDistributionAnalysis",
    "CorrelationAnalysis", "FinancialRatioAnalysis",
    "FeatureRelationshipAnalysis", "DimensionalityAnalysis",
]
