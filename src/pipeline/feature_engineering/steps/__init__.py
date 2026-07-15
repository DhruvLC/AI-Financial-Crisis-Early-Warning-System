"""Concrete feature-engineering steps.

Ordering defines the canonical fit-on-train execution order:
generate → drop multicollinear → select → reduce → score importance.
"""
from .generation import FeatureGeneration
from .multicollinearity import MulticollinearityFilter
from .selection import FeatureSelection
from .reduction import DimensionalityReduction
from .importance import FeatureImportance

#: Fit-on-train execution order (also the report order).
FEATURE_STEPS = [
    FeatureGeneration,
    MulticollinearityFilter,
    FeatureSelection,
    DimensionalityReduction,
    FeatureImportance,
]

#: Registry by name for config-driven construction and tests.
STEP_REGISTRY = {cls.name: cls for cls in FEATURE_STEPS}

__all__ = [
    "FeatureGeneration", "MulticollinearityFilter", "FeatureSelection",
    "DimensionalityReduction", "FeatureImportance",
    "FEATURE_STEPS", "STEP_REGISTRY",
]
