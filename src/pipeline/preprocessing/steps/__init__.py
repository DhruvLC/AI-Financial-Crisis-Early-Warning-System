"""Concrete preprocessing steps.

Ordering here defines the canonical pre-split → fit-on-train execution order:
clean → de-duplicate → impute → treat outliers → encode → scale.
"""
from .cleaning import DataCleaner
from .duplicates import DuplicateRemover
from .imputation import Imputer
from .outliers import OutlierTreatment
from .encoding import CategoricalEncoder
from .scaling import FeatureScaler

#: Pre-split steps run once on the full frame (row-level operations that must
#: happen before the train/val/test partition so splits never share records).
PRE_SPLIT_STEPS = [DataCleaner, DuplicateRemover]

#: Post-split steps are fit on the train split and applied to val/test.
POST_SPLIT_STEPS = [Imputer, OutlierTreatment, CategoricalEncoder, FeatureScaler]

#: Registry by name for config-driven construction and tests.
STEP_REGISTRY = {
    cls.name: cls
    for cls in (*PRE_SPLIT_STEPS, *POST_SPLIT_STEPS)
}

__all__ = [
    "DataCleaner", "DuplicateRemover", "Imputer", "OutlierTreatment",
    "CategoricalEncoder", "FeatureScaler",
    "PRE_SPLIT_STEPS", "POST_SPLIT_STEPS", "STEP_REGISTRY",
]
