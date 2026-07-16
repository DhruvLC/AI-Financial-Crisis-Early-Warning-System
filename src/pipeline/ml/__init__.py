"""Machine Learning module for the Financial Crisis Early Warning System.

Consumes engineered features from the versioned feature store
(``data/features/``) and produces trained models, evaluation + explainability
reports (``reports/ml/``), and a model registry (``models/registry.json``).

Public API re-exported here so callers (runner, tests) import one place.
"""
from .base import BaseModel, EvaluationResult, MLError, TrainedModel
from .data_loader import MLDataLoader, MLDataset
from .evaluation import METRIC_NAMES, ModelEvaluator, ThresholdOptimizer
from .explain import ModelExplainer
from .models import MODEL_REGISTRY, build_model
from .pipeline import MLPipeline, MLResult
from .registry import ModelRegistry
from .report import MLReport
from .tuning import HyperparameterTuner, make_cv_splitter
from .visualization import MLVisualizer

__all__ = [
    "BaseModel", "EvaluationResult", "MLError", "TrainedModel",
    "MLDataLoader", "MLDataset", "METRIC_NAMES", "ModelEvaluator",
    "ThresholdOptimizer", "ModelExplainer", "MODEL_REGISTRY", "build_model",
    "MLPipeline", "MLResult", "ModelRegistry", "MLReport",
    "HyperparameterTuner", "make_cv_splitter", "MLVisualizer",
]
