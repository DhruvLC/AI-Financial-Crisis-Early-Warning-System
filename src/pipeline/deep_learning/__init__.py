"""Deep Learning module for the Financial Crisis Early Warning System.

Consumes engineered features from the versioned feature store
(``data/features/``) via the ML data loader, trains configurable PyTorch
networks (MLP, deep FC, residual, wide & deep), and produces evaluation +
explainability reports (``reports/deep_learning/``), figures, and a model
registry (``models/deep_learning/``).

Public API re-exported here so callers (runner, tests) import one place.
"""
from .base import (DLError, EpochRecord, TrainedNetwork, TrainingHistory,
                   count_parameters, resolve_device, seed_all)
from .data_loader import DLData, DLDataLoader, TabularDataset
from .evaluation import (METRIC_NAMES, ModelEvaluator, ThresholdOptimizer,
                         classification_text_report, predict_proba)
from .models import (ACTIVATIONS, NETWORK_REGISTRY, BaseNetwork,
                     build_network)
from .pipeline import DLPipeline, DLResult
from .prediction import DLPredictor
from .registry import DLModelRegistry
from .report import DLReport
from .trainer import (EarlyStopping, FocalLoss, Trainer, build_loss,
                      build_optimizer, build_scheduler, load_checkpoint)
from .visualization import DLVisualizer

__all__ = [
    "DLError", "EpochRecord", "TrainedNetwork", "TrainingHistory",
    "count_parameters", "resolve_device", "seed_all",
    "DLData", "DLDataLoader", "TabularDataset",
    "METRIC_NAMES", "ModelEvaluator", "ThresholdOptimizer",
    "classification_text_report", "predict_proba",
    "ACTIVATIONS", "NETWORK_REGISTRY", "BaseNetwork", "build_network",
    "DLPipeline", "DLResult", "DLPredictor", "DLModelRegistry", "DLReport",
    "EarlyStopping", "FocalLoss", "Trainer", "build_loss",
    "build_optimizer", "build_scheduler", "load_checkpoint",
    "DLVisualizer",
]
