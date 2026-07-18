"""Transformer Models module for the Financial Crisis Early Warning System.

Stage 10. Consumes engineered features from the versioned feature store
(``data/features/``) via the shared deep-learning data loader, trains
configurable attention-based tabular networks (FT-Transformer,
TabTransformer, plain Transformer Encoder), and produces evaluation +
attention-explainability reports (``reports/transformers/``), figures, and
a model registry (``models/transformers/``).

Deliberately mirrors :mod:`pipeline.deep_learning` and reuses its
infrastructure — base datatypes, data loading, training engine, metric
suite, and visualizer — adding only what is transformer-specific:
architectures with attention capture, attention analysis/plots, and the
transformer pipeline/registry/report/prediction wiring.

Public API re-exported here so callers (runner, tests) import one place.
"""
from .base import (TransformerError, TrainedTransformer, attention_entropy,
                   mean_attention_by_feature)
from .data_loader import TransformerData, TransformerDataLoader
from .evaluation import (METRIC_NAMES, ModelEvaluator, ThresholdOptimizer,
                         classification_text_report, predict_proba)
from .models import (TRANSFORMER_REGISTRY, BaseTransformer, FTTransformer,
                     TabTransformer, TabularEncoderTransformer,
                     build_transformer)
from .pipeline import TransformerPipeline, TransformerResult
from .prediction import TransformerPredictor
from .registry import TransformerModelRegistry
from .report import TransformerReport
from .trainer import TransformerTrainer
from .visualization import TransformerVisualizer

__all__ = [
    "TransformerError", "TrainedTransformer", "attention_entropy",
    "mean_attention_by_feature",
    "TransformerData", "TransformerDataLoader",
    "METRIC_NAMES", "ModelEvaluator", "ThresholdOptimizer",
    "classification_text_report", "predict_proba",
    "TRANSFORMER_REGISTRY", "BaseTransformer", "FTTransformer",
    "TabTransformer", "TabularEncoderTransformer", "build_transformer",
    "TransformerPipeline", "TransformerResult", "TransformerPredictor",
    "TransformerModelRegistry", "TransformerReport", "TransformerTrainer",
    "TransformerVisualizer",
]
