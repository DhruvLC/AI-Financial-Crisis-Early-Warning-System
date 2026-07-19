"""Self-Supervised Learning module for the Financial Crisis Early
Warning System.

Stage 11. Consumes engineered features from the versioned feature store
(``data/features/``) via the shared deep-learning data loader, pretrains
configurable encoders (MLP, Residual, Transformer — the latter built from
the Transformer module's blocks) with SimCLR-style contrastive learning
(NT-Xent / Barlow Twins / VICReg) over configurable tabular
augmentations, extracts + exports latent representations for downstream
reuse, evaluates them with frozen-encoder linear-probe / KNN evaluation,
and produces figures + reports (``reports/self_supervised/``) and a
model registry (``models/self_supervised/``).

Deliberately mirrors :mod:`pipeline.transformers` and reuses the
deep-learning infrastructure — base datatypes, data loading, optimizer /
scheduler / early-stopping / checkpoint machinery, metric suite,
visualizer, and report writers — adding only what is SSL-specific:
augmentations, encoders + projection head, contrastive losses, the
two-view pretraining loop, representation extraction, probe evaluation,
and the SSL pipeline/registry/report wiring.

Public API re-exported here so callers (runner, tests) import one place.
"""
from .augmentations import (AUGMENTATION_REGISTRY, AugmentationPipeline,
                            ColumnShuffle, FeatureDropout, FeatureMasking,
                            GaussianNoise, Mixup, RandomCorruption,
                            build_augmentations)
from .base import (SSLError, TrainedEncoder, count_parameters,
                   embedding_statistics, resolve_device, seed_all)
from .data_loader import (ContrastiveDataset, SSLData, SSLDataLoader,
                          build_contrastive_loader)
from .encoder import (ENCODER_REGISTRY, BaseEncoder, MLPEncoder,
                      ResidualEncoder, TransformerSSLEncoder,
                      build_encoder)
from .evaluation import (METRIC_NAMES, KNNProbe, LinearProbe,
                         ModelEvaluator, ThresholdOptimizer)
from .losses import (BarlowTwinsLoss, NTXentLoss, VICRegLoss,
                     build_ssl_loss)
from .pipeline import SSLPipeline, SSLResult
from .projection_head import ProjectionHead, build_projection_head
from .registry import SSLModelRegistry
from .report import SSLReport
from .representation import RepresentationExporter, extract_embeddings
from .trainer import SSLTrainer, load_checkpoint
from .visualization import SSLVisualizer

__all__ = [
    "AUGMENTATION_REGISTRY", "AugmentationPipeline", "ColumnShuffle",
    "FeatureDropout", "FeatureMasking", "GaussianNoise", "Mixup",
    "RandomCorruption", "build_augmentations",
    "SSLError", "TrainedEncoder", "count_parameters",
    "embedding_statistics", "resolve_device", "seed_all",
    "ContrastiveDataset", "SSLData", "SSLDataLoader",
    "build_contrastive_loader",
    "ENCODER_REGISTRY", "BaseEncoder", "MLPEncoder", "ResidualEncoder",
    "TransformerSSLEncoder", "build_encoder",
    "METRIC_NAMES", "KNNProbe", "LinearProbe", "ModelEvaluator",
    "ThresholdOptimizer",
    "BarlowTwinsLoss", "NTXentLoss", "VICRegLoss", "build_ssl_loss",
    "SSLPipeline", "SSLResult",
    "ProjectionHead", "build_projection_head",
    "SSLModelRegistry", "SSLReport",
    "RepresentationExporter", "extract_embeddings",
    "SSLTrainer", "load_checkpoint",
    "SSLVisualizer",
]
