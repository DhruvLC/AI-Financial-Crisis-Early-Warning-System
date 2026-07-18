"""Inference for the Transformer Models module.

:class:`TransformerPredictor` reuses the deep-learning predictor's loading
+ schema-enforcement flow (``training_config.json`` + ``best_model.pt``,
missing-feature checks, thresholded prediction, risk scores) but
reconstructs networks through the **transformer** factory so registered
FT-Transformer / TabTransformer / encoder checkpoints round-trip.
"""
from __future__ import annotations

from ingestion.logging_config import get_logger
from pipeline.deep_learning.prediction import DLPredictor

from . import models as transformer_models
from .base import TransformerError

__all__ = ["TransformerPredictor"]

log = get_logger("transformers.prediction")


class TransformerPredictor(DLPredictor):
    """Load a registered transformer and predict on new data."""

    def __init__(self, registry_dir: str = "models/transformers",
                 device: str = "auto") -> None:
        super().__init__(registry_dir, device)

    def load(self, checkpoint: str | None = None) -> "TransformerPredictor":
        """Load the best (default) or an explicit checkpoint; return self.

        Temporarily swaps the deep-learning ``build_network`` factory for
        :func:`build_transformer` so the parent's loading flow (config +
        metadata validation, state-dict restore, threshold) is reused
        verbatim while constructing transformer architectures.
        """
        import pipeline.deep_learning.prediction as dl_prediction

        original = dl_prediction.build_network
        dl_prediction.build_network = transformer_models.build_transformer
        try:
            super().load(checkpoint)
        except TransformerError:
            raise
        finally:
            dl_prediction.build_network = original
        return self
