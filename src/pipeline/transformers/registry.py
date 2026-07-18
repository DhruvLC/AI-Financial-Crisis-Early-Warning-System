"""Model registry for the Transformer Models module.

Pure reuse of :class:`pipeline.deep_learning.registry.DLModelRegistry`
pointed at ``models/transformers/`` — the artefact suite is identical:

* ``<name>_best.pt`` / ``<name>_last.pt`` — per-transformer checkpoints
* ``best_model.pt`` / ``last_model.pt``   — the winning transformer
* ``training_config.json``                — architectures + hyperparameters
* ``metrics.json``                        — per-model per-split metrics
* ``history.json``                        — per-epoch training traces
* ``feature_metadata.json``               — feature schema + store version
* ``registry.json``                       — versioned entry index
"""
from __future__ import annotations

from pipeline.deep_learning.registry import DLModelRegistry

__all__ = ["TransformerModelRegistry"]


class TransformerModelRegistry(DLModelRegistry):
    """Versioned on-disk registry of trained transformers."""

    def __init__(self, models_dir: str = "models/transformers") -> None:
        super().__init__(models_dir)
