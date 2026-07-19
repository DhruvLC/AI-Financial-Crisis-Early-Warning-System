"""Model registry for the Self-Supervised Learning module.

Extends :class:`pipeline.deep_learning.registry.DLModelRegistry` pointed
at ``models/self_supervised/`` with the SSL naming and artefacts:

* ``<name>_best.pt`` / ``<name>_last.pt``   — per-encoder checkpoints
  (encoder + projection head + optimizer state, resumable)
* ``best_encoder.pt`` / ``last_encoder.pt`` — the winning encoder
* ``training_config.json``                  — architectures + hyperparams
* ``metrics.json``                          — probe metrics per split
* ``history.json``                          — per-epoch pretraining traces
* ``feature_metadata.json``                 — feature schema + store version
* ``representations/representation_metadata.json`` — embedding exports
* ``registry.json``                         — versioned entry index
"""
from __future__ import annotations

import os
import shutil

from ingestion.logging_config import get_logger
from pipeline.deep_learning.registry import DLModelRegistry, _jsonable

from .base import TrainedEncoder

__all__ = ["SSLModelRegistry"]

log = get_logger("ssl.registry")


class SSLModelRegistry(DLModelRegistry):
    """Versioned on-disk registry of pretrained encoders."""

    def __init__(self, models_dir: str = "models/self_supervised") -> None:
        super().__init__(models_dir)

    def register(self, trained: TrainedEncoder, features: list[str],
                 dataset_version: str, target_col: str) -> dict:
        """Record one encoder's metadata; return the entry."""
        entry = super().register(trained, features, dataset_version,
                                 target_col)
        # SSL additions the shared entry doesn't know about.
        entry["representations"] = _jsonable(trained.representations)
        entry["embedding_stats"] = _jsonable(trained.embedding_stats)
        entry["knn_metrics"] = {
            split: ev.metrics
            for split, ev in trained.knn_evaluations.items()}
        registry = self._read()
        registry["models"][-1] = entry
        self._write(registry)
        return entry

    def register_best(self, trained: TrainedEncoder, entry: dict,
                      features: list[str], dataset_version: str,
                      target_col: str, config: dict) -> dict[str, str]:
        """Save the winning encoder's artefact suite (``best_encoder.pt``
        / ``last_encoder.pt`` naming); return the paths."""
        paths = super().register_best(trained, entry, features,
                                      dataset_version, target_col, config)
        # rename the generic best/last model artefacts to encoder names
        for old, new in (("best_model.pt", "best_encoder.pt"),
                         ("last_model.pt", "last_encoder.pt")):
            src = paths.pop(old, None)
            if src and os.path.exists(src):
                dst = os.path.join(self.models_dir, new)
                shutil.move(src, dst)
                paths[new] = dst
        registry = self._read()
        best = registry.get("best_model") or {}
        best["artefact"] = paths.get("best_encoder.pt")
        registry["best_model"] = best
        self._write(registry)
        log.info("best encoder: %s (%s) -> %s", trained.name,
                 entry["model_version"], paths.get("best_encoder.pt"))
        return paths
