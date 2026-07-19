"""Core datatypes and helpers for the Self-Supervised Learning module.

Reuses the deep-learning plumbing wholesale — :class:`TrainingHistory`,
:class:`EpochRecord`, device/seed utilities — and adds only what is
SSL-specific: the error type (a :class:`DLError` subclass so every shared
component keeps raising/catching one family) and a trained-encoder
container that also carries representation + probe-evaluation artefacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ingestion.logging_config import get_logger
from pipeline.deep_learning.base import (DLError, EpochRecord,  # noqa: F401
                                         TrainingHistory, count_parameters,
                                         resolve_device, seed_all)

log = get_logger("ssl.base")

__all__ = ["SSLError", "TrainedEncoder", "EpochRecord", "TrainingHistory",
           "count_parameters", "resolve_device", "seed_all",
           "embedding_statistics"]


class SSLError(DLError):
    """Raised for fatal self-supervised problems (invalid config, empty
    data, NaN embeddings/losses, corrupt checkpoints, invalid
    augmentations) when configured to fail fast.

    Subclasses :class:`DLError` so the reused deep-learning components
    (data loader, optimizer/scheduler factories, checkpoint loader)
    interoperate without translation.
    """


@dataclass
class TrainedEncoder:
    """One encoder's full pretraining + evaluation + representation record."""

    name: str
    model: Any                                    # encoder nn.Module
    projection_head: Any = None                   # head nn.Module (train-only)
    architecture: dict = field(default_factory=dict)
    hyperparameters: dict = field(default_factory=dict)
    history: TrainingHistory | None = None
    evaluations: dict[str, Any] = field(default_factory=dict)  # probe evals
    knn_evaluations: dict[str, Any] = field(default_factory=dict)
    threshold: float = 0.5
    threshold_method: str = "default"
    train_seconds: float = 0.0
    n_parameters: int = 0
    device: str = "cpu"
    checkpoints: dict[str, str] = field(default_factory=dict)
    representations: dict[str, str] = field(default_factory=dict)  # split→path
    embedding_stats: dict[str, dict] = field(default_factory=dict)
    permutation_importance: Any = None    # unused; DLReport compatibility
    shap_summary: dict | None = None      # unused; DLReport compatibility
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    def metric(self, name: str, split: str = "test") -> float:
        ev = self.evaluations.get(split)
        return (float(ev.metrics.get(name, float("nan")))
                if ev else float("nan"))

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "status": "failed" if self.failed else "trained",
            "error": self.error,
            "architecture": self.architecture,
            "hyperparameters": self.hyperparameters,
            "n_parameters": self.n_parameters,
            "device": self.device,
            "threshold": self.threshold,
            "threshold_method": self.threshold_method,
            "train_seconds": round(self.train_seconds, 3),
            "history": self.history.as_dict() if self.history else None,
            "checkpoints": self.checkpoints,
            "representations": self.representations,
            "embedding_stats": self.embedding_stats,
            "evaluations": {s: e.as_dict()
                            for s, e in self.evaluations.items()},
            "knn_evaluations": {s: e.as_dict()
                                for s, e in self.knn_evaluations.items()},
        }


def embedding_statistics(Z: np.ndarray) -> dict:
    """Summary statistics of one embedding matrix (logged + reported).

    Raises :class:`SSLError` on non-finite embeddings — a NaN latent space
    means pretraining diverged and every downstream artefact would be junk.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.size == 0:
        raise SSLError("cannot summarise an empty embedding matrix")
    if not np.isfinite(Z).all():
        raise SSLError("embeddings contain non-finite values (NaN/Inf) — "
                       "pretraining diverged; lower the learning rate")
    norms = np.linalg.norm(Z, axis=1)
    return {"n_samples": int(Z.shape[0]),
            "dim": int(Z.shape[1]),
            "mean": float(Z.mean()),
            "std": float(Z.std()),
            "min": float(Z.min()),
            "max": float(Z.max()),
            "mean_l2_norm": float(norms.mean()),
            "std_l2_norm": float(norms.std())}
