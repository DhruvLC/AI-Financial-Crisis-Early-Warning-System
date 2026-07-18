"""Core datatypes and helpers for the Transformer Models module.

Reuses the deep-learning plumbing wholesale — :class:`TrainedNetwork`,
:class:`TrainingHistory`, device/seed utilities — and adds only what is
transformer-specific: the error type (a :class:`DLError` subclass so all
shared components keep raising/catching one family), a trained-transformer
container that also carries attention artefacts, and small attention
summary helpers used by explainability and reporting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ingestion.logging_config import get_logger
from pipeline.deep_learning.base import (DLError, EpochRecord,  # noqa: F401
                                         TrainedNetwork, TrainingHistory,
                                         count_parameters, resolve_device,
                                         seed_all)

log = get_logger("transformers.base")

__all__ = ["TransformerError", "TrainedTransformer", "EpochRecord",
           "TrainingHistory", "count_parameters", "resolve_device",
           "seed_all", "attention_entropy", "mean_attention_by_feature"]


class TransformerError(DLError):
    """Raised for fatal transformer problems (invalid config, empty data,
    NaN losses, corrupt checkpoints) when configured to fail fast.

    Subclasses :class:`DLError` so the reused deep-learning components
    (data loader, trainer, evaluation) interoperate without translation.
    """


@dataclass
class TrainedTransformer(TrainedNetwork):
    """One transformer's full training + evaluation + attention record."""

    attention: "AttentionSummary | None" = None

    def as_dict(self) -> dict:
        d = super().as_dict()
        d["attention"] = self.attention.as_dict() if self.attention else None
        return d


@dataclass
class AttentionSummary:
    """Aggregated attention diagnostics for one trained transformer.

    * ``feature_attention`` — mean attention each feature token receives,
      averaged over samples, heads, layers, and query positions.
    * ``matrix`` — mean token×token attention of the last encoder layer
      (rows = queries, cols = keys), used for the heatmap.
    * ``entropy`` — mean attention entropy per layer (higher = more
      diffuse attention).
    """

    feature_names: list[str] = field(default_factory=list)
    feature_attention: dict[str, float] = field(default_factory=dict)
    matrix: np.ndarray | None = None
    matrix_labels: list[str] = field(default_factory=list)
    entropy: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0

    def as_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "feature_attention": {k: float(v) for k, v in
                                  self.feature_attention.items()},
            "entropy_per_layer": {k: float(v) for k, v in
                                  self.entropy.items()},
        }


# ── attention math helpers ──────────────────────────────────────────────────
def attention_entropy(attn: np.ndarray, eps: float = 1e-12) -> float:
    """Mean Shannon entropy of attention rows.

    ``attn`` is any array whose last axis is a probability distribution
    over keys (e.g. ``(batch, heads, query, key)``).
    """
    a = np.asarray(attn, dtype=np.float64)
    ent = -(a * np.log(a + eps)).sum(axis=-1)
    return float(ent.mean())


def mean_attention_by_feature(attn: np.ndarray, labels: list[str],
                              skip_cls: bool = True) -> dict[str, float]:
    """Mean attention each key token receives, averaged over everything else.

    ``attn`` has shape ``(..., query, key)``; ``labels`` names the key
    tokens (index-aligned). With ``skip_cls`` the leading ``[CLS]`` token
    is dropped from the returned map (it is not a feature).
    """
    a = np.asarray(attn, dtype=np.float64)
    received = a.mean(axis=tuple(range(a.ndim - 1)))     # mean over all but key
    pairs = list(zip(labels, received))
    if skip_cls and pairs and pairs[0][0] == "[CLS]":
        pairs = pairs[1:]
    order = sorted(pairs, key=lambda kv: -kv[1])
    return {name: float(val) for name, val in order}
