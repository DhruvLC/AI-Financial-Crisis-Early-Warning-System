"""Projection head for the Self-Supervised Learning module.

SimCLR-style MLP head mapping encoder embeddings into the space where the
contrastive loss is computed. Used **only during pretraining** — the
downstream representations come from the encoder output, which is known
to transfer better than the projected space (Chen et al., 2020).
"""
from __future__ import annotations

import torch
from torch import nn

from pipeline.deep_learning.models import make_activation

from .base import SSLError

__all__ = ["ProjectionHead", "build_projection_head"]


class ProjectionHead(nn.Module):
    """Configurable MLP: embedding_dim → hidden(s) → projection_dim."""

    def __init__(self, embedding_dim: int, hidden_dims: list[int],
                 projection_dim: int, activation: str = "relu",
                 batch_norm: bool = True) -> None:
        super().__init__()
        if embedding_dim < 1 or projection_dim < 1:
            raise SSLError(f"invalid projection head dims "
                           f"({embedding_dim} -> {projection_dim})")
        dims = [int(embedding_dim)] + [int(h) for h in hidden_dims]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(make_activation(activation))
        layers.append(nn.Linear(dims[-1], int(projection_dim)))
        self.net = nn.Sequential(*layers)
        self.projection_dim = int(projection_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def build_projection_head(embedding_dim: int,
                          cfg: dict | None = None) -> ProjectionHead:
    """Config → projection head (raises :class:`SSLError`)."""
    cfg = cfg or {}
    return ProjectionHead(
        embedding_dim=embedding_dim,
        hidden_dims=list(cfg.get("hidden_dims", [128])),
        projection_dim=int(cfg.get("projection_dim", 64)),
        activation=str(cfg.get("activation", "relu")),
        batch_norm=bool(cfg.get("batch_norm", True)))
