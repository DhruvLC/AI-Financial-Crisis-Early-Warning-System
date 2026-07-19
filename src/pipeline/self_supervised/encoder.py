"""Encoder architectures for the Self-Supervised Learning module.

Config-driven encoders mapping ``(batch, n_features)`` →
``(batch, embedding_dim)`` representations (no classification head — that
is what SSL removes):

* :class:`MLPEncoder`         — dense stack (Linear→BN→activation→dropout)
* :class:`ResidualEncoder`    — dense stem + pre-activation residual
  blocks (mirrors :class:`pipeline.deep_learning.models.ResidualNetwork`)
* :class:`TransformerSSLEncoder` — **reuses the Transformer module's
  building blocks** (:class:`FeatureTokenizer`, :class:`EncoderBlock`)
  with mean pooling into the embedding space.

``ENCODER_REGISTRY`` maps config keys → classes; :func:`build_encoder`
is the single factory used by pipeline, representation extractor, and
tests — mirroring :func:`pipeline.transformers.models.build_transformer`.
"""
from __future__ import annotations

import torch
from torch import nn

from pipeline.deep_learning.models import make_activation
from pipeline.transformers.models import EncoderBlock, FeatureTokenizer

from .base import SSLError

__all__ = ["ENCODER_REGISTRY", "BaseEncoder", "MLPEncoder",
           "ResidualEncoder", "TransformerSSLEncoder", "build_encoder"]


class BaseEncoder(nn.Module):
    """Template shared by every SSL encoder in the registry.

    Subclasses set :attr:`name`, provide :meth:`default_params`, build
    layers from the merged default + user params in ``self.params``, and
    must expose :attr:`embedding_dim`.
    """

    name: str = "base"
    display_name: str = "Base Encoder"

    def __init__(self, n_features: int, params: dict | None = None) -> None:
        super().__init__()
        if n_features < 1:
            raise SSLError(f"{self.name}: invalid n_features {n_features}")
        self.n_features = n_features
        self.params = {**self.default_params(), **(params or {})}
        d = float(self.params.get("dropout", 0.0))
        if not 0.0 <= d < 1.0:
            raise SSLError(f"{self.name}: invalid dropout {d}")
        self.embedding_dim = int(self.params.get("embedding_dim", 64))
        if self.embedding_dim < 1:
            raise SSLError(f"{self.name}: invalid embedding_dim "
                           f"{self.embedding_dim}")

    def default_params(self) -> dict:
        return {}

    def architecture(self) -> dict:
        """Config summary recorded in reports and the registry."""
        return {"encoder": self.name, "n_features": self.n_features,
                **self.params}


class MLPEncoder(BaseEncoder):
    """Dense encoder: Linear → BatchNorm → activation → dropout blocks."""

    name = "mlp"
    display_name = "MLP Encoder"

    def default_params(self) -> dict:
        return {"hidden_dims": [256, 128], "embedding_dim": 64,
                "activation": "relu", "dropout": 0.1, "batch_norm": True}

    def __init__(self, n_features: int, params: dict | None = None) -> None:
        super().__init__(n_features, params)
        p = self.params
        dims = [n_features] + [int(h) for h in p["hidden_dims"]]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if p.get("batch_norm", True):
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(make_activation(str(p["activation"])))
            layers.append(nn.Dropout(float(p["dropout"])))
        layers.append(nn.Linear(dims[-1], self.embedding_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResBlock(nn.Module):
    """Pre-activation residual block over a constant width."""

    def __init__(self, dim: int, activation: str, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim), make_activation(activation),
            nn.Linear(dim, dim), nn.Dropout(dropout),
            nn.BatchNorm1d(dim), make_activation(activation),
            nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualEncoder(BaseEncoder):
    """Dense stem + residual blocks + linear embedding projection."""

    name = "residual"
    display_name = "Residual Encoder"

    def default_params(self) -> dict:
        return {"width": 256, "n_blocks": 3, "embedding_dim": 64,
                "activation": "relu", "dropout": 0.1}

    def __init__(self, n_features: int, params: dict | None = None) -> None:
        super().__init__(n_features, params)
        p = self.params
        width = int(p["width"])
        if width < 1 or int(p["n_blocks"]) < 1:
            raise SSLError(f"{self.name}: invalid width/n_blocks")
        self.stem = nn.Linear(n_features, width)
        self.blocks = nn.Sequential(*[
            _ResBlock(width, str(p["activation"]), float(p["dropout"]))
            for _ in range(int(p["n_blocks"]))])
        self.out = nn.Sequential(nn.BatchNorm1d(width),
                                 make_activation(str(p["activation"])),
                                 nn.Linear(width, self.embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.blocks(self.stem(x)))


class TransformerSSLEncoder(BaseEncoder):
    """Transformer encoder over feature tokens, mean-pooled to an
    embedding — built from the Transformer module's
    :class:`FeatureTokenizer` and :class:`EncoderBlock` (direct reuse)."""

    name = "transformer"
    display_name = "Transformer Encoder"

    def default_params(self) -> dict:
        return {"embed_dim": 32, "n_heads": 4, "n_layers": 2,
                "ff_dim": 64, "dropout": 0.1, "embedding_dim": 64,
                "positional_embedding": True}

    def __init__(self, n_features: int, params: dict | None = None) -> None:
        super().__init__(n_features, params)
        p = self.params
        d = int(p["embed_dim"])
        for key in ("embed_dim", "n_heads", "n_layers", "ff_dim"):
            if int(p.get(key, 1)) < 1:
                raise SSLError(f"{self.name}: invalid {key} {p.get(key)}")
        self.tokenizer = FeatureTokenizer(n_features, d)
        self.positional = None
        if p.get("positional_embedding", True):
            self.positional = nn.Parameter(torch.zeros(1, n_features, d))
            nn.init.normal_(self.positional, std=0.02)
        self.blocks = nn.ModuleList(
            EncoderBlock(d, int(p["n_heads"]), int(p["ff_dim"]),
                         float(p["dropout"]))
            for _ in range(int(p["n_layers"])))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        if self.positional is not None:
            tokens = tokens + self.positional
        for block in self.blocks:
            tokens = block(tokens)
        return self.out(self.norm(tokens.mean(dim=1)))


ENCODER_REGISTRY: dict[str, type[BaseEncoder]] = {
    cls.name: cls
    for cls in (MLPEncoder, ResidualEncoder, TransformerSSLEncoder)
}


def build_encoder(name: str, n_features: int,
                  params: dict | None = None) -> BaseEncoder:
    """Factory: config key → constructed encoder (raises
    :class:`SSLError`)."""
    cls = ENCODER_REGISTRY.get(str(name).lower())
    if cls is None:
        raise SSLError(f"unsupported encoder '{name}' "
                       f"(supported: {sorted(ENCODER_REGISTRY)})")
    return cls(n_features, params)
