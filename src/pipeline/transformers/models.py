"""Attention-based architectures for the Transformer Models module.

Three production tabular transformers, each config-driven (embedding
dimension, heads, encoder depth, feed-forward dimension, dropout, optional
positional embeddings) and each outputting raw **logits** of shape
``(batch,)`` — exactly the contract the shared deep-learning trainer,
evaluator, and predictor expect:

* :class:`FTTransformer` — Feature Tokenizer + Transformer (Gorishniy et
  al., 2021): each numeric feature becomes a learned token
  (``x_j * W_j + b_j``), a ``[CLS]`` token is prepended, the sequence runs
  through pre-norm encoder blocks, and the ``[CLS]`` representation is
  classified.
* :class:`TabTransformer` — TabTransformer (Huang et al., 2020) adapted to
  the all-continuous engineered features: tokenized features are
  contextualized by the encoder, then the flattened contextual embeddings
  are concatenated with layer-normed raw features into an MLP head (the
  paper's continuous pathway).
* :class:`TabularEncoderTransformer` — a plain ``TransformerEncoder`` over
  feature tokens with learned positional embeddings and mean pooling; the
  no-frills baseline of the family.

Every encoder block uses multi-head self-attention + LayerNorm + residual
connections, and can **capture attention weights** (per layer, averaged
over heads) for the explainability stage via ``collect_attention``.

``TRANSFORMER_REGISTRY`` maps config keys → classes; :func:`build_
transformer` is the single factory used by pipeline, predictor, and tests
— mirroring :func:`pipeline.deep_learning.models.build_network`.
"""
from __future__ import annotations

import torch
from torch import nn

from .base import TransformerError

__all__ = ["TRANSFORMER_REGISTRY", "BaseTransformer", "FTTransformer",
           "TabTransformer", "TabularEncoderTransformer",
           "build_transformer"]


# ── building blocks ─────────────────────────────────────────────────────────
class FeatureTokenizer(nn.Module):
    """Per-feature affine embedding of continuous inputs.

    ``x`` of shape ``(batch, n_features)`` → tokens of shape
    ``(batch, n_features, d_model)`` via a learned weight + bias pair per
    feature (Gorishniy et al., 2021, eq. 1).
    """

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight + self.bias


class EncoderBlock(nn.Module):
    """Pre-norm transformer encoder block with attention capture.

    LayerNorm → multi-head self-attention → residual, then LayerNorm →
    feed-forward → residual. When ``collect`` is set the head-averaged
    attention matrix of the batch is stored in :attr:`last_attention`.
    """

    def __init__(self, d_model: int, n_heads: int, ff_dim: int,
                 dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise TransformerError(
                f"embed_dim {d_model} must be divisible by n_heads "
                f"{n_heads}")
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(ff_dim, d_model))
        self.dropout = nn.Dropout(dropout)
        self.collect = False
        self.last_attention: torch.Tensor | None = None   # (batch, T, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, weights = self.attn(h, h, h,
                                      need_weights=self.collect,
                                      average_attn_weights=True)
        if self.collect and weights is not None:
            self.last_attention = weights.detach()
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class BaseTransformer(nn.Module):
    """Template shared by every tabular transformer in the registry.

    Subclasses set :attr:`name`/:attr:`display_name`, provide
    :meth:`default_params`, build their layers in ``__init__`` from the
    merged default + user params in ``self.params``, and expose their
    encoder stack as ``self.blocks`` (a list of :class:`EncoderBlock`) so
    attention capture works uniformly.
    """

    name: str = "base"
    display_name: str = "Base Transformer"

    def __init__(self, n_features: int, params: dict | None = None) -> None:
        super().__init__()
        if n_features < 1:
            raise TransformerError(
                f"{self.name}: invalid n_features {n_features}")
        self.n_features = n_features
        self.params = {**self.default_params(), **(params or {})}
        for key in ("embed_dim", "n_heads", "n_layers", "ff_dim"):
            if int(self.params.get(key, 1)) < 1:
                raise TransformerError(
                    f"{self.name}: invalid {key} {self.params.get(key)}")
        d = float(self.params.get("dropout", 0.0))
        if not 0.0 <= d < 1.0:
            raise TransformerError(f"{self.name}: invalid dropout {d}")
        self.blocks = nn.ModuleList()

    def default_params(self) -> dict:
        return {}

    def architecture(self) -> dict:
        """Config summary recorded in reports and the registry."""
        return {"network": self.name, "n_features": self.n_features,
                **self.params}

    # ── attention capture ─────────────────────────────────────────────────────
    def collect_attention(self, enabled: bool = True) -> None:
        """Toggle attention-weight capture in every encoder block."""
        for block in self.blocks:
            block.collect = enabled
            if not enabled:
                block.last_attention = None

    def attention_weights(self) -> dict[str, torch.Tensor]:
        """Head-averaged attention per layer from the last forward pass."""
        return {f"layer_{i + 1}": block.last_attention
                for i, block in enumerate(self.blocks)
                if block.last_attention is not None}

    def token_labels(self) -> list[str]:
        """Names of the key/query tokens (index-aligned with attention)."""
        return [f"f{i}" for i in range(self.n_features)]

    def _make_blocks(self) -> None:
        p = self.params
        self.blocks = nn.ModuleList(
            EncoderBlock(int(p["embed_dim"]), int(p["n_heads"]),
                         int(p["ff_dim"]), float(p["dropout"]))
            for _ in range(int(p["n_layers"])))

    def _encode(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class FTTransformer(BaseTransformer):
    """FT-Transformer: feature tokenizer + [CLS] + encoder + CLS head."""

    name = "ft_transformer"
    display_name = "FT-Transformer"

    def default_params(self) -> dict:
        return {"embed_dim": 64, "n_heads": 8, "n_layers": 3,
                "ff_dim": 128, "dropout": 0.1}

    def __init__(self, n_features: int, params: dict | None = None) -> None:
        super().__init__(n_features, params)
        p = self.params
        d = int(p["embed_dim"])
        self.tokenizer = FeatureTokenizer(n_features, d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls_token, std=0.02)
        self._make_blocks()
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(d, 1))

    def token_labels(self) -> list[str]:
        return ["[CLS]"] + super().token_labels()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(len(x), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self._encode(tokens)
        return self.head(self.norm(tokens[:, 0])).squeeze(-1)


class TabTransformer(BaseTransformer):
    """TabTransformer adapted to all-continuous features.

    Feature tokens are contextualized by the encoder; the flattened
    contextual embeddings are concatenated with layer-normed raw features
    and classified by an MLP head — the paper's treatment of continuous
    columns, with the tokenizer standing in for categorical embeddings
    since the engineered store has no categorical columns.
    """

    name = "tab_transformer"
    display_name = "TabTransformer"

    def default_params(self) -> dict:
        return {"embed_dim": 32, "n_heads": 8, "n_layers": 3,
                "ff_dim": 64, "dropout": 0.1,
                "mlp_hidden": [128, 64]}

    def __init__(self, n_features: int, params: dict | None = None) -> None:
        super().__init__(n_features, params)
        p = self.params
        d = int(p["embed_dim"])
        self.tokenizer = FeatureTokenizer(n_features, d)
        self._make_blocks()
        self.cont_norm = nn.LayerNorm(n_features)
        dims = [n_features * d + n_features] + [int(h)
                                                for h in p["mlp_hidden"]]
        mlp: list[nn.Module] = []
        for i in range(len(dims) - 1):
            mlp += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(),
                    nn.Dropout(float(p["dropout"]))]
        mlp.append(nn.Linear(dims[-1], 1))
        self.head = nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        contextual = self._encode(self.tokenizer(x)).flatten(1)
        return self.head(torch.cat([contextual, self.cont_norm(x)],
                                   dim=1)).squeeze(-1)


class TabularEncoderTransformer(BaseTransformer):
    """Plain Transformer Encoder over feature tokens with learned
    positional embeddings and mean pooling."""

    name = "tabular_encoder"
    display_name = "Tabular Transformer Encoder"

    def default_params(self) -> dict:
        return {"embed_dim": 64, "n_heads": 4, "n_layers": 2,
                "ff_dim": 128, "dropout": 0.1,
                "positional_embedding": True}

    def __init__(self, n_features: int, params: dict | None = None) -> None:
        super().__init__(n_features, params)
        p = self.params
        d = int(p["embed_dim"])
        self.tokenizer = FeatureTokenizer(n_features, d)
        self.positional = None
        if p.get("positional_embedding", True):
            self.positional = nn.Parameter(torch.zeros(1, n_features, d))
            nn.init.normal_(self.positional, std=0.02)
        self._make_blocks()
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        if self.positional is not None:
            tokens = tokens + self.positional
        tokens = self._encode(tokens)
        return self.head(self.norm(tokens.mean(dim=1))).squeeze(-1)


TRANSFORMER_REGISTRY: dict[str, type[BaseTransformer]] = {
    cls.name: cls
    for cls in (FTTransformer, TabTransformer, TabularEncoderTransformer)
}


def build_transformer(name: str, n_features: int,
                      params: dict | None = None) -> BaseTransformer:
    """Factory: config key → constructed transformer (raises
    :class:`TransformerError`)."""
    cls = TRANSFORMER_REGISTRY.get(str(name).lower())
    if cls is None:
        raise TransformerError(
            f"unsupported transformer '{name}' "
            f"(supported: {sorted(TRANSFORMER_REGISTRY)})")
    return cls(n_features, params)
