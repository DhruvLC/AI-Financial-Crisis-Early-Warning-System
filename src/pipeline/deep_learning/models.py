"""Neural-network zoo for the Deep Learning module.

One configurable ``nn.Module`` per architecture, each built from config —
hidden layers, activation, dropout, batch normalization, and weight
initialization are all parameters. Every network outputs raw **logits** of
shape ``(batch,)`` (losses use ``BCEWithLogitsLoss``-style inputs; sigmoid
is applied at prediction time).

``NETWORK_REGISTRY`` maps config keys → classes; :func:`build_network` is
the single factory used by the pipeline, trainer, and tests — mirroring
:func:`pipeline.ml.models.build_model`.
"""
from __future__ import annotations

import torch
from torch import nn

from .base import DLError

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "leakyrelu": nn.LeakyReLU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
}

INITIALIZERS = ("kaiming", "xavier", "normal", "default")


def make_activation(name: str) -> nn.Module:
    cls = ACTIVATIONS.get(str(name).lower())
    if cls is None:
        raise DLError(f"unsupported activation '{name}' "
                      f"(supported: {sorted(set(ACTIVATIONS))})")
    return cls()


def init_weights(module: nn.Module, scheme: str = "kaiming") -> None:
    """Apply the configured initialization to every Linear layer."""
    scheme = str(scheme).lower()
    if scheme not in INITIALIZERS:
        raise DLError(f"unsupported initialization '{scheme}' "
                      f"(supported: {INITIALIZERS})")
    if scheme == "default":
        return
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if scheme == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif scheme == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif scheme == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _dense_block(in_dim: int, out_dim: int, activation: str,
                 dropout: float, batch_norm: bool) -> list[nn.Module]:
    layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_dim))
    layers.append(make_activation(activation))
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers


class BaseNetwork(nn.Module):
    """Template for one binary-classification network.

    Subclasses set :attr:`name` / :attr:`display_name`, provide
    :meth:`default_params`, and build their layers in ``__init__`` via
    the merged default + user params in ``self.params``.
    """

    name: str = "base"
    display_name: str = "Base Network"

    def __init__(self, n_features: int, params: dict | None = None) -> None:
        super().__init__()
        if n_features < 1:
            raise DLError(f"{self.name}: invalid n_features {n_features}")
        self.n_features = n_features
        self.params = {**self.default_params(), **(params or {})}

    def default_params(self) -> dict:
        return {}

    def architecture(self) -> dict:
        """Config summary recorded in reports and the registry."""
        return {"network": self.name, "n_features": self.n_features,
                **self.params}


class MLPNetwork(BaseNetwork):
    """Compact multi-layer perceptron."""

    name = "mlp"
    display_name = "Multi-Layer Perceptron"

    def default_params(self) -> dict:
        return {"hidden_layers": [64, 32], "activation": "relu",
                "dropout": 0.2, "batch_norm": True,
                "initialization": "kaiming"}

    def __init__(self, n_features: int, params: dict | None = None) -> None:
        super().__init__(n_features, params)
        p = self.params
        layers: list[nn.Module] = []
        dim = n_features
        for width in p["hidden_layers"]:
            layers += _dense_block(dim, width, p["activation"],
                                   p["dropout"], p["batch_norm"])
            dim = width
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)
        init_weights(self, p["initialization"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DeepFCNetwork(MLPNetwork):
    """Deeper fully-connected network (same block structure, more depth)."""

    name = "deep_fc"
    display_name = "Deep Fully-Connected Network"

    def default_params(self) -> dict:
        return {"hidden_layers": [256, 128, 64, 32], "activation": "gelu",
                "dropout": 0.3, "batch_norm": True,
                "initialization": "kaiming"}


class ResidualBlock(nn.Module):
    """Two dense layers with a skip connection at constant width."""

    def __init__(self, width: int, activation: str, dropout: float,
                 batch_norm: bool) -> None:
        super().__init__()
        self.body = nn.Sequential(
            *_dense_block(width, width, activation, dropout, batch_norm),
            nn.Linear(width, width),
            nn.BatchNorm1d(width) if batch_norm else nn.Identity())
        self.act = make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + x)


class ResidualNetwork(BaseNetwork):
    """Residual feed-forward network: input projection + N skip blocks."""

    name = "residual"
    display_name = "Residual Feed-Forward Network"

    def default_params(self) -> dict:
        return {"width": 128, "n_blocks": 3, "activation": "relu",
                "dropout": 0.2, "batch_norm": True,
                "initialization": "kaiming"}

    def __init__(self, n_features: int, params: dict | None = None) -> None:
        super().__init__(n_features, params)
        p = self.params
        self.project = nn.Sequential(
            *_dense_block(n_features, p["width"], p["activation"],
                          p["dropout"], p["batch_norm"]))
        self.blocks = nn.Sequential(
            *[ResidualBlock(p["width"], p["activation"], p["dropout"],
                            p["batch_norm"]) for _ in range(p["n_blocks"])])
        self.head = nn.Linear(p["width"], 1)
        init_weights(self, p["initialization"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.project(x))).squeeze(-1)


class WideDeepNetwork(BaseNetwork):
    """Wide & Deep: a linear (wide) path summed with a deep MLP path."""

    name = "wide_deep"
    display_name = "Wide & Deep Network"

    def default_params(self) -> dict:
        return {"hidden_layers": [128, 64], "activation": "relu",
                "dropout": 0.2, "batch_norm": True,
                "initialization": "kaiming"}

    def __init__(self, n_features: int, params: dict | None = None) -> None:
        super().__init__(n_features, params)
        p = self.params
        self.wide = nn.Linear(n_features, 1)
        layers: list[nn.Module] = []
        dim = n_features
        for width in p["hidden_layers"]:
            layers += _dense_block(dim, width, p["activation"],
                                   p["dropout"], p["batch_norm"])
            dim = width
        layers.append(nn.Linear(dim, 1))
        self.deep = nn.Sequential(*layers)
        init_weights(self, p["initialization"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.wide(x) + self.deep(x)).squeeze(-1)


NETWORK_REGISTRY: dict[str, type[BaseNetwork]] = {
    cls.name: cls
    for cls in (MLPNetwork, DeepFCNetwork, ResidualNetwork, WideDeepNetwork)
}


def build_network(name: str, n_features: int,
                  params: dict | None = None) -> BaseNetwork:
    """Factory: config key → constructed network (raises :class:`DLError`)."""
    cls = NETWORK_REGISTRY.get(str(name).lower())
    if cls is None:
        raise DLError(f"unsupported network '{name}' "
                      f"(supported: {sorted(NETWORK_REGISTRY)})")
    return cls(n_features, params)
