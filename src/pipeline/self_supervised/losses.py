"""Self-supervised losses.

All losses take the two projected views ``(z1, z2)`` of shape
``(batch, dim)`` and return a scalar:

* :class:`NTXentLoss` — normalized-temperature cross entropy (InfoNCE,
  SimCLR; Chen et al., 2020) with configurable temperature
* :class:`BarlowTwinsLoss` — redundancy reduction on the cross-correlation
  matrix (Zbontar et al., 2021)
* :class:`VICRegLoss` — variance / invariance / covariance regularization
  (Bardes et al., 2022)

:func:`build_ssl_loss` maps the config to a loss module and validates
parameters, mirroring :func:`pipeline.deep_learning.trainer.build_loss`.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .base import SSLError

__all__ = ["NTXentLoss", "BarlowTwinsLoss", "VICRegLoss", "build_ssl_loss"]


class NTXentLoss(nn.Module):
    """NT-Xent / InfoNCE over the 2N-view batch.

    Views are L2-normalized, all pairwise cosine similarities are scaled
    by ``1 / temperature``, self-similarities are masked, and each view's
    positive is its counterpart from the other augmentation.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        if temperature <= 0:
            raise SSLError(f"nt_xent: temperature={temperature} "
                           f"must be > 0")
        self.temperature = float(temperature)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        n = len(z1)
        if n < 2:
            raise SSLError("NT-Xent needs batch size >= 2 (no negatives)")
        z = F.normalize(torch.cat([z1, z2]), dim=1)
        sim = z @ z.T / self.temperature                     # (2N, 2N)
        mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float("-inf"))
        targets = torch.cat([torch.arange(n, 2 * n),
                             torch.arange(0, n)]).to(z.device)
        return F.cross_entropy(sim, targets)


class BarlowTwinsLoss(nn.Module):
    """Drive the cross-correlation of the two views to the identity."""

    def __init__(self, lambda_offdiag: float = 5e-3) -> None:
        super().__init__()
        self.lambda_offdiag = float(lambda_offdiag)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        n, d = z1.shape
        if n < 2:
            raise SSLError("Barlow Twins needs batch size >= 2")
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
        c = z1.T @ z2 / n                                    # (d, d)
        on_diag = ((torch.diagonal(c) - 1) ** 2).sum()
        off_diag = (c ** 2).sum() - (torch.diagonal(c) ** 2).sum()
        return on_diag + self.lambda_offdiag * off_diag


class VICRegLoss(nn.Module):
    """Variance + invariance + covariance regularization."""

    def __init__(self, sim_weight: float = 25.0, var_weight: float = 25.0,
                 cov_weight: float = 1.0, gamma: float = 1.0) -> None:
        super().__init__()
        self.sim_weight = float(sim_weight)
        self.var_weight = float(var_weight)
        self.cov_weight = float(cov_weight)
        self.gamma = float(gamma)

    @staticmethod
    def _cov_term(z: torch.Tensor) -> torch.Tensor:
        n, d = z.shape
        z = z - z.mean(0)
        cov = z.T @ z / (n - 1)
        off = cov - torch.diag_embed(torch.diagonal(cov))
        return (off ** 2).sum() / d

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if len(z1) < 2:
            raise SSLError("VICReg needs batch size >= 2")
        sim = F.mse_loss(z1, z2)
        std1 = torch.sqrt(z1.var(0) + 1e-4)
        std2 = torch.sqrt(z2.var(0) + 1e-4)
        var = (F.relu(self.gamma - std1).mean()
               + F.relu(self.gamma - std2).mean())
        cov = self._cov_term(z1) + self._cov_term(z2)
        return (self.sim_weight * sim + self.var_weight * var
                + self.cov_weight * cov)


def build_ssl_loss(cfg: dict | None) -> nn.Module:
    """Config → SSL loss module (raises :class:`SSLError`)."""
    cfg = cfg or {}
    name = str(cfg.get("name", "nt_xent")).lower()
    if name in ("nt_xent", "ntxent", "infonce", "simclr"):
        return NTXentLoss(temperature=float(cfg.get("temperature", 0.5)))
    if name in ("barlow_twins", "barlow"):
        return BarlowTwinsLoss(
            lambda_offdiag=float(cfg.get("lambda_offdiag", 5e-3)))
    if name == "vicreg":
        return VICRegLoss(sim_weight=float(cfg.get("sim_weight", 25.0)),
                          var_weight=float(cfg.get("var_weight", 25.0)),
                          cov_weight=float(cfg.get("cov_weight", 1.0)),
                          gamma=float(cfg.get("gamma", 1.0)))
    raise SSLError(f"unsupported SSL loss '{name}' (supported: nt_xent, "
                   f"barlow_twins, vicreg)")
