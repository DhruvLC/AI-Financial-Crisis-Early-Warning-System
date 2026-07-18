"""Core datatypes and helpers for the Deep Learning module.

Mirrors :mod:`pipeline.ml.base`: a small set of shared containers plus the
exception type and the device/seed utilities every other deep-learning
component uses. Keeping the plumbing here lets the model, trainer, and
pipeline classes stay small, uniform, and independently testable.

The module is **config-driven and backend-aware**: :func:`resolve_device`
picks CUDA → MPS → CPU (with an explicit override), and :func:`seed_all`
makes training deterministic across NumPy and PyTorch.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ingestion.logging_config import get_logger

log = get_logger("dl.base")


class DLError(RuntimeError):
    """Raised for fatal deep-learning problems (empty data, invalid config,
    NaN losses, corrupt checkpoints, failed training) when the pipeline is
    configured to fail fast."""


@dataclass
class EpochRecord:
    """Metrics of one training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_metrics: dict[str, float] = field(default_factory=dict)
    lr: float = 0.0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {"epoch": self.epoch, "train_loss": self.train_loss,
                "val_loss": self.val_loss, "val_metrics": self.val_metrics,
                "lr": self.lr, "seconds": round(self.seconds, 3)}


@dataclass
class TrainingHistory:
    """Full per-epoch training trace of one model."""

    epochs: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    stopped_early: bool = False
    total_seconds: float = 0.0

    def add(self, record: EpochRecord) -> None:
        self.epochs.append(record)

    def as_dict(self) -> dict:
        return {"epochs": [e.as_dict() for e in self.epochs],
                "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val_loss,
                "stopped_early": self.stopped_early,
                "total_seconds": round(self.total_seconds, 3)}


@dataclass
class TrainedNetwork:
    """Everything one architecture produces after training + evaluation."""

    name: str
    model: Any                                   # the trained nn.Module
    architecture: dict = field(default_factory=dict)
    hyperparameters: dict = field(default_factory=dict)
    history: TrainingHistory | None = None
    evaluations: dict[str, Any] = field(default_factory=dict)
    threshold: float = 0.5
    threshold_method: str = "default"
    train_seconds: float = 0.0
    n_parameters: int = 0
    device: str = "cpu"
    checkpoints: dict[str, str] = field(default_factory=dict)
    permutation_importance: Any = None           # pd.DataFrame | None
    shap_summary: dict | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    def metric(self, name: str, split: str = "test") -> float:
        ev = self.evaluations.get(split)
        return float(ev.metrics.get(name, float("nan"))) if ev else float("nan")

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
            "evaluations": {s: e.as_dict()
                            for s, e in self.evaluations.items()},
        }


# ── device & determinism helpers ────────────────────────────────────────────
def resolve_device(preference: str = "auto") -> torch.device:
    """Pick the training device: explicit preference or CUDA → MPS → CPU.

    An unavailable explicit preference degrades to CPU with a warning
    instead of raising, so config written on a GPU box still runs anywhere.
    """
    preference = (preference or "auto").lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        log.warning("CUDA requested but unavailable — falling back to CPU")
        return torch.device("cpu")
    if preference == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        log.warning("MPS requested but unavailable — falling back to CPU")
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    log.info("seeded all RNGs with %d (deterministic=%s)", seed,
             deterministic)


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
