"""Data loading for the Deep Learning module.

Reuses :class:`pipeline.ml.data_loader.MLDataLoader` — the same verified
feature-store view the classical ML stage trains on — and wraps each split
in a PyTorch ``Dataset``/``DataLoader`` pair with configurable batch size,
shuffling, and seeded (reproducible) shuffling order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ingestion.logging_config import get_logger
from pipeline.ml.data_loader import MLDataLoader, MLDataset

from .base import DLError

log = get_logger("dl.data")


class TabularDataset(Dataset):
    """Feature/target tensors for one split of the engineered data."""

    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        if len(X) == 0:
            raise DLError("cannot build a dataset from an empty split")
        if len(X) != len(y):
            raise DLError(f"feature/target length mismatch "
                          f"({len(X)} vs {len(y)})")
        self.X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        self.y = torch.as_tensor(np.asarray(y, dtype=np.float32))
        if not torch.isfinite(self.X).all():
            raise DLError("dataset contains non-finite feature values")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


@dataclass
class DLData:
    """The ready-to-train tensor view of one feature-store version."""

    dataset: MLDataset                            # the verified store view
    loaders: dict[str, DataLoader] = field(default_factory=dict)
    tensors: dict[str, TabularDataset] = field(default_factory=dict)
    n_features: int = 0
    pos_weight: float = 1.0                       # neg/pos for weighted BCE

    @property
    def version(self) -> str:
        return self.dataset.version

    @property
    def features(self) -> list[str]:
        return self.dataset.features

    @property
    def target_col(self) -> str:
        return self.dataset.target_col

    def numpy(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        X, y = self.dataset.xy(split)
        return (np.asarray(X, dtype=np.float32),
                np.asarray(y, dtype=np.float32))


class DLDataLoader:
    """Load + verify engineered splits and build PyTorch loaders."""

    def __init__(self, store_dir: str = "data/features",
                 target_col: str = "Bankrupt?",
                 batch_size: int = 64, shuffle: bool = True,
                 num_workers: int = 0, seed: int = 42) -> None:
        if batch_size < 1:
            raise DLError(f"invalid batch_size {batch_size}")
        self.loader = MLDataLoader(store_dir, target_col)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.num_workers = int(num_workers)
        self.seed = int(seed)

    def load(self, version: str | None = None) -> DLData:
        """Return verified tensors + loaders (raises :class:`DLError`)."""
        dataset = self.loader.load(version)          # full ML verification
        data = DLData(dataset=dataset,
                      n_features=len(dataset.features))

        for split in dataset.splits:
            X, y = dataset.xy(split)
            tds = TabularDataset(X, y)
            data.tensors[split] = tds
            generator = torch.Generator().manual_seed(self.seed)
            data.loaders[split] = DataLoader(
                tds, batch_size=self.batch_size,
                shuffle=self.shuffle and split == "train",
                num_workers=self.num_workers, generator=generator)

        y_train = np.asarray(dataset.xy("train")[1])
        pos = int((y_train == 1).sum())
        data.pos_weight = float((y_train == 0).sum() / pos) if pos else 1.0
        log.info("built loaders (batch=%d): %s | %d features | "
                 "pos_weight=%.2f", self.batch_size,
                 ", ".join(f"{n}={len(t)}" for n, t in data.tensors.items()),
                 data.n_features, data.pos_weight)
        return data
