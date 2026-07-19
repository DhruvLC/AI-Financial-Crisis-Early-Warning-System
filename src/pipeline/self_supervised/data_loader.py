"""Data loading for the Self-Supervised Learning module.

Thin, explicit reuse of :class:`pipeline.deep_learning.data_loader.
DLDataLoader` — SSL pretrains on exactly the same verified feature-store
view (train/val/test parquet splits + metadata) and the same PyTorch
``Dataset``/``DataLoader`` machinery (batching, seeded shuffling,
configurable batch size) as the deep-learning and transformer stages.

Adds :class:`ContrastiveDataset`, which wraps one split's feature tensor
and applies a configurable augmentation pipeline **twice** per sample to
produce the two correlated views SimCLR-style objectives need. Labels are
never used during pretraining — they are only carried through so the
linear-probe evaluation can reuse the same loaders.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from ingestion.logging_config import get_logger
from pipeline.deep_learning.data_loader import (DLData, DLDataLoader,
                                                TabularDataset)

from .base import SSLError

__all__ = ["SSLData", "SSLDataLoader", "ContrastiveDataset",
           "TabularDataset", "build_contrastive_loader"]

log = get_logger("ssl.data")

# The tensor view of one feature-store version (loaders, tensors,
# pos_weight, feature names) — identical needs, identical container.
SSLData = DLData


class ContrastiveDataset(Dataset):
    """Two augmented views of each sample for contrastive pretraining.

    Wraps a :class:`TabularDataset` and an augmentation callable
    ``(x, generator) -> x'``. Each ``__getitem__`` returns
    ``(view_1, view_2, y)``; the label is passed through untouched (unused
    by the loss, kept for probe evaluation convenience). A per-dataset
    seeded :class:`torch.Generator` makes view sampling deterministic.
    """

    def __init__(self, base: TabularDataset, augment, seed: int = 42) -> None:
        if augment is None:
            raise SSLError("ContrastiveDataset requires an augmentation "
                           "pipeline (identity views collapse the loss)")
        self.base = base
        self.augment = augment
        self.generator = torch.Generator().manual_seed(int(seed))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return (self.augment(x, self.generator),
                self.augment(x, self.generator), y)


def build_contrastive_loader(dataset: ContrastiveDataset, batch_size: int,
                             shuffle: bool, seed: int = 42,
                             num_workers: int = 0,
                             drop_last: bool = True) -> DataLoader:
    """Seeded loader over two-view batches.

    ``drop_last`` defaults to True for training: NT-Xent degenerates on
    batches of size 1 (no negatives).
    """
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle,
                      num_workers=int(num_workers), generator=generator,
                      drop_last=drop_last)


class SSLDataLoader(DLDataLoader):
    """Load + verify engineered splits and build PyTorch loaders.

    Identical semantics to :class:`DLDataLoader` (which already validates
    finite features, non-empty splits, and length agreement); kept as a
    subclass so the SSL pipeline logs under its own name and adds
    :meth:`contrastive_loaders` for the two-view pretraining loaders.
    """

    def load(self, version: str | None = None) -> SSLData:
        data = super().load(version)
        log.info("SSL data ready: version=%s, %d features (labels unused "
                 "during pretraining)", data.version, data.n_features)
        return data

    def contrastive_loaders(self, data: SSLData, augment,
                            splits: tuple[str, ...] = ("train", "val")
                            ) -> dict[str, DataLoader]:
        """Two-view loaders for the given splits (train shuffled)."""
        loaders: dict[str, DataLoader] = {}
        for split in splits:
            if split not in data.tensors:
                raise SSLError(f"split '{split}' not in the feature store")
            ds = ContrastiveDataset(data.tensors[split], augment,
                                    seed=self.seed)
            loaders[split] = build_contrastive_loader(
                ds, self.batch_size,
                shuffle=self.shuffle and split == "train",
                seed=self.seed, num_workers=self.num_workers,
                drop_last=split == "train")
        log.info("built contrastive loaders (batch=%d): %s",
                 self.batch_size,
                 ", ".join(f"{s}={len(l.dataset)}"
                           for s, l in loaders.items()))
        return loaders
