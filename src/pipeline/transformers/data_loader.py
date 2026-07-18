"""Data loading for the Transformer Models module.

Thin, explicit reuse of :class:`pipeline.deep_learning.data_loader.
DLDataLoader` — the transformer stage trains on exactly the same verified
feature-store view (train/val/test parquet splits + metadata) and the same
PyTorch ``Dataset``/``DataLoader`` machinery (batching, seeded shuffling,
configurable batch size) as the deep-learning stage, so the two families
are directly comparable.

Aliased under transformer names so call sites and reports read naturally
and so a future divergence (e.g. per-feature tokenized datasets) only
touches this file.
"""
from __future__ import annotations

from ingestion.logging_config import get_logger
from pipeline.deep_learning.data_loader import (DLData, DLDataLoader,
                                                TabularDataset)

__all__ = ["TransformerData", "TransformerDataLoader", "TabularDataset"]

log = get_logger("transformers.data")

# The tensor view of one feature-store version (loaders, tensors,
# pos_weight, feature names) — identical needs, identical container.
TransformerData = DLData


class TransformerDataLoader(DLDataLoader):
    """Load + verify engineered splits and build PyTorch loaders.

    Identical semantics to :class:`DLDataLoader`; kept as a subclass so the
    transformer pipeline logs under its own name and can specialize later
    without touching the deep-learning stage.
    """

    def load(self, version: str | None = None) -> TransformerData:
        data = super().load(version)
        log.info("transformer data ready: version=%s, %d features",
                 data.version, data.n_features)
        return data
