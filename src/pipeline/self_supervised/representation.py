"""Representation extraction for the Self-Supervised Learning module.

Batched, no-grad extraction of encoder embeddings for every split, with
NaN validation via :func:`~pipeline.self_supervised.base.
embedding_statistics`, plus persistence of the latent representations as
parquet files (embedding columns + the original target) so downstream
modules can consume them exactly like an engineered feature set.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from ingestion.logging_config import get_logger

from .base import SSLError, embedding_statistics
from .data_loader import SSLData

__all__ = ["extract_embeddings", "RepresentationExporter"]

log = get_logger("ssl.representation")


@torch.no_grad()
def extract_embeddings(encoder: torch.nn.Module, X: np.ndarray,
                       device: torch.device | str = "cpu",
                       batch_size: int = 1024) -> np.ndarray:
    """Encoder embeddings for ``X`` (batched, validated)."""
    if len(X) == 0:
        raise SSLError("cannot extract embeddings from empty input")
    X = np.asarray(X, dtype=np.float32)
    if not np.isfinite(X).all():
        raise SSLError("extraction input contains non-finite values")
    encoder = encoder.to(device)
    encoder.eval()
    out = []
    for i in range(0, len(X), batch_size):
        batch = torch.as_tensor(X[i:i + batch_size]).to(device)
        z = encoder(batch)
        if not torch.isfinite(z).all():
            raise SSLError("encoder produced non-finite embeddings")
        out.append(z.cpu().numpy())
    return np.concatenate(out).astype(np.float32)


class RepresentationExporter:
    """Persist per-split embeddings + metadata for downstream reuse.

    Layout under ``representations_dir``::

        <encoder>_train.parquet   # z_000..z_NNN + target column
        <encoder>_val.parquet
        <encoder>_test.parquet
        representation_metadata.json
    """

    def __init__(self, representations_dir: str =
                 "models/self_supervised/representations") -> None:
        self.dir = representations_dir
        os.makedirs(self.dir, exist_ok=True)

    def export(self, name: str, encoder: torch.nn.Module, data: SSLData,
               device: torch.device | str = "cpu",
               batch_size: int = 1024
               ) -> tuple[dict[str, str], dict[str, dict]]:
        """Extract + save every split; return (paths, embedding stats)."""
        paths: dict[str, str] = {}
        stats: dict[str, dict] = {}
        for split in data.tensors:
            X, y = data.numpy(split)
            Z = extract_embeddings(encoder, X, device, batch_size)
            stats[split] = embedding_statistics(Z)
            cols = [f"z_{i:03d}" for i in range(Z.shape[1])]
            df = pd.DataFrame(Z, columns=cols)
            df[data.target_col] = np.asarray(y)
            path = os.path.join(self.dir, f"{name}_{split}.parquet")
            df.to_parquet(path, index=False)
            paths[split] = path
            log.info("representations saved: %s (%d x %d, "
                     "mean_l2=%.3f)", path, *Z.shape,
                     stats[split]["mean_l2_norm"])
        return paths, stats

    def write_metadata(self, entries: list[dict],
                       dataset_version: str) -> str:
        """Write ``representation_metadata.json`` describing every export."""
        path = os.path.join(self.dir, "representation_metadata.json")
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_version": dataset_version,
            "representations": entries,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        log.info("representation metadata written: %s", path)
        return path
