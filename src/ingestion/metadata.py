"""Metadata sidecar writer — one .meta.json per ingested dataset."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd


class MetadataWriter:
    """Writes a JSON metadata file describing a stored dataset."""

    def __init__(self, layer_dir: str) -> None:
        self.layer_dir = layer_dir
        os.makedirs(layer_dir, exist_ok=True)

    @staticmethod
    def _checksum(path: str) -> str | None:
        if not os.path.exists(path):
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def write(
        self,
        source: str,
        df: pd.DataFrame,
        raw_path: str | None,
        interim_path: str | None,
        validation: dict,
        extra: dict | None = None,
    ) -> str:
        meta = {
            "source": source,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "columns": list(map(str, df.columns)),
            "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
            "raw_path": raw_path,
            "interim_path": interim_path,
            "interim_checksum_sha256": self._checksum(interim_path or ""),
            "validation": validation,
            "extra": extra or {},
        }
        out_path = os.path.join(self.layer_dir, f"{source}.meta.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        return out_path
