"""Feature store — versioned persistence of engineered feature sets.

A lightweight file-based store under ``data/features/``: each run writes a new
version directory (``v001``, ``v002``, …) holding the engineered train/val/test
splits (parquet, CSV fallback) plus a ``metadata.json`` describing the feature
set — final schema, per-step lineage, importance scores, config, and content
hashes for reproducibility checks. ``latest.json`` at the store root points to
the newest version, so downstream (ML) stages can load features without knowing
version numbers.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd

from ingestion.logging_config import get_logger

log = get_logger("features.store")


class FeatureStore:
    """Versioned on-disk store for engineered feature sets."""

    def __init__(self, root: str = "data/features") -> None:
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    # ── write ─────────────────────────────────────────────────────────────────
    def save(self, splits: dict[str, pd.DataFrame], metadata: dict) -> dict:
        """Persist ``splits`` (+ metadata) as a new version; return its record."""
        version = self._next_version()
        vdir = os.path.join(self.root, version)
        os.makedirs(vdir, exist_ok=True)

        files, hashes = {}, {}
        for name, df in splits.items():
            path = os.path.join(vdir, f"{name}.parquet")
            try:
                df.to_parquet(path, index=False)
            except Exception as exc:  # noqa: BLE001 - parquet engine optional
                path = os.path.join(vdir, f"{name}.csv")
                df.to_csv(path, index=False)
                log.warning("parquet unavailable (%s); wrote CSV instead", exc)
            files[name] = path
            hashes[name] = self._hash(df)
            log.info("feature store: wrote %s (%d rows x %d cols)",
                     path, len(df), df.shape[1])

        record = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
            "hashes": hashes,
            "splits": {n: {"rows": len(d), "cols": d.shape[1]}
                       for n, d in splits.items()},
            **metadata,
        }
        with open(os.path.join(vdir, "metadata.json"), "w",
                  encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)
        with open(os.path.join(self.root, "latest.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"version": version, "path": vdir}, f, indent=2)
        log.info("feature store: registered version %s", version)
        return record

    # ── read ──────────────────────────────────────────────────────────────────
    def load(self, version: str | None = None) -> tuple[dict, dict]:
        """Return ``(splits, metadata)`` for ``version`` (default: latest)."""
        version = version or self.latest_version()
        if version is None:
            raise FileNotFoundError(f"feature store at {self.root} is empty")
        vdir = os.path.join(self.root, version)
        meta_path = os.path.join(vdir, "metadata.json")
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        splits = {}
        for name, path in metadata.get("files", {}).items():
            reader = pd.read_parquet if path.endswith(".parquet") else pd.read_csv
            splits[name] = reader(path)
        return splits, metadata

    def latest_version(self) -> str | None:
        pointer = os.path.join(self.root, "latest.json")
        if os.path.exists(pointer):
            with open(pointer, encoding="utf-8") as f:
                return json.load(f).get("version")
        versions = self.list_versions()
        return versions[-1] if versions else None

    def list_versions(self) -> list[str]:
        if not os.path.isdir(self.root):
            return []
        return sorted(d for d in os.listdir(self.root)
                      if d.startswith("v") and
                      os.path.isdir(os.path.join(self.root, d)))

    # ── helpers ───────────────────────────────────────────────────────────────
    def _next_version(self) -> str:
        versions = self.list_versions()
        last = int(versions[-1][1:]) if versions else 0
        return f"v{last + 1:03d}"

    @staticmethod
    def _hash(df: pd.DataFrame) -> str:
        """Stable content hash of a frame (schema + values)."""
        h = hashlib.sha256()
        h.update(",".join(df.columns).encode())
        h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
        return h.hexdigest()[:16]
