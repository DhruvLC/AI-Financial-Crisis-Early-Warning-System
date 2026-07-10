"""Orchestrates all enabled ingestors with per-source failure isolation."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import yaml

from .base import IngestionResult
from .logging_config import configure_logging, get_logger
from .sources import SOURCE_REGISTRY


class IngestionRunner:
    """Loads config, instantiates enabled sources, runs them, writes a manifest."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        log_cfg = self.cfg.get("logging", {})
        configure_logging(log_cfg.get("level", "INFO"), log_cfg.get("file"))
        self.log = get_logger("ingest.runner")

        st = self.cfg["storage"]
        # Metadata sidecars live under the configured layer's dir (default: raw).
        layer = st.get("metadata_layer", "raw")
        layer_dir = st.get(f"{layer}_dir", st["raw_dir"])
        self.storage = {
            "raw_dir": st["raw_dir"],
            "interim_dir": st["interim_dir"],
            "metadata_layer_dir": os.path.join(layer_dir, "_metadata"),
        }
        os.makedirs(self.storage["metadata_layer_dir"], exist_ok=True)
        self.http_cfg = self.cfg.get("http", {})
        self.validation_cfg = self.cfg.get("validation", {})

    def _enabled_sources(self) -> list[str]:
        return [name for name, s in self.cfg.get("sources", {}).items()
                if s.get("enabled") and name in SOURCE_REGISTRY]

    def run(self, only: list[str] | None = None) -> list[IngestionResult]:
        targets = self._enabled_sources()
        if only:
            targets = [t for t in targets if t in only]
        self.log.info("ingestion targets: %s", targets)

        results: list[IngestionResult] = []
        for name in targets:
            source_cfg = self.cfg["sources"][name]
            ingestor = SOURCE_REGISTRY[name](
                source_cfg, self.storage, self.http_cfg, self.validation_cfg
            )
            results.append(ingestor.run())  # run() never raises — isolates failures

        self._write_manifest(results)
        ok = sum(r.status == "success" for r in results)
        self.log.info("ingestion complete: %d/%d succeeded", ok, len(results))
        return results

    def _write_manifest(self, results: list[IngestionResult]) -> None:
        manifest = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "results": [r.__dict__ for r in results],
        }
        path = os.path.join(self.storage["metadata_layer_dir"], "run_manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        self.log.info("manifest written: %s", path)
