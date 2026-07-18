"""Model registry for the Deep Learning module.

Mirrors :class:`pipeline.ml.registry.ModelRegistry`, adapted to PyTorch
artefacts under ``models/deep_learning/``:

* ``<name>_best.pt`` / ``<name>_last.pt`` — per-network checkpoints
* ``best_model.pt`` / ``last_model.pt``   — the winning network
* ``training_config.json``                — architectures + hyperparameters
* ``metrics.json``                        — per-model per-split metrics
* ``history.json``                        — per-epoch training traces
* ``feature_metadata.json``               — feature schema + store version
* ``registry.json``                       — versioned entry index
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone

from ingestion.logging_config import get_logger

from .base import DLError, TrainedNetwork

log = get_logger("dl.registry")


class DLModelRegistry:
    """Versioned on-disk registry of trained networks."""

    def __init__(self, models_dir: str = "models/deep_learning") -> None:
        self.models_dir = models_dir
        self.registry_path = os.path.join(models_dir, "registry.json")
        os.makedirs(models_dir, exist_ok=True)

    # ── write ─────────────────────────────────────────────────────────────────
    def register(self, trained: TrainedNetwork, features: list[str],
                 dataset_version: str, target_col: str) -> dict:
        """Record one trained network's metadata; return the entry."""
        registry = self._read()
        version = f"v{len(registry['models']) + 1:03d}"
        entry = {
            "model_version": version,
            "network": trained.name,
            "checkpoints": trained.checkpoints,
            "architecture": trained.architecture,
            "hyperparameters": _jsonable(trained.hyperparameters),
            "n_parameters": trained.n_parameters,
            "device": trained.device,
            "threshold": trained.threshold,
            "threshold_method": trained.threshold_method,
            "metrics": {split: ev.metrics
                        for split, ev in trained.evaluations.items()},
            "best_epoch": (trained.history.best_epoch
                           if trained.history else None),
            "epochs_trained": (len(trained.history.epochs)
                               if trained.history else 0),
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
            "train_seconds": round(trained.train_seconds, 3),
            "feature_schema": {"target_col": target_col,
                               "n_features": len(features),
                               "features": features},
            "dataset_version": dataset_version,
        }
        registry["models"].append(entry)
        self._write(registry)
        log.info("registered %s as %s", trained.name, version)
        return entry

    def register_best(self, trained: TrainedNetwork, entry: dict,
                      features: list[str], dataset_version: str,
                      target_col: str, config: dict) -> dict[str, str]:
        """Save the winning network's artefact suite; return the paths."""
        paths: dict[str, str] = {}
        for tag, target in (("best", "best_model.pt"),
                            ("last", "last_model.pt")):
            src = trained.checkpoints.get(tag)
            if src and os.path.exists(src):
                dst = os.path.join(self.models_dir, target)
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copy2(src, dst)
                paths[target] = dst

        best_descriptor = {**entry, "network": trained.name,
                           "artefact": paths.get("best_model.pt")}
        paths["training_config.json"] = self._dump(
            "training_config.json",
            {"best_model": best_descriptor, "config": _jsonable(config)})
        paths["metrics.json"] = self._dump(
            "metrics.json",
            {split: ev.as_dict()
             for split, ev in trained.evaluations.items()})
        paths["history.json"] = self._dump(
            "history.json",
            trained.history.as_dict() if trained.history else {})
        paths["feature_metadata.json"] = self._dump(
            "feature_metadata.json",
            {"dataset_version": dataset_version, "target_col": target_col,
             "n_features": len(features), "features": features})

        registry = self._read()
        registry["best_model"] = {"network": trained.name,
                                  "model_version": entry["model_version"],
                                  "artefact": paths.get("best_model.pt")}
        self._write(registry)
        log.info("best deep model: %s (%s) -> %s", trained.name,
                 entry["model_version"], paths.get("best_model.pt"))
        return paths

    # ── read ──────────────────────────────────────────────────────────────────
    def entries(self) -> list[dict]:
        return self._read()["models"]

    def best(self) -> dict | None:
        return self._read().get("best_model")

    def best_checkpoint(self) -> str:
        best = self.best()
        if not best or not best.get("artefact"):
            raise DLError("registry has no best deep model")
        return best["artefact"]

    # ── plumbing ──────────────────────────────────────────────────────────────
    def _read(self) -> dict:
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as exc:
                raise DLError(
                    f"corrupt registry {self.registry_path}: {exc}") from exc
        return {"models": [], "best_model": None}

    def _write(self, registry: dict) -> None:
        registry["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, default=str)

    def _dump(self, name: str, payload: dict) -> str:
        path = os.path.join(self.models_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return path


def _jsonable(obj):
    """Round-trip through json (default=str) to guarantee serialisability."""
    return json.loads(json.dumps(obj, default=str))
