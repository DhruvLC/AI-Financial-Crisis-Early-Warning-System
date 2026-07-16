"""Model Registry and persistence for the ML module.

A lightweight file-based registry mirroring the
:class:`~pipeline.feature_engineering.store.FeatureStore` design: each
registered model gets a versioned entry in ``models/registry.json`` plus a
joblib artefact under ``models/``; the best model is additionally saved to
``models/best_model.joblib`` with a ``models/best_model.json`` descriptor.
Every entry records the algorithm, hyperparameters, metrics, training
timestamp, feature schema, and the feature-store dataset version — enough to
reproduce or audit any trained model.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib

from ingestion.logging_config import get_logger

from .base import TrainedModel

log = get_logger("ml.registry")


class ModelRegistry:
    """Versioned on-disk registry of trained models."""

    def __init__(self, models_dir: str = "models") -> None:
        self.models_dir = models_dir
        self.registry_path = os.path.join(models_dir, "registry.json")
        os.makedirs(models_dir, exist_ok=True)

    # ── write ─────────────────────────────────────────────────────────────────
    def register(self, trained: TrainedModel, features: list[str],
                 dataset_version: str, target_col: str) -> dict:
        """Persist a trained model + its metadata; return the entry."""
        registry = self._read()
        version = f"v{len(registry['models']) + 1:03d}"
        artefact = os.path.join(self.models_dir,
                                f"{trained.name}_{version}.joblib")
        joblib.dump(trained.model, artefact)

        entry = {
            "model_version": version,
            "algorithm": trained.name,
            "artefact": artefact,
            "hyperparameters": _jsonable(trained.hyperparameters),
            "tuning": _jsonable(trained.tuning),
            "cv_scores": trained.cv_scores,
            "threshold": trained.threshold,
            "threshold_method": trained.threshold_method,
            "metrics": {split: ev.metrics
                        for split, ev in trained.evaluations.items()},
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
            "train_seconds": round(trained.train_seconds, 3),
            "feature_schema": {"target_col": target_col,
                               "n_features": len(features),
                               "features": features},
            "dataset_version": dataset_version,
        }
        registry["models"].append(entry)
        self._write(registry)
        log.info("registered %s as %s -> %s", trained.name, version, artefact)
        return entry

    def register_best(self, trained: TrainedModel, entry: dict) -> str:
        """Save the winning model separately and point the registry at it."""
        best_path = os.path.join(self.models_dir, "best_model.joblib")
        joblib.dump(trained.model, best_path)
        descriptor = {**entry, "artefact": best_path}
        with open(os.path.join(self.models_dir, "best_model.json"), "w",
                  encoding="utf-8") as f:
            json.dump(descriptor, f, indent=2, default=str)
        registry = self._read()
        registry["best_model"] = {"algorithm": trained.name,
                                  "model_version": entry["model_version"],
                                  "artefact": best_path}
        self._write(registry)
        log.info("best model: %s (%s) -> %s", trained.name,
                 entry["model_version"], best_path)
        return best_path

    # ── read ──────────────────────────────────────────────────────────────────
    def entries(self) -> list[dict]:
        return self._read()["models"]

    def best(self) -> dict | None:
        return self._read().get("best_model")

    def load_model(self, algorithm: str, version: str | None = None):
        """Load a registered model's artefact (latest entry by default)."""
        matches = [e for e in self.entries() if e["algorithm"] == algorithm
                   and (version is None or e["model_version"] == version)]
        if not matches:
            raise FileNotFoundError(
                f"no registry entry for '{algorithm}'"
                + (f" {version}" if version else ""))
        return joblib.load(matches[-1]["artefact"])

    def load_best(self):
        best = self.best()
        if not best:
            raise FileNotFoundError("registry has no best model")
        return joblib.load(best["artefact"])

    # ── plumbing ──────────────────────────────────────────────────────────────
    def _read(self) -> dict:
        if os.path.exists(self.registry_path):
            with open(self.registry_path, encoding="utf-8") as f:
                return json.load(f)
        return {"models": [], "best_model": None}

    def _write(self, registry: dict) -> None:
        registry["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, default=str)


def _jsonable(obj):
    """Round-trip through json (default=str) to guarantee serialisability."""
    return json.loads(json.dumps(obj, default=str))
