"""Model loading service — wraps the existing ModelRegistry + FeatureStore.

Loads (once, at startup) the best registered model, its descriptor
(threshold, feature schema, metrics), and the feature-store metadata.
Never retrains anything.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from ingestion.logging_config import get_logger
from pipeline.feature_engineering.store import FeatureStore
from pipeline.ml.registry import ModelRegistry

from .config import APISettings
from .exceptions import (MissingArtifactError, ModelLoadingError,
                         ModelNotLoadedError, RegistryError)

log = get_logger("api.model_loader")

#: Registry family for the top-level ``models/registry.json``.
CLASSICAL_FAMILY = "classical_ml"


def _normalise_entry(entry: dict, family: str) -> dict:
    """Map a registry entry of any model family onto the classical schema.

    Non-classical registries (deep learning, transformers, self-supervised)
    record ``network`` instead of ``algorithm`` and ``checkpoints`` instead of
    a single ``artefact``; normalising here lets every downstream consumer
    (routes, schemas, frontend) treat all families uniformly.
    """
    norm = dict(entry)
    norm.setdefault("algorithm", entry.get("network", "unknown"))
    if "artefact" not in norm:
        checkpoints = entry.get("checkpoints", {}) or {}
        norm["artefact"] = checkpoints.get("best") or checkpoints.get("last", "")
    norm["family"] = family
    return norm


def discover_registries(models_dir: str) -> dict[str, str]:
    """Find every ``registry.json`` under ``models_dir`` (one level deep).

    Returns ``{family: registry_path}`` — the top-level registry is the
    classical-ML family; each immediate subdirectory containing a
    ``registry.json`` is its own family (deep_learning, transformers,
    self_supervised, and any future addition).
    """
    registries: dict[str, str] = {}
    top = os.path.join(models_dir, "registry.json")
    if os.path.exists(top):
        registries[CLASSICAL_FAMILY] = top
    try:
        subdirs = sorted(os.listdir(models_dir))
    except OSError:
        subdirs = []
    for name in subdirs:
        candidate = os.path.join(models_dir, name, "registry.json")
        if os.path.isfile(candidate):
            registries[name] = candidate
    return registries


class ModelService:
    """Holds the loaded best model + all metadata needed for inference."""

    def __init__(self, settings: APISettings) -> None:
        self.settings = settings
        self.model = None
        self.descriptor: dict = {}
        self.feature_metadata: dict = {}
        self.loaded_at: str | None = None
        self.registry = ModelRegistry(models_dir=settings.models_dir)
        self.feature_store = FeatureStore(root=settings.feature_store_root)

    # ── loading ───────────────────────────────────────────────────────────────
    def load(self) -> None:
        """Load best model artefact, descriptor, and feature-store metadata."""
        descriptor_path = os.path.join(self.settings.models_dir,
                                       "best_model.json")
        if not os.path.exists(descriptor_path):
            raise MissingArtifactError(
                f"best model descriptor not found: {descriptor_path}")
        try:
            with open(descriptor_path, encoding="utf-8") as f:
                self.descriptor = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(f"cannot read best model descriptor: {exc}")

        try:
            self.model = self.registry.load_best()
        except FileNotFoundError as exc:
            raise MissingArtifactError(str(exc))
        except Exception as exc:  # noqa: BLE001 - joblib/env failures
            raise ModelLoadingError(f"failed to load model artefact: {exc}")

        # Feature-store metadata for the dataset version the model trained on.
        try:
            version = (self.descriptor.get("dataset_version")
                       or self.feature_store.latest_version())
            meta_path = os.path.join(self.settings.feature_store_root,
                                     version or "", "metadata.json")
            with open(meta_path, encoding="utf-8") as f:
                self.feature_metadata = json.load(f)
        except Exception as exc:  # noqa: BLE001 - metadata is informational
            log.warning("feature store metadata unavailable: %s", exc)
            self.feature_metadata = {}

        self.loaded_at = datetime.now(timezone.utc).isoformat()
        log.info("loaded best model %s (%s), threshold=%.4f, %d features",
                 self.descriptor.get("model_version"),
                 self.descriptor.get("algorithm"),
                 self.threshold, len(self.features))

    # ── accessors ─────────────────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def require_loaded(self) -> None:
        if not self.is_loaded:
            raise ModelNotLoadedError("model is not loaded")

    @property
    def features(self) -> list[str]:
        return self.descriptor.get("feature_schema", {}).get("features", [])

    @property
    def threshold(self) -> float:
        return float(self.descriptor.get("threshold", 0.5))

    @property
    def model_version(self) -> str:
        return self.descriptor.get("model_version", "unknown")

    @property
    def algorithm(self) -> str:
        return self.descriptor.get("algorithm", "unknown")

    @property
    def dataset_version(self) -> str | None:
        return self.descriptor.get("dataset_version")

    def registry_entries(self) -> list[dict]:
        """All entries across every discovered registry, normalised.

        The classical registry is read through :class:`ModelRegistry`; every
        subdirectory registry (deep learning, transformers, self-supervised,
        or any future family) is read directly from its ``registry.json``.
        """
        try:
            entries: list[dict] = []
            for family, path in discover_registries(
                    self.settings.models_dir).items():
                if family == CLASSICAL_FAMILY:
                    raw = self.registry.entries()
                else:
                    with open(path, encoding="utf-8") as f:
                        raw = json.load(f).get("models", [])
                entries.extend(_normalise_entry(e, family) for e in raw)
            return entries
        except Exception as exc:  # noqa: BLE001
            raise RegistryError(f"cannot read model registry: {exc}")

    def best_entry(self) -> dict | None:
        try:
            return self.registry.best()
        except Exception as exc:  # noqa: BLE001
            raise RegistryError(f"cannot read model registry: {exc}")
