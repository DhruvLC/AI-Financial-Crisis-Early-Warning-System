"""Inference for the Deep Learning module.

:class:`DLPredictor` reconstructs a registered network from its
``training_config.json`` + ``best_model.pt`` and scores new feature frames —
enforcing the stored feature schema so training/serving skew fails loudly
instead of silently.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from ingestion.logging_config import get_logger

from .base import DLError, resolve_device
from .evaluation import predict_proba
from .models import build_network
from .trainer import load_checkpoint

log = get_logger("dl.prediction")


class DLPredictor:
    """Load a registered deep model and predict on new data."""

    def __init__(self, registry_dir: str = "models/deep_learning",
                 device: str = "auto") -> None:
        self.registry_dir = registry_dir
        self.device = resolve_device(device)
        self.model = None
        self.threshold = 0.5
        self.features: list[str] = []
        self.config: dict = {}

    # ── loading ───────────────────────────────────────────────────────────────
    def load(self, checkpoint: str | None = None) -> "DLPredictor":
        """Load the best (default) or an explicit checkpoint; return self."""
        cfg_path = os.path.join(self.registry_dir, "training_config.json")
        meta_path = os.path.join(self.registry_dir, "feature_metadata.json")
        for path in (cfg_path, meta_path):
            if not os.path.exists(path):
                raise DLError(f"registry incomplete: missing {path}")
        with open(cfg_path, encoding="utf-8") as f:
            self.config = json.load(f)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.features = list(meta.get("features", []))
        if not self.features:
            raise DLError("feature_metadata.json has no feature list")

        best = self.config.get("best_model", {})
        checkpoint = checkpoint or os.path.join(self.registry_dir,
                                                "best_model.pt")
        state = load_checkpoint(checkpoint, self.device)
        network = best.get("network") or state.get("name")
        params = best.get("architecture", {})
        params = {k: v for k, v in params.items()
                  if k not in ("network", "n_features")}
        self.model = build_network(network, len(self.features), params)
        self.model.load_state_dict(state["model_state"])
        self.model.to(self.device).eval()
        self.threshold = float(best.get("threshold", 0.5))
        log.info("loaded %s from %s (%d features, threshold=%.3f)",
                 network, checkpoint, len(self.features), self.threshold)
        return self

    # ── inference ─────────────────────────────────────────────────────────────
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Positive-class probabilities for a feature frame."""
        if self.model is None:
            raise DLError("predict before load() — call load() first")
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise DLError(f"{len(missing)} feature(s) missing at predict "
                          f"time (e.g. {missing[:3]})")
        return predict_proba(self.model,
                             np.asarray(X[self.features], dtype=np.float32),
                             self.device)

    def predict(self, X: pd.DataFrame,
                threshold: float | None = None) -> np.ndarray:
        """Binary crisis labels at the stored (or given) threshold."""
        t = self.threshold if threshold is None else float(threshold)
        return (self.predict_proba(X) >= t).astype(int)

    def risk_scores(self, X: pd.DataFrame, scale: float = 100.0) -> np.ndarray:
        """Probability mapped to a 0..scale risk score."""
        return self.predict_proba(X) * scale
