"""Production inference pipeline.

Validates incoming feature payloads against the trained model's feature
schema, assembles a correctly-ordered feature matrix, and returns
prediction + probability + 0-100 risk score + confidence, reusing the
risk-banding logic of :mod:`pipeline.risk_score`.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ingestion.logging_config import get_logger

from .exceptions import InvalidRequestError, PredictionError
from .model_loader import ModelService

log = get_logger("api.predict")


class InferencePipeline:
    """Stateless-per-request inference over a loaded :class:`ModelService`."""

    def __init__(self, service: ModelService) -> None:
        self.service = service

    # ── validation ────────────────────────────────────────────────────────────
    def validate_instances(self, instances: list[dict]) -> dict:
        """Validate raw feature dicts against the model schema.

        Returns ``{"valid": bool, "errors": [...], "warnings": [...]}`` with
        one issue per (instance, feature) problem — mirrors the structured
        style of the project's data-validation reports.
        """
        self.service.require_loaded()
        expected = self.service.features
        expected_set = set(expected)
        errors, warnings = [], []
        for i, features in enumerate(instances):
            if not isinstance(features, dict) or not features:
                errors.append({"field": f"instances[{i}]",
                               "issue": "features must be a non-empty object"})
                continue
            missing = expected_set - set(features)
            extra = set(features) - expected_set
            for name in sorted(missing):
                errors.append({"field": f"instances[{i}].{name}",
                               "issue": "missing required feature"})
            for name in sorted(extra):
                warnings.append({"field": f"instances[{i}].{name}",
                                 "issue": "unknown feature (ignored)"})
            for name, value in features.items():
                if name not in expected_set:
                    continue
                if value is None or (isinstance(value, float)
                                     and (math.isnan(value)
                                          or math.isinf(value))):
                    errors.append({"field": f"instances[{i}].{name}",
                                   "issue": "value must be a finite number"})
                elif not isinstance(value, (int, float)):
                    errors.append({"field": f"instances[{i}].{name}",
                                   "issue": f"expected number, got "
                                            f"{type(value).__name__}"})
        return {"valid": not errors, "errors": errors, "warnings": warnings}

    def _matrix(self, instances: list[dict]) -> pd.DataFrame:
        """Validate and build the model-ordered feature matrix."""
        report = self.validate_instances(instances)
        if not report["valid"]:
            raise InvalidRequestError("feature validation failed",
                                      detail=report["errors"])
        # Enforce exact feature order the model was trained with.
        return pd.DataFrame(
            [[float(inst[f]) for f in self.service.features]
             for inst in instances],
            columns=self.service.features)

    # ── inference ─────────────────────────────────────────────────────────────
    def predict(self, instances: list[dict],
                ids: list | None = None) -> tuple[list[dict], float]:
        """Run inference; return ``(results, inference_ms)``."""
        svc = self.service
        svc.require_loaded()
        X = self._matrix(instances)
        t0 = time.perf_counter()
        try:
            proba = np.asarray(svc.model.predict_proba(X), dtype=float)
            # ML-module wrappers already return the 1-D positive-class
            # probability; raw sklearn estimators return (n, 2).
            if proba.ndim == 2:
                proba = proba[:, 1]
        except Exception as exc:  # noqa: BLE001 - model runtime failure
            raise PredictionError(f"model inference failed: {exc}")
        inference_ms = (time.perf_counter() - t0) * 1000

        scale = svc.settings.risk_score_scale
        threshold = svc.threshold
        timestamp = datetime.now(timezone.utc).isoformat()
        results = []
        for i, p in enumerate(proba):
            pred = int(p >= threshold)
            # Confidence: distance from the decision threshold, normalized to
            # 0-1 within the side of the threshold the probability falls on.
            span = (1.0 - threshold) if pred else threshold
            confidence = float(abs(p - threshold) / span) if span > 0 else 1.0
            risk = round(p * scale, 1)
            results.append({
                "id": ids[i] if ids else None,
                "prediction": pred,
                "probability": round(float(p), 6),
                "risk_score": risk,
                "risk_level": _band(risk, scale),
                "confidence_score": round(min(confidence, 1.0), 4),
                "threshold": round(threshold, 6),
                "model_version": svc.model_version,
                "algorithm": svc.algorithm,
                "prediction_timestamp": timestamp,
            })
        log.info("predicted %d instance(s) in %.2f ms (model %s)",
                 len(results), inference_ms, svc.model_version)
        return results, round(inference_ms, 3)


def _band(risk: float, scale: int) -> str:
    """Same banding as :func:`pipeline.risk_score.score`."""
    if risk < scale * 0.33:
        return "Low"
    if risk < scale * 0.66:
        return "Medium"
    return "High"
