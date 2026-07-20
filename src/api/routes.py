"""REST API routes — prediction, validation, models, metrics."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from .config import get_settings
from .dependencies import inference_pipeline, model_service, monitor
from .exceptions import InvalidRequestError
from .model_loader import ModelService
from .monitoring import Monitor
from .predict import InferencePipeline
from .schemas import (BatchPredictionResponse, BatchPredictRequest,
                      MetricsResponse, ModelInfo, ModelsResponse,
                      PredictionResult, PredictRequest, ValidationResponse)

router = APIRouter()


# ── models & metrics ──────────────────────────────────────────────────────────
@router.get("/models", response_model=ModelsResponse, tags=["models"],
            summary="List all registered models")
def list_models(svc: ModelService = Depends(model_service)) -> ModelsResponse:
    best = svc.best_entry() or {}
    best_version = best.get("model_version")
    infos = [ModelInfo(
        model_version=e["model_version"],
        algorithm=e["algorithm"],
        artefact=e["artefact"],
        metrics=e.get("metrics", {}).get("test", {}),
        training_timestamp=e.get("training_timestamp"),
        is_best=e["model_version"] == best_version,
    ) for e in svc.registry_entries()]
    best_info = next((m for m in infos if m.is_best), None)
    return ModelsResponse(best_model=best_info, models=infos, count=len(infos))


@router.get("/metrics", response_model=MetricsResponse, tags=["monitoring"],
            summary="Service metrics (counters, latency, uptime)")
def metrics(request: Request,
            mon: Monitor = Depends(monitor)) -> MetricsResponse:
    svc = request.app.state.model_service
    return MetricsResponse(**mon.snapshot(
        model_version=svc.model_version if svc.is_loaded else None))


# ── prediction ────────────────────────────────────────────────────────────────
@router.post("/predict", response_model=PredictionResult, tags=["prediction"],
             summary="Predict financial-distress risk for one instance")
def predict(body: PredictRequest,
            pipe: InferencePipeline = Depends(inference_pipeline),
            mon: Monitor = Depends(monitor)) -> PredictionResult:
    results, inference_ms = pipe.predict([body.features], ids=[body.id])
    mon.record_prediction(1, inference_ms)
    return PredictionResult(**results[0])


@router.post("/predict/batch", response_model=BatchPredictionResponse,
             tags=["prediction"],
             summary="Predict financial-distress risk for a batch")
def predict_batch(body: BatchPredictRequest,
                  pipe: InferencePipeline = Depends(inference_pipeline),
                  mon: Monitor = Depends(monitor)) -> BatchPredictionResponse:
    settings = get_settings()
    if len(body.instances) > settings.max_batch_size:
        raise InvalidRequestError(
            f"batch size {len(body.instances)} exceeds maximum "
            f"{settings.max_batch_size}")
    results, inference_ms = pipe.predict(
        [inst.features for inst in body.instances],
        ids=[inst.id for inst in body.instances])
    mon.record_prediction(len(results), inference_ms)
    return BatchPredictionResponse(
        predictions=[PredictionResult(**r) for r in results],
        count=len(results),
        model_version=results[0]["model_version"] if results else "unknown",
        inference_ms=inference_ms,
    )


# ── validation ────────────────────────────────────────────────────────────────
@router.post("/validate", response_model=ValidationResponse,
             tags=["validation"],
             summary="Validate a payload against the model's feature schema")
def validate(body: BatchPredictRequest,
             pipe: InferencePipeline = Depends(inference_pipeline)
             ) -> ValidationResponse:
    report = pipe.validate_instances([i.features for i in body.instances])
    return ValidationResponse(valid=report["valid"],
                              n_instances=len(body.instances),
                              errors=report["errors"],
                              warnings=report["warnings"])
