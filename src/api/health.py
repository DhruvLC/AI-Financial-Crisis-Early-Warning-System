"""Health & version endpoints."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

from . import API_VERSION
from .config import get_settings
from .schemas import HealthResponse, VersionResponse

router = APIRouter(tags=["monitoring"])


@router.get("/health", response_model=HealthResponse,
            summary="Liveness/readiness probe")
def health(request: Request) -> HealthResponse:
    svc = request.app.state.model_service
    monitor = request.app.state.monitor
    loaded = svc.is_loaded
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=svc.model_version if loaded else None,
        algorithm=svc.algorithm if loaded else None,
        feature_store_version=svc.feature_metadata.get("version"),
        uptime_seconds=monitor.uptime_seconds,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/version", response_model=VersionResponse,
            summary="API + model version info")
def version(request: Request) -> VersionResponse:
    svc = request.app.state.model_service
    settings = get_settings()
    return VersionResponse(
        api_version=API_VERSION,
        app_version=settings.app_version,
        model_version=svc.model_version if svc.is_loaded else None,
        algorithm=svc.algorithm if svc.is_loaded else None,
        dataset_version=svc.dataset_version if svc.is_loaded else None,
    )
