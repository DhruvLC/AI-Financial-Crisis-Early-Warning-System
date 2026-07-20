"""FastAPI dependency-injection providers (wired to ``app.state``)."""
from __future__ import annotations

from fastapi import Request

from .config import APISettings, get_settings
from .model_loader import ModelService
from .monitoring import Monitor
from .predict import InferencePipeline


def settings_dep() -> APISettings:
    return get_settings()


def model_service(request: Request) -> ModelService:
    svc: ModelService = request.app.state.model_service
    svc.require_loaded()
    return svc


def inference_pipeline(request: Request) -> InferencePipeline:
    return InferencePipeline(model_service(request))


def monitor(request: Request) -> Monitor:
    return request.app.state.monitor
