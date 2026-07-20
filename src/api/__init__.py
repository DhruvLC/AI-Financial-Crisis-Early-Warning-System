"""Backend Deployment module — production FastAPI serving layer.

Exposes the trained best model from the existing ModelRegistry behind a
versioned REST API (``/api/v1``) with validation, monitoring, structured
errors, and reuse of the project's logging + configuration frameworks.
"""
from __future__ import annotations

API_VERSION = "v1"

from .app import create_app  # noqa: E402
from .config import APISettings, get_settings  # noqa: E402
from .model_loader import ModelService  # noqa: E402
from .predict import InferencePipeline  # noqa: E402

__all__ = ["API_VERSION", "create_app", "APISettings", "get_settings",
           "ModelService", "InferencePipeline"]
