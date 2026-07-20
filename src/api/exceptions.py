"""Standardized API exceptions and global handlers.

Every error leaves the API as the same JSON envelope::

    {"error": {"type": "...", "message": "...", "detail": ..., "request_id": ...}}
"""
from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ingestion.logging_config import get_logger

log = get_logger("api.exceptions")


class APIError(Exception):
    """Base class for structured API errors."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type = "internal_error"

    def __init__(self, message: str, detail=None):
        super().__init__(message)
        self.message = message
        self.detail = detail


class ModelNotLoadedError(APIError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_type = "model_not_loaded"


class ModelLoadingError(APIError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type = "model_loading_failure"


class RegistryError(APIError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type = "registry_error"


class MissingArtifactError(APIError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type = "missing_artifact"


class ConfigurationError(APIError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type = "configuration_error"


class InvalidRequestError(APIError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_type = "invalid_request"


class PredictionError(APIError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type = "prediction_failure"


def _envelope(request: Request, error_type: str, message: str,
              detail=None) -> dict:
    return {"error": {
        "type": error_type,
        "message": message,
        "detail": detail,
        "request_id": getattr(request.state, "request_id", None),
        "path": str(request.url.path),
    }}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the global handlers to ``app``."""

    @app.exception_handler(APIError)
    async def _api_error(request: Request, exc: APIError):
        log.error("%s: %s", exc.error_type, exc.message)
        return JSONResponse(status_code=exc.status_code,
                            content=_envelope(request, exc.error_type,
                                              exc.message, exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_envelope(request, "invalid_request",
                              "request validation failed", exc.errors()))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code,
                            content=_envelope(request, "http_error",
                                              str(exc.detail)))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope(request, "internal_error",
                              "an unexpected error occurred"))
