"""FastAPI application factory + uvicorn entry point.

Run locally::

    .venv/bin/python -m uvicorn api.app:app --app-dir src --port 8000

or ``python src/run_api.py``.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from ingestion.logging_config import configure_logging, get_logger

from . import API_VERSION
from .config import APISettings, get_settings
from .exceptions import register_exception_handlers
from .health import router as health_router
from .middleware import RequestContextMiddleware
from .model_loader import ModelService
from .monitoring import Monitor
from .routes import router as api_router
from .schemas import RootResponse

log = get_logger("api.app")


def create_app(settings: APISettings | None = None) -> FastAPI:
    """Build the production FastAPI app (factory — testable, configurable)."""
    settings = settings or get_settings()
    configure_logging(level=settings.log_level,
                      logfile=settings.log_file or None)

    monitor = Monitor()
    model_service = ModelService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: load the best model from the registry (never retrain).
        log.info("starting %s (api %s)", settings.title, API_VERSION)
        try:
            model_service.load()
        except Exception as exc:  # noqa: BLE001 - keep serving /health
            log.error("model loading failed at startup: %s", exc)
        yield
        # Shutdown.
        log.info("shutting down; served %d requests, %d predictions",
                 monitor.request_count, monitor.prediction_count)

    app = FastAPI(
        title=settings.title,
        description=settings.description,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings
    app.state.monitor = monitor
    app.state.model_service = model_service

    app.add_middleware(RequestContextMiddleware, monitor=monitor)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)

    prefix = f"/api/{API_VERSION}"
    app.include_router(health_router, prefix=prefix)
    app.include_router(api_router, prefix=prefix)
    # Unversioned aliases for probes/load-balancers.
    app.include_router(health_router)

    @app.get("/", response_model=RootResponse, tags=["root"],
             summary="Service index")
    def root(request: Request) -> RootResponse:
        return RootResponse(name=settings.title, api_version=API_VERSION,
                            docs="/docs", redoc="/redoc",
                            health=f"{prefix}/health")

    return app


app = create_app()
