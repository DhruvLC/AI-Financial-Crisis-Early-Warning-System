"""Pydantic request/response schemas for the API (OpenAPI-documented)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel as _PydanticBase, Field


class BaseModel(_PydanticBase):
    """Project base schema — allows ``model_version`` field names."""
    model_config = {"protected_namespaces": ()}


class PredictRequest(BaseModel):
    """A single prediction request: engineered feature name -> value."""
    features: Dict[str, float] = Field(
        ..., description="Mapping of engineered feature name to value; must "
                         "match the model's feature schema exactly.")
    id: Optional[str] = Field(None, description="Optional caller-supplied id "
                                                "echoed back in the response.")

    model_config = {"protected_namespaces": (), "json_schema_extra": {"example": {
        "id": "company-001",
        "features": {"Debt ratio %": 0.42, "Borrowing dependency": 0.37},
    }}}


class BatchPredictRequest(BaseModel):
    """A batch of prediction requests."""
    instances: List[PredictRequest] = Field(..., min_length=1)


class PredictionResult(BaseModel):
    """Single prediction output."""
    id: Optional[str] = None
    prediction: int = Field(..., description="1 = crisis/bankruptcy risk flagged")
    probability: float = Field(..., description="Predicted probability of the "
                                                "positive (distress) class")
    risk_score: float = Field(..., description="Probability mapped to 0-100")
    risk_level: str = Field(..., description="Low | Medium | High")
    confidence_score: float = Field(..., description="Distance of the "
                                    "probability from the decision threshold, "
                                    "normalized to 0-1")
    threshold: float
    model_version: str
    algorithm: str
    prediction_timestamp: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    count: int
    model_version: str
    inference_ms: float


class ValidationIssue(BaseModel):
    field: str
    issue: str


class ValidationResponse(BaseModel):
    valid: bool
    n_instances: int
    errors: List[ValidationIssue] = []
    warnings: List[ValidationIssue] = []


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    algorithm: Optional[str] = None
    feature_store_version: Optional[str] = None
    uptime_seconds: float
    timestamp: str


class VersionResponse(BaseModel):
    api_version: str
    app_version: str
    model_version: Optional[str] = None
    algorithm: Optional[str] = None
    dataset_version: Optional[str] = None


class ModelInfo(BaseModel):
    model_version: str
    algorithm: str
    artefact: str
    metrics: Dict[str, Any] = {}
    training_timestamp: Optional[str] = None
    is_best: bool = False


class ModelsResponse(BaseModel):
    best_model: Optional[ModelInfo] = None
    models: List[ModelInfo]
    count: int


class MetricsResponse(BaseModel):
    uptime_seconds: float
    request_count: int
    prediction_count: int
    error_count: int
    avg_latency_ms: float
    avg_inference_ms: float
    active_model_version: Optional[str] = None
    started_at: str


class RootResponse(BaseModel):
    name: str
    api_version: str
    docs: str
    redoc: str
    health: str
