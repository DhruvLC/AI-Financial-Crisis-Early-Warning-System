"""API configuration — environment variables layered over configs/config.yaml.

Every setting has a sane default, can be overridden by an ``EWS_``-prefixed
environment variable, and falls back to the project's central YAML config for
paths shared with earlier phases (models dir, feature store root, risk scale).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _env(name: str, default: str) -> str:
    return os.environ.get(f"EWS_{name}", default)


@dataclass(frozen=True)
class APISettings:
    """Immutable runtime settings for the API process."""
    title: str = "Financial Crisis Early Warning System API"
    description: str = ("Production inference API for corporate bankruptcy / "
                        "financial-distress risk (0-100 risk score).")
    api_version: str = "v1"
    app_version: str = "1.0.0"

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_file: str = ""
    cors_origins: tuple = ("*",)

    project_root: str = _PROJECT_ROOT
    config_path: str = os.path.join(_PROJECT_ROOT, "configs", "config.yaml")
    models_dir: str = os.path.join(_PROJECT_ROOT, "models")
    feature_store_root: str = os.path.join(_PROJECT_ROOT, "data", "features")

    risk_score_scale: int = 100
    max_batch_size: int = 1000

    yaml_config: dict = field(default_factory=dict, hash=False, compare=False)


@lru_cache(maxsize=1)
def get_settings() -> APISettings:
    """Build settings once per process: YAML config + env-var overrides."""
    root = _env("PROJECT_ROOT", _PROJECT_ROOT)
    config_path = _env("CONFIG_PATH", os.path.join(root, "configs", "config.yaml"))
    yaml_cfg: dict = {}
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}

    output_cfg = yaml_cfg.get("output", {})
    return APISettings(
        host=_env("HOST", "0.0.0.0"),
        port=int(_env("PORT", "8000")),
        log_level=_env("LOG_LEVEL", "INFO"),
        log_file=_env("LOG_FILE", ""),
        cors_origins=tuple(o.strip() for o in _env("CORS_ORIGINS", "*").split(",")),
        project_root=root,
        config_path=config_path,
        models_dir=_env("MODELS_DIR",
                        os.path.join(root, output_cfg.get("models_dir", "models"))
                        if not os.path.isabs(output_cfg.get("models_dir", "models"))
                        else output_cfg.get("models_dir", "models")),
        feature_store_root=_env("FEATURE_STORE_ROOT",
                                os.path.join(root, "data", "features")),
        risk_score_scale=int(_env("RISK_SCORE_SCALE",
                                  str(output_cfg.get("risk_score_scale", 100)))),
        max_batch_size=int(_env("MAX_BATCH_SIZE", "1000")),
        yaml_config=yaml_cfg,
    )
