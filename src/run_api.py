"""Entry point — run the Backend Deployment FastAPI service.

Usage::

    .venv/bin/python src/run_api.py            # honours EWS_HOST / EWS_PORT
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn  # noqa: E402

from api.config import get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()
    uvicorn.run("api.app:app", host=settings.host, port=settings.port,
                app_dir=os.path.dirname(os.path.abspath(__file__)),
                log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
