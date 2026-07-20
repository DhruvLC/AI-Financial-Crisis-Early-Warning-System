"""In-process monitoring — request/prediction counters, latency, uptime."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone


class Monitor:
    """Thread-safe rolling counters for the /metrics endpoint."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._start = time.monotonic()
        self.request_count = 0
        self.prediction_count = 0
        self.error_count = 0
        self._latency_total_ms = 0.0
        self._inference_total_ms = 0.0
        self._inference_calls = 0

    # ── recording ─────────────────────────────────────────────────────────────
    def record_request(self, latency_ms: float, is_error: bool) -> None:
        with self._lock:
            self.request_count += 1
            self._latency_total_ms += latency_ms
            if is_error:
                self.error_count += 1

    def record_prediction(self, n: int, inference_ms: float) -> None:
        with self._lock:
            self.prediction_count += n
            self._inference_total_ms += inference_ms
            self._inference_calls += 1

    # ── reading ───────────────────────────────────────────────────────────────
    @property
    def uptime_seconds(self) -> float:
        return round(time.monotonic() - self._start, 3)

    def snapshot(self, model_version: str | None = None) -> dict:
        with self._lock:
            avg_latency = (self._latency_total_ms / self.request_count
                           if self.request_count else 0.0)
            avg_inference = (self._inference_total_ms / self._inference_calls
                             if self._inference_calls else 0.0)
            return {
                "uptime_seconds": self.uptime_seconds,
                "request_count": self.request_count,
                "prediction_count": self.prediction_count,
                "error_count": self.error_count,
                "avg_latency_ms": round(avg_latency, 3),
                "avg_inference_ms": round(avg_inference, 3),
                "active_model_version": model_version,
                "started_at": self.started_at,
            }
