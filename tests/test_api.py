"""Unit tests for the Backend Deployment module (``src/api``).

Run with the project's venv::

    .venv/bin/python -m pytest tests/test_api.py -v
    .venv/bin/python -m unittest tests.test_api -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

# Make ``src`` importable exactly like the run_*.py entry points do.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from fastapi.testclient import TestClient  # noqa: E402

from api import API_VERSION  # noqa: E402
from api.app import create_app  # noqa: E402
from api.config import get_settings  # noqa: E402
from api.exceptions import MissingArtifactError, ModelNotLoadedError  # noqa: E402
from api.model_loader import ModelService  # noqa: E402
from api.monitoring import Monitor  # noqa: E402
from api.predict import InferencePipeline, _band  # noqa: E402

PREFIX = f"/api/{API_VERSION}"


def _client() -> TestClient:
    return TestClient(create_app())


def _payload(app, value: float = 0.1, id_: str | None = "t-1") -> dict:
    feats = app.state.model_service.features
    return {"id": id_, "features": {f: value for f in feats}}


class TestModelLoading(unittest.TestCase):
    def test_loads_best_model(self):
        svc = ModelService(get_settings())
        svc.load()
        self.assertTrue(svc.is_loaded)
        self.assertTrue(svc.model_version.startswith("v"))
        self.assertGreater(len(svc.features), 0)
        self.assertGreater(svc.threshold, 0.0)
        self.assertLess(svc.threshold, 1.0)

    def test_missing_artifact_raises(self):
        settings = get_settings()
        with tempfile.TemporaryDirectory() as tmp:
            broken = type(settings)(
                models_dir=os.path.join(tmp, "models"),
                feature_store_root=settings.feature_store_root)
            svc = ModelService(broken)
            with self.assertRaises(MissingArtifactError):
                svc.load()

    def test_require_loaded(self):
        svc = ModelService(get_settings())
        with self.assertRaises(ModelNotLoadedError):
            svc.require_loaded()


class TestRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = _client()
        cls.client.__enter__()  # trigger lifespan (model load)
        cls.app = cls.client.app

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    # ── root / docs ───────────────────────────────────────────────────────────
    def test_root(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["api_version"], API_VERSION)

    def test_docs_and_redoc(self):
        self.assertEqual(self.client.get("/docs").status_code, 200)
        self.assertEqual(self.client.get("/redoc").status_code, 200)
        self.assertEqual(self.client.get("/openapi.json").status_code, 200)

    # ── health / version / models / metrics ───────────────────────────────────
    def test_health(self):
        for path in ("/health", f"{PREFIX}/health"):
            r = self.client.get(path)
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertEqual(body["status"], "ok")
            self.assertTrue(body["model_loaded"])
            self.assertGreaterEqual(body["uptime_seconds"], 0)

    def test_version(self):
        body = self.client.get(f"{PREFIX}/version").json()
        self.assertEqual(body["api_version"], API_VERSION)
        self.assertIsNotNone(body["model_version"])

    def test_models(self):
        body = self.client.get(f"{PREFIX}/models").json()
        self.assertGreater(body["count"], 0)
        self.assertIsNotNone(body["best_model"])
        self.assertTrue(body["best_model"]["is_best"])

    def test_metrics(self):
        body = self.client.get(f"{PREFIX}/metrics").json()
        for key in ("uptime_seconds", "request_count", "prediction_count",
                    "error_count", "avg_latency_ms", "active_model_version"):
            self.assertIn(key, body)

    def test_request_id_headers(self):
        r = self.client.get(f"{PREFIX}/health")
        self.assertIn("X-Request-ID", r.headers)
        self.assertIn("X-Response-Time-Ms", r.headers)

    # ── predict ───────────────────────────────────────────────────────────────
    def test_predict(self):
        r = self.client.post(f"{PREFIX}/predict", json=_payload(self.app))
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn(body["prediction"], (0, 1))
        self.assertGreaterEqual(body["probability"], 0.0)
        self.assertLessEqual(body["probability"], 1.0)
        self.assertGreaterEqual(body["risk_score"], 0.0)
        self.assertLessEqual(body["risk_score"], 100.0)
        self.assertIn(body["risk_level"], ("Low", "Medium", "High"))
        self.assertGreaterEqual(body["confidence_score"], 0.0)
        self.assertLessEqual(body["confidence_score"], 1.0)
        self.assertEqual(body["id"], "t-1")
        self.assertIn("prediction_timestamp", body)
        self.assertIsNotNone(body["model_version"])

    def test_predict_batch(self):
        payload = _payload(self.app)
        r = self.client.post(f"{PREFIX}/predict/batch",
                             json={"instances": [payload, payload, payload]})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["count"], 3)
        self.assertEqual(len(body["predictions"]), 3)
        self.assertGreater(body["inference_ms"], 0)

    def test_predict_missing_features_422(self):
        r = self.client.post(f"{PREFIX}/predict",
                             json={"features": {"not a feature": 1.0}})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["error"]["type"], "invalid_request")

    def test_predict_malformed_body_422(self):
        r = self.client.post(f"{PREFIX}/predict", json={"nope": True})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["error"]["type"], "invalid_request")

    def test_predict_nan_rejected(self):
        payload = _payload(self.app)
        first = next(iter(payload["features"]))
        payload["features"][first] = None
        r = self.client.post(f"{PREFIX}/predict", json=payload)
        self.assertEqual(r.status_code, 422)

    # ── validate ──────────────────────────────────────────────────────────────
    def test_validate_ok(self):
        r = self.client.post(f"{PREFIX}/validate",
                             json={"instances": [_payload(self.app)]})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["valid"])

    def test_validate_reports_missing_and_unknown(self):
        r = self.client.post(
            f"{PREFIX}/validate",
            json={"instances": [{"features": {"bogus": 1.0}}]})
        body = r.json()
        self.assertFalse(body["valid"])
        n_features = len(self.app.state.model_service.features)
        self.assertEqual(len(body["errors"]), n_features)  # all missing
        self.assertEqual(len(body["warnings"]), 1)         # unknown 'bogus'


class TestErrorHandlingWithoutModel(unittest.TestCase):
    """Prediction endpoints return 503 when the model failed to load."""

    def test_degraded_service(self):
        client = _client()
        with client:
            client.app.state.model_service.model = None  # simulate load failure
            body = client.get(f"{PREFIX}/health").json()
            self.assertEqual(body["status"], "degraded")
            r = client.post(f"{PREFIX}/predict",
                            json={"features": {"x": 1.0}})
            self.assertEqual(r.status_code, 503)
            self.assertEqual(r.json()["error"]["type"], "model_not_loaded")


class TestInferencePipelineUnit(unittest.TestCase):
    def test_band(self):
        self.assertEqual(_band(10, 100), "Low")
        self.assertEqual(_band(50, 100), "Medium")
        self.assertEqual(_band(90, 100), "High")

    def test_validate_instances_types(self):
        svc = ModelService(get_settings())
        svc.load()
        pipe = InferencePipeline(svc)
        feats = {f: 0.0 for f in svc.features}
        feats[svc.features[0]] = "not-a-number"
        report = pipe.validate_instances([feats])
        self.assertFalse(report["valid"])
        self.assertTrue(any("expected number" in e["issue"]
                            for e in report["errors"]))


class TestMonitor(unittest.TestCase):
    def test_counters(self):
        mon = Monitor()
        mon.record_request(10.0, is_error=False)
        mon.record_request(30.0, is_error=True)
        mon.record_prediction(5, 12.0)
        snap = mon.snapshot("v004")
        self.assertEqual(snap["request_count"], 2)
        self.assertEqual(snap["error_count"], 1)
        self.assertEqual(snap["prediction_count"], 5)
        self.assertAlmostEqual(snap["avg_latency_ms"], 20.0)
        self.assertEqual(snap["active_model_version"], "v004")


if __name__ == "__main__":
    unittest.main(verbosity=2)
