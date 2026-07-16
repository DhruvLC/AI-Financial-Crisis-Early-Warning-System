"""Unit tests for the Machine Learning module (``src/pipeline/ml``).

Run with the project's venv, no extra deps required::

    .venv/bin/python -m unittest tests.test_machine_learning -v
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

# Make ``src`` importable exactly like the run_*.py entry points do.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pipeline.feature_engineering.store import FeatureStore  # noqa: E402
from pipeline.ml import (  # noqa: E402
    METRIC_NAMES, MLDataLoader, MLError, MLPipeline, MODEL_REGISTRY,
    ModelEvaluator, ModelExplainer, ModelRegistry, ThresholdOptimizer,
    build_model, make_cv_splitter,
)
from pipeline.ml.base import TrainedModel  # noqa: E402
from pipeline.ml.tuning import HyperparameterTuner  # noqa: E402

TARGET = "Bankrupt?"


def make_df(n: int = 240, seed: int = 42) -> pd.DataFrame:
    """Synthetic frame with signal + noise — like the engineered data."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    return pd.DataFrame({
        "f1": y * 2.0 + rng.normal(0, 1, n),
        "f2": -y * 1.5 + rng.normal(0, 1, n),
        "f3": rng.normal(0, 1, n),
        "f4": rng.normal(0, 1, n),
        TARGET: y,
    })


def make_splits() -> dict[str, pd.DataFrame]:
    return {"train": make_df(300, 1), "val": make_df(120, 2),
            "test": make_df(120, 3)}


class TmpDirTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


# ── models ──────────────────────────────────────────────────────────────────
class TestModels(unittest.TestCase):
    def setUp(self) -> None:
        df = make_df()
        self.X, self.y = df.drop(columns=[TARGET]), df[TARGET]

    def test_all_available_algorithms_fit_and_predict(self) -> None:
        for name, cls in MODEL_REGISTRY.items():
            ok, _ = cls.available()
            if not ok:
                continue
            with self.subTest(model=name):
                params = ({"n_estimators": 20} if "n_estimators"
                          in cls().default_params() else
                          {"iterations": 20} if name == "catboost" else
                          {"max_iter": 50} if name == "mlp" else None)
                model = build_model(name, params).fit(self.X, self.y)
                proba = model.predict_proba(self.X)
                self.assertEqual(len(proba), len(self.X))
                self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
                preds = model.predict(self.X)
                self.assertTrue(set(np.unique(preds)) <= {0, 1})

    def test_unsupported_model_raises(self) -> None:
        with self.assertRaises(MLError):
            build_model("quantum_forest")

    def test_empty_dataset_raises(self) -> None:
        with self.assertRaises(MLError):
            build_model("logistic_regression").fit(
                self.X.iloc[:0], self.y.iloc[:0])

    def test_missing_features_at_predict_raises(self) -> None:
        model = build_model("logistic_regression").fit(self.X, self.y)
        with self.assertRaises(MLError):
            model.predict(self.X.drop(columns=["f1"]))

    def test_predict_before_fit_raises(self) -> None:
        with self.assertRaises(MLError):
            build_model("decision_tree").predict(self.X)

    def test_params_merge_defaults_with_overrides(self) -> None:
        model = build_model("random_forest", {"n_estimators": 7})
        self.assertEqual(model.params["n_estimators"], 7)
        self.assertEqual(model.params["class_weight"], "balanced")

    def test_native_importance(self) -> None:
        model = build_model("decision_tree").fit(self.X, self.y)
        imp = model.native_importance()
        self.assertEqual(set(imp["feature"]), set(self.X.columns))


# ── evaluation & thresholds ─────────────────────────────────────────────────
class TestEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.y = pd.Series((rng.random(200) < 0.3).astype(int))
        self.proba = np.clip(self.y * 0.6 + rng.random(200) * 0.4, 0, 1)

    def test_all_metrics_computed(self) -> None:
        ev = ModelEvaluator().evaluate("m", "test", self.y, self.proba)
        for metric in METRIC_NAMES:
            self.assertIn(metric, ev.metrics)
            self.assertFalse(np.isnan(ev.metrics[metric]))
        self.assertEqual(np.sum(ev.confusion), len(self.y))

    def test_empty_split_raises(self) -> None:
        with self.assertRaises(MLError):
            ModelEvaluator().evaluate("m", "test", pd.Series(dtype=int),
                                      np.array([]))

    def test_threshold_youden_and_f1(self) -> None:
        for method in ("youden", "max_f1"):
            t, used = ThresholdOptimizer(
                {"method": method}).optimize(self.y, self.proba)
            self.assertEqual(used, method if method != "max_f1" else "max_f1")
            self.assertTrue(0.0 <= t <= 1.0)

    def test_threshold_custom_and_disabled(self) -> None:
        t, used = ThresholdOptimizer(
            {"method": "custom", "custom_threshold": 0.42}
        ).optimize(self.y, self.proba)
        self.assertEqual((t, used), (0.42, "custom"))
        t, used = ThresholdOptimizer({"enabled": False}).optimize(
            self.y, self.proba)
        self.assertEqual((t, used), (0.5, "default"))

    def test_threshold_unsupported_method_raises(self) -> None:
        with self.assertRaises(MLError):
            ThresholdOptimizer({"method": "magic"}).optimize(
                self.y, self.proba)


# ── tuning & cross-validation ───────────────────────────────────────────────
class TestTuning(unittest.TestCase):
    def setUp(self) -> None:
        df = make_df(200)
        self.X, self.y = df.drop(columns=[TARGET]), df[TARGET]

    def test_cv_splitters(self) -> None:
        self.assertEqual(make_cv_splitter(
            {"strategy": "stratified_kfold", "n_splits": 3}).get_n_splits(), 3)
        self.assertEqual(make_cv_splitter(
            {"strategy": "time_series", "n_splits": 4}).get_n_splits(), 4)
        with self.assertRaises(MLError):
            make_cv_splitter({"strategy": "bogus"})

    def test_random_search_updates_params(self) -> None:
        tuner = HyperparameterTuner(
            {"enabled": True, "method": "random", "n_iter": 2,
             "search_spaces": {"decision_tree": {"max_depth": [2, 4]}}},
            {"n_splits": 3})
        model = build_model("decision_tree")
        result = tuner.tune(model, self.X, self.y)
        self.assertIn(model.params["max_depth"], (2, 4))
        self.assertIn("best_score", result)

    def test_grid_search(self) -> None:
        tuner = HyperparameterTuner(
            {"enabled": True, "method": "grid",
             "search_spaces": {"logistic_regression": {"C": [0.1, 1.0]}}},
            {"n_splits": 3})
        result = tuner.tune(build_model("logistic_regression"),
                            self.X, self.y)
        self.assertEqual(result["method"], "grid")

    def test_disabled_tuning_is_noop(self) -> None:
        self.assertEqual(HyperparameterTuner({"enabled": False}, {}).tune(
            build_model("decision_tree"), self.X, self.y), {})

    def test_cross_validate(self) -> None:
        cv = HyperparameterTuner({}, {"n_splits": 3}).cross_validate(
            build_model("logistic_regression"), self.X, self.y)
        self.assertEqual(len(cv["scores"]), 3)
        self.assertTrue(0.0 <= cv["mean"] <= 1.0)


# ── data loading ────────────────────────────────────────────────────────────
class TestDataLoader(TmpDirTest):
    def _store(self, splits=None, **meta) -> str:
        store = FeatureStore(self.tmp)
        splits = splits or make_splits()
        features = [c for c in splits["train"].columns if c != TARGET]
        store.save(splits, {"target_col": TARGET, "features": features,
                            **meta})
        return self.tmp

    def test_load_verifies_and_returns_dataset(self) -> None:
        ds = MLDataLoader(self._store(), TARGET).load()
        self.assertEqual(ds.version, "v001")
        self.assertEqual(len(ds.features), 4)
        self.assertTrue(all(ds.checks.values()))
        X, y = ds.xy("train")
        self.assertNotIn(TARGET, X.columns)
        self.assertEqual(len(X), len(y))

    def test_empty_store_raises(self) -> None:
        with self.assertRaises(MLError):
            MLDataLoader(self.tmp, TARGET).load()

    def test_missing_target_raises(self) -> None:
        splits = make_splits()
        splits["val"] = splits["val"].drop(columns=[TARGET])
        FeatureStore(self.tmp).save(splits, {"target_col": TARGET})
        with self.assertRaises(MLError):
            MLDataLoader(self.tmp, TARGET).load()

    def test_schema_mismatch_raises(self) -> None:
        splits = make_splits()
        splits["test"] = splits["test"].drop(columns=["f4"])
        FeatureStore(self.tmp).save(splits, {"target_col": TARGET})
        with self.assertRaises(MLError):
            MLDataLoader(self.tmp, TARGET).load()

    def test_nan_raises(self) -> None:
        splits = make_splits()
        splits["train"].loc[0, "f1"] = np.nan
        FeatureStore(self.tmp).save(splits, {"target_col": TARGET})
        with self.assertRaises(MLError):
            MLDataLoader(self.tmp, TARGET).load()

    def test_hash_tamper_detected(self) -> None:
        root = self._store()
        # tamper with the stored parquet after registration
        path = os.path.join(root, "v001", "train.parquet")
        df = pd.read_parquet(path)
        df.loc[0, "f1"] += 99.0
        df.to_parquet(path, index=False)
        with self.assertRaises(MLError):
            MLDataLoader(root, TARGET).load()


# ── registry ────────────────────────────────────────────────────────────────
class TestRegistry(TmpDirTest):
    def _trained(self) -> TrainedModel:
        df = make_df()
        model = build_model("decision_tree").fit(
            df.drop(columns=[TARGET]), df[TARGET])
        t = TrainedModel(name="decision_tree", model=model,
                         hyperparameters=model.get_params())
        t.evaluations["test"] = ModelEvaluator().evaluate(
            "decision_tree", "test", df[TARGET],
            model.predict_proba(df.drop(columns=[TARGET])))
        return t

    def test_register_and_load(self) -> None:
        reg = ModelRegistry(self.tmp)
        entry = reg.register(self._trained(), ["f1", "f2", "f3", "f4"],
                             "v001", TARGET)
        self.assertEqual(entry["model_version"], "v001")
        self.assertEqual(entry["dataset_version"], "v001")
        self.assertTrue(os.path.exists(entry["artefact"]))
        loaded = reg.load_model("decision_tree")
        self.assertEqual(len(loaded.predict(make_df(10).drop(
            columns=[TARGET]))), 10)

    def test_register_best(self) -> None:
        reg = ModelRegistry(self.tmp)
        t = self._trained()
        entry = reg.register(t, ["f1", "f2", "f3", "f4"], "v001", TARGET)
        best_path = reg.register_best(t, entry)
        self.assertTrue(os.path.exists(best_path))
        self.assertEqual(reg.best()["algorithm"], "decision_tree")
        self.assertIsNotNone(reg.load_best())
        with open(os.path.join(self.tmp, "best_model.json")) as f:
            self.assertEqual(json.load(f)["algorithm"], "decision_tree")

    def test_versions_increment(self) -> None:
        reg = ModelRegistry(self.tmp)
        reg.register(self._trained(), ["f1"], "v001", TARGET)
        entry2 = reg.register(self._trained(), ["f1"], "v001", TARGET)
        self.assertEqual(entry2["model_version"], "v002")

    def test_load_missing_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            ModelRegistry(self.tmp).load_model("nope")


# ── explainability ──────────────────────────────────────────────────────────
class TestExplainability(unittest.TestCase):
    def test_permutation_and_native(self) -> None:
        df = make_df()
        X, y = df.drop(columns=[TARGET]), df[TARGET]
        model = build_model("random_forest", {"n_estimators": 30}).fit(X, y)
        out = ModelExplainer({"shap": False,
                              "permutation_repeats": 2}).explain(model, X, y)
        self.assertIsNotNone(out["native"])
        self.assertIsNotNone(out["permutation"])
        self.assertIsNone(out["shap"])


# ── end-to-end pipeline ─────────────────────────────────────────────────────
class TestPipeline(TmpDirTest):
    def _cfg(self) -> dict:
        store_dir = os.path.join(self.tmp, "features")
        FeatureStore(store_dir).save(
            make_splits(),
            {"target_col": TARGET,
             "features": ["f1", "f2", "f3", "f4"]})
        return {
            "data": {"target_col": TARGET},
            "feature_engineering": {"store": {"dir": store_dir}},
            "ml": {
                "algorithms": ["logistic_regression", "decision_tree",
                               "naive_bayes"],
                "cross_validation": {"enabled": True, "n_splits": 3},
                "tuning": {"enabled": False},
                "threshold_optimization": {"method": "youden"},
                "explainability": {"shap": False,
                                   "permutation_repeats": 2},
                "output": {
                    "models_dir": os.path.join(self.tmp, "models"),
                    "reports_dir": os.path.join(self.tmp, "reports"),
                    "figures_dir": os.path.join(self.tmp, "figures")},
            },
        }

    def test_full_run(self) -> None:
        result = MLPipeline(self._cfg()).run()
        self.assertEqual(len(result.trained), 3)
        self.assertFalse(any(t.failed for t in result.trained))
        # leaderboard ranked by test roc_auc
        board = result.leaderboard
        self.assertEqual(list(board["roc_auc"]),
                         sorted(board["roc_auc"], reverse=True))
        self.assertEqual(result.best.name, board.iloc[0]["model"])
        # registry + best model persisted
        self.assertEqual(len(result.registry_entries), 3)
        self.assertTrue(os.path.exists(
            os.path.join(self.tmp, "models", "best_model.joblib")))
        self.assertTrue(os.path.exists(
            os.path.join(self.tmp, "models", "registry.json")))
        # reports + figures written
        self.assertTrue(any(p.endswith("leaderboard.csv")
                            for p in result.reports))
        self.assertTrue(any("roc_" in p for p in result.figures))
        # every model evaluated on both splits with full metric suite
        for t in result.trained:
            for split in ("val", "test"):
                self.assertEqual(set(t.evaluations[split].metrics),
                                 set(METRIC_NAMES))

    def test_failing_model_is_isolated(self) -> None:
        cfg = self._cfg()
        cfg["ml"]["algorithms"] = ["logistic_regression", "mlp"]
        cfg["ml"]["model_params"] = {"mlp": {"hidden_layer_sizes": "bad"}}
        result = MLPipeline(cfg).run()
        names = {t.name: t.failed for t in result.trained}
        self.assertFalse(names["logistic_regression"])
        self.assertTrue(names["mlp"])
        self.assertEqual(result.best.name, "logistic_regression")

    def test_all_models_fail_raises(self) -> None:
        cfg = self._cfg()
        cfg["ml"]["algorithms"] = ["mlp"]
        cfg["ml"]["model_params"] = {"mlp": {"hidden_layer_sizes": "bad"}}
        with self.assertRaises(MLError):
            MLPipeline(cfg).run()


if __name__ == "__main__":
    unittest.main()
