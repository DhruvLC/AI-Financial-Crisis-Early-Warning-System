"""Unit tests for the Transformer Models module (``src/pipeline/transformers``).

Run with the project's venv (needs torch)::

    .venv/bin/python -m unittest tests.test_transformers -v
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
import torch

# Make ``src`` importable exactly like the run_*.py entry points do.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pipeline.transformers import (  # noqa: E402
    TRANSFORMER_REGISTRY, TransformerDataLoader, TransformerError,
    TransformerModelRegistry, TransformerPipeline, TransformerPredictor,
    TransformerReport, TransformerTrainer, attention_entropy,
    build_transformer, mean_attention_by_feature, predict_proba,
)
from pipeline.transformers.base import seed_all  # noqa: E402
from pipeline.feature_engineering.store import FeatureStore  # noqa: E402

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


def small_cfg(tmp: str) -> dict:
    """Minimal fast config used by the pipeline tests."""
    return {
        "data": {"target_col": TARGET},
        "feature_engineering": {"store": {"dir": os.path.join(tmp,
                                                              "features")}},
        "transformers": {
            "fail_fast": True,
            "random_state": 42,
            "models": ["ft_transformer"],
            "data": {"batch_size": 32},
            "model_params": {"ft_transformer": {"embed_dim": 8,
                                                "n_heads": 2,
                                                "n_layers": 1,
                                                "ff_dim": 16,
                                                "dropout": 0.0}},
            "training": {"epochs": 3, "device": "cpu", "log_every": 10},
            "early_stopping": {"patience": 5},
            "checkpoint": {"dir": os.path.join(tmp, "models")},
            "explainability": {"permutation": True,
                               "permutation_repeats": 1, "shap": False,
                               "attention": True,
                               "attention_max_samples": 32},
            "comparison": {"ml_leaderboard": None, "dl_leaderboard": None},
            "output": {
                "models_dir": os.path.join(tmp, "models"),
                "reports_dir": os.path.join(tmp, "reports"),
                "figures_dir": os.path.join(tmp, "reports", "figures"),
            },
        },
    }


class TmpDirTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="tf_test_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _store(self) -> str:
        store_dir = os.path.join(self.tmp, "features")
        splits = make_splits()
        FeatureStore(store_dir).save(
            splits, {"target_col": TARGET,
                     "features": [c for c in splits["train"].columns
                                  if c != TARGET]})
        return store_dir


# ── data loading ────────────────────────────────────────────────────────────
class TestDataLoading(TmpDirTest):
    def test_loader_builds_batches_and_pos_weight(self) -> None:
        loader = TransformerDataLoader(self._store(), TARGET,
                                       batch_size=32, seed=7)
        data = loader.load()
        self.assertEqual(data.n_features, 4)
        self.assertEqual(set(data.loaders), {"train", "val", "test"})
        X, y = next(iter(data.loaders["train"]))
        self.assertEqual(X.shape, (32, 4))
        self.assertEqual(X.dtype, torch.float32)
        self.assertGreater(data.pos_weight, 1.0)

    def test_loader_rejects_bad_batch_size(self) -> None:
        with self.assertRaises(TransformerError.__mro__[1]):  # DLError
            TransformerDataLoader(self._store(), TARGET, batch_size=0)

    def test_loader_shuffle_is_reproducible(self) -> None:
        store = self._store()
        first = next(iter(TransformerDataLoader(
            store, TARGET, seed=5).load().loaders["train"]))[0]
        second = next(iter(TransformerDataLoader(
            store, TARGET, seed=5).load().loaders["train"]))[0]
        self.assertTrue(torch.equal(first, second))


# ── models ──────────────────────────────────────────────────────────────────
class TestModels(unittest.TestCase):
    def test_registry_contains_required_models(self) -> None:
        self.assertEqual(set(TRANSFORMER_REGISTRY),
                         {"ft_transformer", "tab_transformer",
                          "tabular_encoder"})

    def test_forward_pass_all_models(self) -> None:
        x = torch.randn(6, 5)
        for name in TRANSFORMER_REGISTRY:
            model = build_transformer(name, 5)
            out = model(x)
            self.assertEqual(out.shape, (6,), name)
            self.assertTrue(torch.isfinite(out).all(), name)

    def test_configurable_hyperparameters(self) -> None:
        model = build_transformer("ft_transformer", 5,
                                  {"embed_dim": 16, "n_heads": 4,
                                   "n_layers": 2, "ff_dim": 32,
                                   "dropout": 0.05})
        arch = model.architecture()
        self.assertEqual(arch["embed_dim"], 16)
        self.assertEqual(arch["n_heads"], 4)
        self.assertEqual(len(model.blocks), 2)

    def test_positional_embedding_toggle(self) -> None:
        with_pos = build_transformer("tabular_encoder", 5,
                                     {"positional_embedding": True})
        without = build_transformer("tabular_encoder", 5,
                                    {"positional_embedding": False})
        self.assertIsNotNone(with_pos.positional)
        self.assertIsNone(without.positional)

    def test_invalid_configs_raise(self) -> None:
        with self.assertRaises(TransformerError):
            build_transformer("nope", 5)
        with self.assertRaises(TransformerError):
            build_transformer("ft_transformer", 0)
        with self.assertRaises(TransformerError):
            build_transformer("ft_transformer", 5,
                              {"embed_dim": 10, "n_heads": 3})
        with self.assertRaises(TransformerError):
            build_transformer("ft_transformer", 5, {"dropout": 1.5})

    def test_attention_capture(self) -> None:
        model = build_transformer("ft_transformer", 4,
                                  {"embed_dim": 8, "n_heads": 2,
                                   "n_layers": 2, "ff_dim": 16})
        model.eval()                        # captured attention is used in eval
        x = torch.randn(3, 4)
        model(x)
        self.assertEqual(model.attention_weights(), {})
        model.collect_attention(True)
        model(x)
        weights = model.attention_weights()
        self.assertEqual(set(weights), {"layer_1", "layer_2"})
        w = weights["layer_1"]
        self.assertEqual(w.shape, (3, 5, 5))          # [CLS] + 4 features
        self.assertTrue(torch.allclose(w.sum(-1),
                                       torch.ones(3, 5), atol=1e-5))
        self.assertEqual(model.token_labels()[0], "[CLS]")
        model.collect_attention(False)
        self.assertEqual(model.attention_weights(), {})


# ── attention math ──────────────────────────────────────────────────────────
class TestAttentionMath(unittest.TestCase):
    def test_entropy_uniform_vs_peaked(self) -> None:
        uniform = np.full((2, 4, 4), 0.25)
        peaked = np.tile(np.eye(4), (2, 1, 1))
        self.assertGreater(attention_entropy(uniform),
                           attention_entropy(peaked))

    def test_mean_attention_by_feature(self) -> None:
        attn = np.array([[[0.1, 0.2, 0.7], [0.1, 0.2, 0.7],
                          [0.1, 0.2, 0.7]]])
        out = mean_attention_by_feature(attn, ["[CLS]", "a", "b"])
        self.assertEqual(list(out), ["b", "a"])       # sorted desc, no CLS
        self.assertAlmostEqual(out["b"], 0.7)


# ── training / evaluation / prediction ──────────────────────────────────────
class TestTrainingAndPipeline(TmpDirTest):
    def _run(self):
        cfg = small_cfg(self.tmp)
        self._store()
        seed_all(42)
        return TransformerPipeline(cfg).run(), cfg

    def test_end_to_end_pipeline(self) -> None:
        result, cfg = self._run()
        t = result.trained[0]
        self.assertFalse(t.failed)
        self.assertEqual(len(t.history.epochs), 3)
        self.assertIn("test", t.evaluations)
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc",
                       "pr_auc", "balanced_accuracy", "mcc"):
            self.assertIn(metric, t.evaluations["test"].metrics)
        self.assertIsNotNone(t.attention)
        self.assertTrue(t.attention.feature_attention)
        self.assertIsNotNone(result.leaderboard)
        self.assertEqual(result.best.name, "ft_transformer")

        # registry artefacts
        models_dir = cfg["transformers"]["output"]["models_dir"]
        for artefact in ("best_model.pt", "last_model.pt", "metrics.json",
                         "history.json", "training_config.json",
                         "feature_metadata.json", "registry.json"):
            self.assertTrue(os.path.exists(os.path.join(models_dir,
                                                        artefact)),
                            artefact)

        # reports
        reports_dir = cfg["transformers"]["output"]["reports_dir"]
        for name in ("transformer_report.json", "transformer_report.md",
                     "transformer_report.html", "leaderboard.csv",
                     "metrics_summary.csv"):
            self.assertTrue(os.path.exists(os.path.join(reports_dir,
                                                        name)), name)
        with open(os.path.join(reports_dir,
                               "transformer_report.json")) as f:
            payload = json.load(f)
        self.assertIn("attention", payload)
        self.assertIn("ft_transformer", payload["attention"])
        md = open(os.path.join(reports_dir, "transformer_report.md")).read()
        self.assertIn("Attention analysis", md)

        # figures include attention plots
        joined = " ".join(result.figures)
        self.assertIn("attention_heatmap_ft_transformer", joined)
        self.assertIn("attention_features_ft_transformer", joined)

    def test_predictor_round_trip(self) -> None:
        result, cfg = self._run()
        predictor = TransformerPredictor(
            cfg["transformers"]["output"]["models_dir"],
            device="cpu").load()
        df = make_df(30, 9).drop(columns=[TARGET])
        proba = predictor.predict_proba(df)
        self.assertEqual(proba.shape, (30,))
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())
        preds = predictor.predict(df)
        self.assertTrue(set(np.unique(preds)) <= {0, 1})
        # schema enforcement
        with self.assertRaises(Exception):
            predictor.predict_proba(df.drop(columns=["f1"]))

    def test_checkpoint_resume(self) -> None:
        cfg = small_cfg(self.tmp)
        self._store()
        result, _ = self._run()
        last = os.path.join(cfg["transformers"]["output"]["models_dir"],
                            "ft_transformer_last.pt")
        self.assertTrue(os.path.exists(last))
        cfg["transformers"]["training"]["epochs"] = 5
        cfg["transformers"]["training"]["resume_from"] = last
        result2 = TransformerPipeline(cfg).run()
        # resumed from epoch 3 → epochs 4..5 only
        epochs = [e.epoch for e in result2.trained[0].history.epochs]
        self.assertEqual(epochs, [4, 5])

    def test_predict_proba_rejects_bad_input(self) -> None:
        model = build_transformer("tabular_encoder", 4,
                                  {"embed_dim": 8, "n_heads": 2,
                                   "n_layers": 1, "ff_dim": 8})
        with self.assertRaises(Exception):
            predict_proba(model, np.empty((0, 4)))
        bad = np.ones((3, 4), dtype=np.float32)
        bad[0, 0] = np.nan
        with self.assertRaises(Exception):
            predict_proba(model, bad)


# ── registry ────────────────────────────────────────────────────────────────
class TestRegistry(TmpDirTest):
    def test_registry_round_trip(self) -> None:
        cfg = small_cfg(self.tmp)
        self._store()
        TransformerPipeline(cfg).run()
        reg = TransformerModelRegistry(
            cfg["transformers"]["output"]["models_dir"])
        entries = reg.entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["network"], "ft_transformer")
        self.assertIn("metrics", entries[0])
        best = reg.best()
        self.assertEqual(best["network"], "ft_transformer")
        self.assertTrue(os.path.exists(reg.best_checkpoint()))


# ── trainer plumbing ────────────────────────────────────────────────────────
class TestTrainer(unittest.TestCase):
    def test_trainer_defaults_to_transformer_dir(self) -> None:
        tmp = tempfile.mkdtemp(prefix="tf_trainer_")
        try:
            tr = TransformerTrainer({"checkpoint": {"dir": tmp}})
            self.assertEqual(tr.checkpoint_dir, tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
