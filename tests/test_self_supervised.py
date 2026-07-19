"""Unit tests for the Self-Supervised Learning module
(``src/pipeline/self_supervised``).

Run with the project's venv (needs torch)::

    .venv/bin/python -m unittest tests.test_self_supervised -v
"""
from __future__ import annotations

import json
import os
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

from pipeline.self_supervised import (  # noqa: E402
    AUGMENTATION_REGISTRY, ENCODER_REGISTRY, BarlowTwinsLoss,
    ContrastiveDataset, KNNProbe, LinearProbe, NTXentLoss, ProjectionHead,
    RepresentationExporter, SSLDataLoader, SSLError, SSLModelRegistry,
    SSLPipeline, SSLTrainer, VICRegLoss, build_augmentations,
    build_contrastive_loader, build_encoder, build_projection_head,
    build_ssl_loss, embedding_statistics, extract_embeddings, seed_all,
)
from pipeline.feature_engineering.store import FeatureStore  # noqa: E402
from pipeline.deep_learning.base import DLError  # noqa: E402

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


def make_store(tmp: str) -> str:
    splits = {"train": make_df(300, 1), "val": make_df(120, 2),
              "test": make_df(120, 3)}
    store_dir = os.path.join(tmp, "features")
    FeatureStore(store_dir).save(
        splits, {"target_col": TARGET,
                 "features": [c for c in splits["train"].columns
                              if c != TARGET]})
    return store_dir


def small_cfg(tmp: str) -> dict:
    """Minimal fast config used by the pipeline tests."""
    return {
        "data": {"target_col": TARGET},
        "feature_engineering": {"store": {"dir": os.path.join(
            tmp, "features")}},
        "self_supervised": {
            "fail_fast": True,
            "random_state": 42,
            "encoders": ["mlp"],
            "data": {"batch_size": 64},
            "encoder_params": {"mlp": {"hidden_dims": [16],
                                       "embedding_dim": 8,
                                       "dropout": 0.0}},
            "projection_head": {"hidden_dims": [16],
                                "projection_dim": 8},
            "loss": {"name": "nt_xent", "temperature": 0.5},
            "augmentations": [{"name": "feature_masking", "ratio": 0.2},
                              {"name": "gaussian_noise", "sigma": 0.1}],
            "training": {"epochs": 3, "device": "cpu", "log_every": 10},
            "early_stopping": {"patience": 5},
            "checkpoint": {"dir": os.path.join(tmp, "models")},
            "evaluation": {"knn": {"enabled": True, "k": 5}},
            "visualization": {"dpi": 60, "projections": ["pca"],
                              "projection_max_samples": 100,
                              "similarity_max_samples": 50},
            "output": {
                "models_dir": os.path.join(tmp, "models"),
                "reports_dir": os.path.join(tmp, "reports"),
                "figures_dir": os.path.join(tmp, "reports", "figures"),
                "representations_dir": os.path.join(tmp, "models",
                                                    "representations"),
            },
        },
    }


class TestAugmentations(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(32, 8, generator=torch.Generator()
                             .manual_seed(0))
        self.gen = torch.Generator().manual_seed(1)

    def test_registry_complete(self):
        for name in ("feature_masking", "gaussian_noise",
                     "feature_dropout", "random_corruption",
                     "column_shuffle", "mixup"):
            self.assertIn(name, AUGMENTATION_REGISTRY)

    def test_each_augmentation_preserves_shape(self):
        for name, cls in AUGMENTATION_REGISTRY.items():
            out = cls()(self.x, self.gen)
            self.assertEqual(out.shape, self.x.shape, name)
            self.assertTrue(torch.isfinite(out).all(), name)

    def test_masking_zeroes_features(self):
        aug = AUGMENTATION_REGISTRY["feature_masking"](ratio=0.5)
        out = aug(self.x, self.gen)
        self.assertGreater(int((out == 0).sum()), 0)

    def test_deterministic_under_seed(self):
        aug = build_augmentations([{"name": "gaussian_noise",
                                    "sigma": 0.2}])
        a = aug(self.x, torch.Generator().manual_seed(7))
        b = aug(self.x, torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(a, b))

    def test_invalid_config_raises(self):
        with self.assertRaises(SSLError):
            build_augmentations([{"name": "nope"}])
        with self.assertRaises(SSLError):
            build_augmentations([{"name": "feature_masking",
                                  "ratio": 1.5}])
        with self.assertRaises(SSLError):
            build_augmentations([])


class TestEncodersAndHead(unittest.TestCase):
    def test_registry_complete(self):
        for name in ("mlp", "residual", "transformer"):
            self.assertIn(name, ENCODER_REGISTRY)

    def test_forward_shapes(self):
        x = torch.randn(16, 10)
        for name in ENCODER_REGISTRY:
            enc = build_encoder(name, 10, {"embedding_dim": 12})
            enc.eval()
            z = enc(x)
            self.assertEqual(z.shape, (16, 12), name)

    def test_unknown_encoder_raises(self):
        with self.assertRaises(SSLError):
            build_encoder("nope", 10)

    def test_invalid_params_raise(self):
        with self.assertRaises(SSLError):
            build_encoder("mlp", 0)
        with self.assertRaises(SSLError):
            build_encoder("mlp", 10, {"dropout": 2.0})

    def test_projection_head(self):
        head = build_projection_head(12, {"hidden_dims": [8],
                                          "projection_dim": 6})
        self.assertIsInstance(head, ProjectionHead)
        head.eval()
        self.assertEqual(head(torch.randn(4, 12)).shape, (4, 6))
        with self.assertRaises(SSLError):
            ProjectionHead(0, [8], 6)


class TestLosses(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.z1 = torch.randn(16, 8, generator=g)
        self.z2 = self.z1 + 0.05 * torch.randn(16, 8, generator=g)

    def test_ntxent_scalar_and_positive(self):
        loss = NTXentLoss(temperature=0.5)(self.z1, self.z2)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(float(loss), 0.0)

    def test_ntxent_prefers_aligned_views(self):
        aligned = NTXentLoss()(self.z1, self.z1)
        random = NTXentLoss()(self.z1, torch.randn(16, 8))
        self.assertLess(float(aligned), float(random))

    def test_temperature_matters(self):
        a = NTXentLoss(0.1)(self.z1, self.z2)
        b = NTXentLoss(1.0)(self.z1, self.z2)
        self.assertNotAlmostEqual(float(a), float(b))

    def test_batch_of_one_raises(self):
        with self.assertRaises(SSLError):
            NTXentLoss()(self.z1[:1], self.z2[:1])

    def test_optional_losses(self):
        for loss in (BarlowTwinsLoss(), VICRegLoss()):
            v = loss(self.z1, self.z2)
            self.assertTrue(torch.isfinite(v))

    def test_factory(self):
        self.assertIsInstance(build_ssl_loss({"name": "nt_xent"}),
                              NTXentLoss)
        self.assertIsInstance(build_ssl_loss({"name": "barlow_twins"}),
                              BarlowTwinsLoss)
        self.assertIsInstance(build_ssl_loss({"name": "vicreg"}),
                              VICRegLoss)
        with self.assertRaises(SSLError):
            build_ssl_loss({"name": "nope"})
        with self.assertRaises(SSLError):
            build_ssl_loss({"name": "nt_xent", "temperature": 0})


class TestDataLoading(unittest.TestCase):
    def test_contrastive_dataset_and_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = make_store(tmp)
            loader = SSLDataLoader(store, TARGET, batch_size=32, seed=42)
            data = loader.load()
            aug = build_augmentations(None)
            loaders = loader.contrastive_loaders(data, aug)
            v1, v2, y = next(iter(loaders["train"]))
            self.assertEqual(v1.shape, v2.shape)
            self.assertEqual(v1.shape[0], 32)
            self.assertFalse(torch.equal(v1, v2))  # independent views
            self.assertEqual(len(y), 32)

    def test_dataset_requires_augment(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = make_store(tmp)
            data = SSLDataLoader(store, TARGET).load()
            with self.assertRaises(SSLError):
                ContrastiveDataset(data.tensors["train"], None)

    def test_deterministic_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = make_store(tmp)
            aug = build_augmentations(None)

            def first_batch():
                data = SSLDataLoader(store, TARGET, batch_size=16,
                                     seed=7).load()
                ds = ContrastiveDataset(data.tensors["train"], aug, seed=7)
                dl = build_contrastive_loader(ds, 16, shuffle=True, seed=7)
                return next(iter(dl))

            a1, a2, _ = first_batch()
            b1, b2, _ = first_batch()
            self.assertTrue(torch.equal(a1, b1))
            self.assertTrue(torch.equal(a2, b2))


class TestTrainerAndRepresentation(unittest.TestCase):
    def _fit(self, tmp: str, epochs: int = 2, resume_from=None):
        store = make_store(tmp)
        loader = SSLDataLoader(store, TARGET, batch_size=64, seed=42)
        data = loader.load()
        aug = build_augmentations(None)
        loaders = loader.contrastive_loaders(data, aug)
        enc = build_encoder("mlp", data.n_features,
                           {"hidden_dims": [16], "embedding_dim": 8,
                            "dropout": 0.0})
        head = build_projection_head(8, {"hidden_dims": [16],
                                         "projection_dim": 8})
        trainer = SSLTrainer(
            {"training": {"epochs": epochs, "device": "cpu",
                          "log_every": 10},
             "loss": {"name": "nt_xent"},
             "checkpoint": {"dir": os.path.join(tmp, "models")}},
            checkpoint_dir=os.path.join(tmp, "models"))
        history, ckpts = trainer.fit("mlp", enc, head, loaders["train"],
                                     loaders["val"],
                                     resume_from=resume_from)
        return data, enc, trainer, history, ckpts

    def test_training_and_checkpoints(self):
        seed_all(42)
        with tempfile.TemporaryDirectory() as tmp:
            data, enc, trainer, history, ckpts = self._fit(tmp)
            self.assertEqual(len(history.epochs), 2)
            self.assertTrue(os.path.exists(ckpts["best"]))
            self.assertTrue(os.path.exists(ckpts["last"]))
            state = torch.load(ckpts["best"], weights_only=False)
            self.assertIn("encoder_state", state)
            self.assertIn("head_state", state)

    def test_resume_training(self):
        seed_all(42)
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, h1, ckpts = self._fit(tmp, epochs=2)
            _, _, _, h2, _ = self._fit(tmp, epochs=4,
                                       resume_from=ckpts["last"])
            self.assertEqual(h2.epochs[0].epoch, 3)  # resumed after 2

    def test_corrupt_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad.pt")
            with open(bad, "w") as f:
                f.write("not a checkpoint")
            # corrupt checkpoints raise the shared DLError family
            # (SSLError subclasses it)
            with self.assertRaises(DLError):
                self._fit(tmp, resume_from=bad)

    def test_extract_embeddings_and_stats(self):
        seed_all(42)
        with tempfile.TemporaryDirectory() as tmp:
            data, enc, trainer, _, _ = self._fit(tmp)
            X, _ = data.numpy("test")
            Z = extract_embeddings(enc, X, "cpu")
            self.assertEqual(Z.shape, (len(X), 8))
            stats = embedding_statistics(Z)
            self.assertEqual(stats["n_samples"], len(X))
            with self.assertRaises(SSLError):
                extract_embeddings(enc, np.empty((0, 4)))
            with self.assertRaises(SSLError):
                embedding_statistics(np.array([[np.nan, 1.0]]))

    def test_representation_export(self):
        seed_all(42)
        with tempfile.TemporaryDirectory() as tmp:
            data, enc, trainer, _, _ = self._fit(tmp)
            exporter = RepresentationExporter(os.path.join(tmp, "reps"))
            paths, stats = exporter.export("mlp", enc, data, "cpu")
            for split in ("train", "val", "test"):
                self.assertTrue(os.path.exists(paths[split]))
                df = pd.read_parquet(paths[split])
                self.assertIn(TARGET, df.columns)
                self.assertEqual(df.shape[1], 8 + 1)
            meta = exporter.write_metadata(
                [{"encoder": "mlp", "paths": paths}], "v001")
            with open(meta) as f:
                payload = json.load(f)
            self.assertEqual(payload["dataset_version"], "v001")


class TestProbes(unittest.TestCase):
    def test_linear_and_knn_probes(self):
        rng = np.random.default_rng(0)
        Z, y = {}, {}
        for split, n in (("train", 200), ("val", 80), ("test", 80)):
            labels = (rng.random(n) < 0.3).astype(int)
            Z[split] = (labels[:, None] * 2.0
                        + rng.normal(0, 1, (n, 6))).astype(np.float32)
            y[split] = labels
        evals, threshold, method = LinearProbe().evaluate("enc", Z, y)
        self.assertIn("test", evals)
        self.assertGreater(evals["test"].metrics["roc_auc"], 0.7)
        self.assertTrue(0.0 <= threshold <= 1.0)
        knn = KNNProbe({"k": 5}).evaluate("enc", Z, y)
        self.assertGreater(knn["test"].metrics["roc_auc"], 0.6)

    def test_probe_requires_train(self):
        with self.assertRaises(SSLError):
            LinearProbe().evaluate("enc", {"test": np.zeros((4, 2))},
                                   {"test": np.zeros(4)})


class TestPipelineEndToEnd(unittest.TestCase):
    def test_full_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_store(tmp)
            cfg = small_cfg(tmp)
            result = SSLPipeline(cfg).run()

            self.assertEqual(len(result.trained), 1)
            t = result.trained[0]
            self.assertFalse(t.failed)
            # probe evaluations + leaderboard
            self.assertIn("test", t.evaluations)
            self.assertEqual(int(result.leaderboard.iloc[0]["rank"]), 1)
            # checkpoints + best-encoder artefacts
            mdir = cfg["self_supervised"]["output"]["models_dir"]
            for f in ("mlp_best.pt", "mlp_last.pt", "best_encoder.pt",
                      "last_encoder.pt", "training_config.json",
                      "metrics.json", "history.json", "registry.json",
                      "feature_metadata.json"):
                self.assertTrue(os.path.exists(os.path.join(mdir, f)), f)
            # representations + metadata
            self.assertTrue(os.path.exists(result.representation_metadata))
            for split in ("train", "val", "test"):
                self.assertTrue(os.path.exists(t.representations[split]))
            # reports + figures
            rdir = cfg["self_supervised"]["output"]["reports_dir"]
            for f in ("self_supervised_report.json",
                      "self_supervised_report.md",
                      "self_supervised_report.html", "leaderboard.csv",
                      "metrics_summary.csv", "training_history_mlp.csv"):
                self.assertTrue(os.path.exists(os.path.join(rdir, f)), f)
            self.assertGreater(len(result.figures), 0)
            # registry contents
            reg = SSLModelRegistry(mdir)
            self.assertEqual(len(reg.entries()), 1)
            self.assertTrue(reg.best_checkpoint().endswith(
                "best_encoder.pt"))
            # JSON report carries SSL-specific sections
            with open(os.path.join(
                    rdir, "self_supervised_report.json")) as f:
                report = json.load(f)
            self.assertIn("augmentations", report)
            self.assertIn("representations", report)

    def test_unknown_encoder_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_store(tmp)
            cfg = small_cfg(tmp)
            cfg["self_supervised"]["encoders"] = ["nope"]
            with self.assertRaises(SSLError):
                SSLPipeline(cfg).run()


if __name__ == "__main__":
    unittest.main(verbosity=2)
