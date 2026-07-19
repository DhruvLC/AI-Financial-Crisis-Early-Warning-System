# Self-Supervised Learning — Completion Report

**Stage 11** of the AI Financial Crisis Early Warning System.
Generated: 2026-07-19. Status: **COMPLETE & VERIFIED**.

## Summary

Contrastive (SimCLR-style) representation learning over the engineered
feature store: configurable tabular augmentations produce two views per
sample, configurable encoders + a projection head are pretrained with
NT-Xent (Barlow Twins / VICReg also available), latent representations are
exported for downstream reuse, and representation quality is judged with a
frozen-encoder linear probe and a KNN probe using the shared metric suite.

## Files created

```
src/pipeline/self_supervised/
    __init__.py          # public API
    base.py              # SSLError, TrainedEncoder, embedding_statistics
    data_loader.py       # SSLDataLoader, ContrastiveDataset (two views)
    augmentations.py     # 6 augmentations + registry + pipeline factory
    encoder.py           # MLP / Residual / Transformer encoders + registry
    projection_head.py   # SimCLR projection head
    losses.py            # NT-Xent (InfoNCE), Barlow Twins, VICReg
    trainer.py           # SSLTrainer (two-view contrastive loop)
    representation.py    # extract_embeddings, RepresentationExporter
    evaluation.py        # LinearProbe, KNNProbe (shared metric suite)
    registry.py          # SSLModelRegistry (best_encoder.pt naming)
    visualization.py     # projections, similarity, distributions
    report.py            # SSLReport (JSON/MD/HTML/CSV)
    pipeline.py          # SSLPipeline orchestrator
src/run_self_supervised.py    # CLI entry point
tests/test_self_supervised.py # 28 unit tests
configs/config.yaml           # + self_supervised: section
```

## Reused infrastructure (no previous phase modified)

- **Feature Store** — `SSLDataLoader` extends `DLDataLoader` →
  `MLDataLoader` over `data/features/` versioned splits.
- **Deep Learning** — `TrainingHistory`/`EpochRecord`, device/seed
  utilities, optimizer/scheduler/early-stopping factories,
  `load_checkpoint`, `DLModelRegistry`, `DLVisualizer`, `DLReport`.
- **Transformer module** — `FeatureTokenizer` + `EncoderBlock` power the
  SSL Transformer encoder directly.
- **ML module** — `ModelEvaluator` / `ThresholdOptimizer` metric suite;
  probe numbers are directly comparable to all prior leaderboards.
- **Logging** — `ingestion.logging_config` throughout (`ssl.*` loggers).

## Encoder architectures

| Encoder | Key params | Output |
|---|---|---|
| `mlp` | hidden_dims [256,128], BN, dropout | 64-d embedding |
| `residual` | width 256, 3 pre-activation res blocks | 64-d embedding |
| `transformer` | 2 encoder layers, 4 heads, mean pooling | 64-d embedding |

Projection head (pretrain-only): 64 → 128 → 64 MLP with BN.

## Augmentations (configurable, deterministic under the pipeline seed)

feature_masking (0.15), gaussian_noise (0.1) enabled by default;
feature_dropout, random_corruption, column_shuffle (swap-noise), mixup
available via config.

## Loss

NT-Xent / InfoNCE with temperature 0.5 (configurable); optional
Barlow Twins and VICReg implemented and tested.

## Training engine

Mini-batch two-view loop, CUDA → MPS → CPU auto-selection, AMP on CUDA,
gradient clipping, cosine LR schedule, early stopping (patience 15) on
val contrastive loss with best-weight restoration, NaN detection,
best/last checkpointing (encoder + head + optimizer, resumable).

## End-to-end verification (feature store v001, 22 features, 30 epochs)

Linear-probe leaderboard (test split):

| Rank | Encoder | ROC-AUC | F1 | Recall | PR-AUC |
|---|---|---|---|---|---|
| 1 | transformer | 0.9385 | 0.3043 | 0.8485 | 0.3229 |
| 2 | residual | 0.9376 | 0.3521 | 0.7576 | 0.3553 |
| 3 | mlp | 0.9337 | 0.3133 | 0.7879 | 0.3010 |

Best encoder: **transformer** (v003) → `models/self_supervised/best_encoder.pt`.

## Artefacts

- **Registry** (`models/self_supervised/`): per-encoder `*_best.pt` /
  `*_last.pt`, `best_encoder.pt`, `last_encoder.pt`,
  `training_config.json`, `metrics.json`, `history.json`,
  `feature_metadata.json`, `registry.json` (3 entries).
- **Representations** (`models/self_supervised/representations/`):
  `<encoder>_{train,val,test}.parquet` (64 embedding cols + target;
  e.g. transformer_test: 1023×65) + `representation_metadata.json`
  with per-split embedding statistics — ready for downstream modules.
- **Reports** (`reports/self_supervised/`): JSON, Markdown, HTML,
  `leaderboard.csv`, `metrics_summary.csv`,
  `training_history_<encoder>.csv` — covering architecture,
  hyperparameters, augmentations, loss, representations, probe metrics,
  checkpoints, and figures.
- **Figures** (22 under `reports/self_supervised/figures/`): loss/LR
  curves, PCA + t-SNE embedding projections (UMAP auto-skipped — not
  installed), cosine-similarity matrices, representation distributions,
  cross-encoder comparison bars.

## Exception handling

NaN embeddings/losses (`SSLError` with remediation hint), invalid
augmentations/configs (fail-fast at build time), corrupt checkpoints
(validated on load), device fallback (unavailable device → CPU with a
warning), per-encoder failure isolation honouring `fail_fast`.

## Unit tests

`tests/test_self_supervised.py`: 28 tests — augmentations (registry,
shape/finiteness, determinism, invalid config), encoders + projection
head, all three losses + factory, contrastive data loading +
determinism, training + checkpointing + resume + corrupt-checkpoint,
representation extraction/export + NaN guards, linear/KNN probes, and
the full pipeline end-to-end. **All pass**, plus the full prior-stage
regression suite (transformers, deep learning, machine learning:
71 tests) still passes — backward compatibility preserved.

## Production readiness

Config-driven, modular, documented (module + class docstrings
throughout), deterministic under seed, device-portable, resumable,
per-component exception isolation, versioned registry, and complete
report/figure suite. Later phases (deployment, API, monitoring) were
deliberately **not** started.
