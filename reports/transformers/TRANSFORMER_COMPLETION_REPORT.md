# Transformer Models — Completion Report

**Stage 10 — Transformer Models** · Generated: 2026-07-18 · Status: **complete and verified**

## Summary

The Transformer Models module trains three attention-based tabular architectures on the
versioned feature store (`data/features/v001`, 22 engineered features), evaluates them with
the shared metric suite, explains them via permutation importance, SHAP, and attention
analysis, and registers every model — reusing the deep-learning training engine, data
loading, metric suite, registry, and visualizer verbatim. No previous phase was modified.

## Files created

| Path | Purpose |
|---|---|
| `src/pipeline/transformers/__init__.py` | Public API re-exports |
| `src/pipeline/transformers/base.py` | Error type, `TrainedTransformer`, `AttentionSummary`, attention math helpers (reuses DL base) |
| `src/pipeline/transformers/data_loader.py` | Feature-store → PyTorch loaders (reuses `DLDataLoader`) |
| `src/pipeline/transformers/models.py` | FT-Transformer, TabTransformer, Tabular Encoder + attention capture |
| `src/pipeline/transformers/trainer.py` | `TransformerTrainer` (reuses DL `Trainer`: AMP, grad clip, checkpoints, resume, early stopping) |
| `src/pipeline/transformers/evaluation.py` | Metric suite re-export (`pipeline.ml.evaluation` via DL) |
| `src/pipeline/transformers/prediction.py` | `TransformerPredictor` — schema-enforced inference |
| `src/pipeline/transformers/registry.py` | `TransformerModelRegistry` under `models/transformers/` |
| `src/pipeline/transformers/report.py` | JSON/MD/HTML/CSV reports + attention & cross-family sections |
| `src/pipeline/transformers/visualization.py` | Attention heatmaps/bars + cross-family chart (extends `DLVisualizer`) |
| `src/pipeline/transformers/pipeline.py` | `TransformerPipeline` orchestrator (with automatic CPU retry on accelerator NaN divergence) |
| `src/run_transformers.py` | CLI entry point |
| `configs/config.yaml` | New `transformers:` section (models, optimizer, scheduler, training, evaluation, explainability, comparison, visualization, checkpoint, output) |
| `tests/test_transformers.py` | 17 unit tests |

## Models implemented

1. **FT-Transformer** — per-feature affine tokenizer + `[CLS]` token + pre-norm encoder, CLS head
2. **TabTransformer** — tokenized features contextualized by the encoder; flattened contextual embeddings ⊕ layer-normed raw features → MLP head (continuous adaptation)
3. **Tabular Transformer Encoder** — feature tokens + learned positional embeddings + encoder + mean pooling

All configurable: `embed_dim`, `n_heads`, `n_layers`, `ff_dim`, `dropout`, positional embeddings (encoder), LayerNorm + residual connections in every block, attention-weight capture per layer.

## Training engine (reused from Deep Learning)

Mini-batch training · CUDA/MPS/CPU with fallback · mixed precision (CUDA) · deterministic
mode (`seed_all`) · gradient clipping · best/last checkpointing · resume (`training.resume_from`)
· losses BCE / BCEWithLogits / weighted BCE (+focal) · optimizers Adam / AdamW (+SGD/RMSProp)
· schedulers CosineAnnealingLR / ReduceLROnPlateau / StepLR (+exponential) · early stopping
(patience, min_delta, best-weight restoration).

## Results (test split, feature store v001)

| Rank | Model | ROC-AUC | F1 | Recall | PR-AUC | Params | Best epoch |
|---|---|---|---|---|---|---|---|
| 1 | tabular_encoder | **0.9393** | 0.3354 | 0.8182 | 0.4112 | 71,361 | 14 |
| 2 | tab_transformer | 0.9330 | 0.3103 | 0.8182 | 0.3368 | — | — |
| 3 | ft_transformer | 0.8242 | 0.1393 | 0.9394 | 0.0939 | — | — |

**Best model: `tabular_encoder`** (threshold 0.676, Youden) — on par with the best deep-learning
network (wide_deep, 0.9370) and the best classical model (extra_trees, 0.9395). Full metric set
(accuracy, precision, recall, F1, ROC-AUC, PR-AUC, MCC, balanced accuracy, Cohen's κ, log loss,
Brier, confusion matrix) in `reports/transformers/metrics_summary.csv` and `metrics.json`.

## Explainability

- Permutation importance (per model, ROC-AUC drop, val split)
- SHAP mean-|value| summaries (reused DL SHAP flow)
- Attention analysis: mean attention received per feature, per-layer attention entropy,
  last-layer token×token heatmap — in the report and as figures

## Reports & figures

`reports/transformers/`: `transformer_report.{json,md,html}`, `leaderboard.csv`,
`metrics_summary.csv`, `training_history_<model>.csv` — including architecture,
hyperparameters, attention analysis, and the **cross-family comparison** with the ML and
DL leaderboards.

`reports/transformers/figures/` (38): loss / accuracy / LR curves, ROC, PR, confusion,
calibration, prediction distribution, permutation importance, attention heatmap,
attention-by-feature per model; metric comparison bars; `cross_family_roc_auc.png`.

## Registry (`models/transformers/`)

`best_model.pt`, `last_model.pt`, per-model `<name>_{best,last}.pt`, `metrics.json`,
`history.json`, `training_config.json`, `feature_metadata.json`, `registry.json` (3 entries).

## Tests

- `tests/test_transformers.py`: **17/17 pass** — data loading (batching, reproducibility, validation),
  model initialization + forward pass for all three architectures, configurability, attention capture
  & math, end-to-end training pipeline, evaluation metric coverage, prediction round-trip + schema
  enforcement, checkpoint resume, registry round-trip, report generation.
- Regression: `tests.test_deep_learning` + `tests.test_machine_learning` — **54/54 pass** (backward compatible).

## End-to-end verification

`python src/run_transformers.py --config configs/config.yaml` — 3/3 models trained,
evaluated, explained, plotted, registered; inference verified by reloading
`best_model.pt` through `TransformerPredictor` and scoring the test split (1023 rows,
probabilities/labels/risk-scores all valid).

## Production readiness

Config-driven (all hyperparameters in `configs/config.yaml`), modular (each concern in its
own file), fail-fast or isolate-and-continue per `transformers.fail_fast`, deterministic
seeding, schema-enforced serving, NaN-loss detection with automatic CPU retry when an
accelerator kernel diverges, structured logging via the shared `ingestion.logging_config`,
and full artefact lineage (feature-store version recorded in every registry entry).
