# Model Card — extra_trees

Generated: 2026-07-16T08:46:23.838230+00:00

## Overview
- **Task**: binary financial-crisis (bankruptcy) prediction
- **Algorithm**: extra_trees
- **Dataset version**: v001
- **Features**: 22
- **Decision threshold**: 0.190 (youden)

## Hyperparameters
`{'n_estimators': 400, 'max_depth': None, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'n_jobs': -1}`

## Performance (test)
| Metric | Value |
|---|---|
| accuracy | 0.9179 |
| precision | 0.2475 |
| recall | 0.7576 |
| f1 | 0.3731 |
| roc_auc | 0.9395 |
| pr_auc | 0.4380 |
| balanced_accuracy | 0.8404 |
| mcc | 0.4032 |
| cohen_kappa | 0.3411 |
| log_loss | 0.0984 |
| brier_score | 0.0280 |

## Intended use & limitations
- Early-warning signal for financial distress; not a standalone credit decisioning system.
- Trained on a heavily imbalanced sample; monitor calibration and drift before production use.
