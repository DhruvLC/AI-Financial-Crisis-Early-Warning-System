# Model Card — random_forest

Generated: 2026-07-16T08:46:23.838156+00:00

## Overview
- **Task**: binary financial-crisis (bankruptcy) prediction
- **Algorithm**: random_forest
- **Dataset version**: v001
- **Features**: 22
- **Decision threshold**: 0.041 (youden)

## Hyperparameters
`{'n_estimators': 400, 'max_depth': None, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'n_jobs': -1}`

## Performance (test)
| Metric | Value |
|---|---|
| accuracy | 0.8319 |
| precision | 0.1436 |
| recall | 0.8485 |
| f1 | 0.2456 |
| roc_auc | 0.9378 |
| pr_auc | 0.4054 |
| balanced_accuracy | 0.8399 |
| mcc | 0.3058 |
| cohen_kappa | 0.2016 |
| log_loss | 0.0878 |
| brier_score | 0.0242 |

## Intended use & limitations
- Early-warning signal for financial distress; not a standalone credit decisioning system.
- Trained on a heavily imbalanced sample; monitor calibration and drift before production use.
