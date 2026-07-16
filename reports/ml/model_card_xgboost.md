# Model Card — xgboost

Generated: 2026-07-16T08:46:23.838302+00:00

## Overview
- **Task**: binary financial-crisis (bankruptcy) prediction
- **Algorithm**: xgboost
- **Dataset version**: v001
- **Features**: 22
- **Decision threshold**: 0.003 (youden)

## Hyperparameters
`{'n_estimators': 600, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'scale_pos_weight': 29.993506493506494, 'eval_metric': 'logloss', 'n_jobs': -1}`

## Performance (test)
| Metric | Value |
|---|---|
| accuracy | 0.8358 |
| precision | 0.1466 |
| recall | 0.8485 |
| f1 | 0.2500 |
| roc_auc | 0.9349 |
| pr_auc | 0.3970 |
| balanced_accuracy | 0.8419 |
| mcc | 0.3101 |
| cohen_kappa | 0.2063 |
| log_loss | 0.1154 |
| brier_score | 0.0285 |

## Intended use & limitations
- Early-warning signal for financial distress; not a standalone credit decisioning system.
- Trained on a heavily imbalanced sample; monitor calibration and drift before production use.
