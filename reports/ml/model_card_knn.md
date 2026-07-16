# Model Card — knn

Generated: 2026-07-16T08:46:23.838461+00:00

## Overview
- **Task**: binary financial-crisis (bankruptcy) prediction
- **Algorithm**: knn
- **Dataset version**: v001
- **Features**: 22
- **Decision threshold**: 0.056 (youden)

## Hyperparameters
`{'n_neighbors': 15, 'weights': 'distance', 'n_jobs': -1}`

## Performance (test)
| Metric | Value |
|---|---|
| accuracy | 0.8661 |
| precision | 0.1667 |
| recall | 0.7879 |
| f1 | 0.2751 |
| roc_auc | 0.8534 |
| pr_auc | 0.3048 |
| balanced_accuracy | 0.8283 |
| mcc | 0.3227 |
| cohen_kappa | 0.2344 |
| log_loss | 0.3090 |
| brier_score | 0.0264 |

## Intended use & limitations
- Early-warning signal for financial distress; not a standalone credit decisioning system.
- Trained on a heavily imbalanced sample; monitor calibration and drift before production use.
