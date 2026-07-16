# Model Card — naive_bayes

Generated: 2026-07-16T08:46:23.838537+00:00

## Overview
- **Task**: binary financial-crisis (bankruptcy) prediction
- **Algorithm**: naive_bayes
- **Dataset version**: v001
- **Features**: 22
- **Decision threshold**: 0.000 (youden)

## Hyperparameters
`{'var_smoothing': 1e-09}`

## Performance (test)
| Metric | Value |
|---|---|
| accuracy | 0.8583 |
| precision | 0.1585 |
| recall | 0.7879 |
| f1 | 0.2640 |
| roc_auc | 0.9152 |
| pr_auc | 0.2352 |
| balanced_accuracy | 0.8242 |
| mcc | 0.3123 |
| cohen_kappa | 0.2222 |
| log_loss | 1.6051 |
| brier_score | 0.0837 |

## Intended use & limitations
- Early-warning signal for financial distress; not a standalone credit decisioning system.
- Trained on a heavily imbalanced sample; monitor calibration and drift before production use.
