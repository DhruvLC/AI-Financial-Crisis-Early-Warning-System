# Model Card — svm

Generated: 2026-07-16T08:46:23.838379+00:00

## Overview
- **Task**: binary financial-crisis (bankruptcy) prediction
- **Algorithm**: svm
- **Dataset version**: v001
- **Features**: 22
- **Decision threshold**: 0.028 (youden)

## Hyperparameters
`{'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'class_weight': 'balanced', 'probability': True}`

## Performance (test)
| Metric | Value |
|---|---|
| accuracy | 0.8270 |
| precision | 0.1471 |
| recall | 0.9091 |
| f1 | 0.2532 |
| roc_auc | 0.9100 |
| pr_auc | 0.2697 |
| balanced_accuracy | 0.8667 |
| mcc | 0.3243 |
| cohen_kappa | 0.2093 |
| log_loss | 0.1012 |
| brier_score | 0.0268 |

## Intended use & limitations
- Early-warning signal for financial distress; not a standalone credit decisioning system.
- Trained on a heavily imbalanced sample; monitor calibration and drift before production use.
