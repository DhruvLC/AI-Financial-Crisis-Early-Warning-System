# Model Card — logistic_regression

Generated: 2026-07-16T08:46:23.837982+00:00

## Overview
- **Task**: binary financial-crisis (bankruptcy) prediction
- **Algorithm**: logistic_regression
- **Dataset version**: v001
- **Features**: 22
- **Decision threshold**: 0.399 (youden)

## Hyperparameters
`{'C': 1.0, 'max_iter': 2000, 'class_weight': 'balanced', 'solver': 'lbfgs'}`

## Performance (test)
| Metric | Value |
|---|---|
| accuracy | 0.8133 |
| precision | 0.1308 |
| recall | 0.8485 |
| f1 | 0.2267 |
| roc_auc | 0.9241 |
| pr_auc | 0.2831 |
| balanced_accuracy | 0.8303 |
| mcc | 0.2870 |
| cohen_kappa | 0.1809 |
| log_loss | 0.3668 |
| brier_score | 0.1079 |

## Intended use & limitations
- Early-warning signal for financial distress; not a standalone credit decisioning system.
- Trained on a heavily imbalanced sample; monitor calibration and drift before production use.
