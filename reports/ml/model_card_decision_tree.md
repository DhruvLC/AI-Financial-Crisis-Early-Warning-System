# Model Card — decision_tree

Generated: 2026-07-16T08:46:23.838072+00:00

## Overview
- **Task**: binary financial-crisis (bankruptcy) prediction
- **Algorithm**: decision_tree
- **Dataset version**: v001
- **Features**: 22
- **Decision threshold**: 0.830 (youden)

## Hyperparameters
`{'max_depth': 8, 'min_samples_leaf': 5, 'class_weight': 'balanced'}`

## Performance (test)
| Metric | Value |
|---|---|
| accuracy | 0.9091 |
| precision | 0.2059 |
| recall | 0.6364 |
| f1 | 0.3111 |
| roc_auc | 0.8109 |
| pr_auc | 0.2263 |
| balanced_accuracy | 0.7773 |
| mcc | 0.3270 |
| cohen_kappa | 0.2758 |
| log_loss | 0.5807 |
| brier_score | 0.0832 |

## Intended use & limitations
- Early-warning signal for financial distress; not a standalone credit decisioning system.
- Trained on a heavily imbalanced sample; monitor calibration and drift before production use.
