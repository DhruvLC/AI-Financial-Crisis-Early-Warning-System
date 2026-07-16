# Model Card — mlp

Generated: 2026-07-16T08:46:23.838609+00:00

## Overview
- **Task**: binary financial-crisis (bankruptcy) prediction
- **Algorithm**: mlp
- **Dataset version**: v001
- **Features**: 22
- **Decision threshold**: 0.066 (youden)

## Hyperparameters
`{'hidden_layer_sizes': (64, 32), 'activation': 'relu', 'alpha': 0.001, 'learning_rate_init': 0.001, 'max_iter': 500, 'early_stopping': True}`

## Performance (test)
| Metric | Value |
|---|---|
| accuracy | 0.8759 |
| precision | 0.1781 |
| recall | 0.7879 |
| f1 | 0.2905 |
| roc_auc | 0.9046 |
| pr_auc | 0.2924 |
| balanced_accuracy | 0.8333 |
| mcc | 0.3367 |
| cohen_kappa | 0.2511 |
| log_loss | 0.1011 |
| brier_score | 0.0271 |

## Intended use & limitations
- Early-warning signal for financial distress; not a standalone credit decisioning system.
- Trained on a heavily imbalanced sample; monitor calibration and drift before production use.
