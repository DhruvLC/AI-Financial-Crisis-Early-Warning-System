# Deep Learning Report

Generated: 2026-07-18T09:02:56.955461+00:00
Dataset: feature store v001 (target `Bankrupt?`, 22 features)

## Training summary

| Model | Status | Params | Device | Epochs | Best epoch | Train (s) | Threshold |
|---|---|---|---|---|---|---|---|
| mlp | trained | 3,777 | mps | 19 | 9 | 18.22 | 0.602 (youden) |
| deep_fc | trained | 50,113 | mps | 19 | 9 | 16.35 | 0.638 (youden) |
| residual | trained | 103,937 | mps | 22 | 12 | 21.67 | 0.421 (youden) |
| wide_deep | trained | 11,672 | mps | 18 | 8 | 12.38 | 0.653 (youden) |

## mlp

### Architecture
`{'network': 'mlp', 'n_features': 22, 'hidden_layers': [64, 32], 'activation': 'relu', 'dropout': 0.2, 'batch_norm': True, 'initialization': 'kaiming'}`

### Hyperparameters
`{'data': {'batch_size': 128}, 'optimizer': {'name': 'adamw', 'lr': 0.001, 'weight_decay': 0.0001}, 'scheduler': {'name': 'plateau', 'factor': 0.5, 'patience': 5}, 'loss': {'name': 'weighted_bce', 'gamma': 2.0, 'alpha': 0.25}, 'training': {'epochs': 60, 'device': 'auto', 'mixed_precision': True, 'gradient_clip': 1.0, 'log_every': 5}}`

### Metrics — val (threshold 0.602)
| Metric | Value |
|---|---|
| accuracy | 0.8905 |
| precision | 0.2117 |
| recall | 0.8788 |
| f1 | 0.3412 |
| roc_auc | 0.9403 |
| pr_auc | 0.4107 |
| balanced_accuracy | 0.8848 |
| mcc | 0.3993 |
| cohen_kappa | 0.3050 |
| log_loss | 0.3392 |
| brier_score | 0.1016 |

Confusion: TN=882 FP=108 FN=4 TP=29

### Metrics — test (threshold 0.602)
| Metric | Value |
|---|---|
| accuracy | 0.8905 |
| precision | 0.1938 |
| recall | 0.7576 |
| f1 | 0.3086 |
| roc_auc | 0.9321 |
| pr_auc | 0.3219 |
| balanced_accuracy | 0.8263 |
| mcc | 0.3473 |
| cohen_kappa | 0.2712 |
| log_loss | 0.3635 |
| brier_score | 0.1050 |

Confusion: TN=886 FP=104 FN=8 TP=25

### Top features (permutation)

| feature | importance | std |
|---|---|---|
| pca__1 | 0.0541 | 0.0114 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0089 | 0.0050 |
| pca__2 | 0.0083 | 0.0033 |
| pca__6 | 0.0065 | 0.0022 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0056 | 0.0058 |
| Allocation rate per person | 0.0038 | 0.0020 |
| pca__7 | 0.0033 | 0.0006 |
| Non-industry income and expenditure/revenue | 0.0032 | 0.0007 |
| ROA(B) before interest and depreciation after tax | 0.0028 | 0.0016 |
| Borrowing dependency | 0.0026 | 0.0033 |
| log__Net Value Growth Rate | 0.0022 | 0.0009 |
| pca__4 | 0.0021 | 0.0007 |
| pca__5 | 0.0009 | 0.0008 |
| pca__3 | 0.0009 | 0.0014 |
| Retained Earnings to Total Assets | 0.0007 | 0.0002 |

### Top features (SHAP)

| Feature | mean abs SHAP |
|---|---|
| pca__1 | 0.0697 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0410 |
| Debt ratio % | 0.0212 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0199 |
| Borrowing dependency | 0.0172 |
| pca__4 | 0.0115 |
| Allocation rate per person | 0.0103 |
| Total income/Total expense | 0.0102 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0086 |
| pca__2 | 0.0080 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0077 |
| ROA(B) before interest and depreciation after tax | 0.0066 |
| pca__3 | 0.0063 |
| Total debt/Total net worth | 0.0056 |
| pca__7 | 0.0046 |

## deep_fc

### Architecture
`{'network': 'deep_fc', 'n_features': 22, 'hidden_layers': [256, 128, 64, 32], 'activation': 'gelu', 'dropout': 0.3, 'batch_norm': True, 'initialization': 'kaiming'}`

### Hyperparameters
`{'data': {'batch_size': 128}, 'optimizer': {'name': 'adamw', 'lr': 0.001, 'weight_decay': 0.0001}, 'scheduler': {'name': 'plateau', 'factor': 0.5, 'patience': 5}, 'loss': {'name': 'weighted_bce', 'gamma': 2.0, 'alpha': 0.25}, 'training': {'epochs': 60, 'device': 'auto', 'mixed_precision': True, 'gradient_clip': 1.0, 'log_every': 5}}`

### Metrics — val (threshold 0.638)
| Metric | Value |
|---|---|
| accuracy | 0.8876 |
| precision | 0.2071 |
| recall | 0.8788 |
| f1 | 0.3353 |
| roc_auc | 0.9374 |
| pr_auc | 0.3788 |
| balanced_accuracy | 0.8833 |
| mcc | 0.3941 |
| cohen_kappa | 0.2986 |
| log_loss | 0.3339 |
| brier_score | 0.1054 |

Confusion: TN=879 FP=111 FN=4 TP=29

### Metrics — test (threshold 0.638)
| Metric | Value |
|---|---|
| accuracy | 0.8847 |
| precision | 0.1898 |
| recall | 0.7879 |
| f1 | 0.3059 |
| roc_auc | 0.9309 |
| pr_auc | 0.3388 |
| balanced_accuracy | 0.8379 |
| mcc | 0.3506 |
| cohen_kappa | 0.2678 |
| log_loss | 0.3496 |
| brier_score | 0.1093 |

Confusion: TN=879 FP=111 FN=7 TP=26

### Top features (permutation)

| feature | importance | std |
|---|---|---|
| pca__1 | 0.0375 | 0.0061 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0210 | 0.0124 |
| Retained Earnings to Total Assets | 0.0094 | 0.0020 |
| ROA(B) before interest and depreciation after tax | 0.0072 | 0.0019 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0031 | 0.0004 |
| Allocation rate per person | 0.0023 | 0.0022 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0022 | 0.0018 |
| Total debt/Total net worth | 0.0014 | 0.0008 |
| pca__7 | 0.0011 | 0.0004 |
| pca__2 | 0.0011 | 0.0025 |
| log__Net Value Growth Rate | 0.0010 | 0.0006 |
| Total income/Total expense | 0.0008 | 0.0005 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0006 | 0.0063 |
| Borrowing dependency | 0.0004 | 0.0026 |
| pca__6 | 0.0003 | 0.0003 |

### Top features (SHAP)

| Feature | mean abs SHAP |
|---|---|
| pca__1 | 0.0634 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0322 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0194 |
| ROA(B) before interest and depreciation after tax | 0.0173 |
| Retained Earnings to Total Assets | 0.0154 |
| Debt ratio % | 0.0153 |
| Allocation rate per person | 0.0144 |
| Borrowing dependency | 0.0136 |
| Total income/Total expense | 0.0109 |
| log__Net Value Growth Rate | 0.0101 |
| pca__4 | 0.0088 |
| pca__3 | 0.0076 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0070 |
| pca__2 | 0.0067 |
| pca__7 | 0.0059 |

## residual

### Architecture
`{'network': 'residual', 'n_features': 22, 'width': 128, 'n_blocks': 3, 'activation': 'relu', 'dropout': 0.2, 'batch_norm': True, 'initialization': 'kaiming'}`

### Hyperparameters
`{'data': {'batch_size': 128}, 'optimizer': {'name': 'adamw', 'lr': 0.001, 'weight_decay': 0.0001}, 'scheduler': {'name': 'plateau', 'factor': 0.5, 'patience': 5}, 'loss': {'name': 'weighted_bce', 'gamma': 2.0, 'alpha': 0.25}, 'training': {'epochs': 60, 'device': 'auto', 'mixed_precision': True, 'gradient_clip': 1.0, 'log_every': 5}}`

### Metrics — val (threshold 0.421)
| Metric | Value |
|---|---|
| accuracy | 0.8602 |
| precision | 0.1765 |
| recall | 0.9091 |
| f1 | 0.2956 |
| roc_auc | 0.9421 |
| pr_auc | 0.3733 |
| balanced_accuracy | 0.8838 |
| mcc | 0.3644 |
| cohen_kappa | 0.2553 |
| log_loss | 0.2949 |
| brier_score | 0.0872 |

Confusion: TN=850 FP=140 FN=3 TP=30

### Metrics — test (threshold 0.421)
| Metric | Value |
|---|---|
| accuracy | 0.8573 |
| precision | 0.1576 |
| recall | 0.7879 |
| f1 | 0.2626 |
| roc_auc | 0.9277 |
| pr_auc | 0.3287 |
| balanced_accuracy | 0.8237 |
| mcc | 0.3110 |
| cohen_kappa | 0.2207 |
| log_loss | 0.3322 |
| brier_score | 0.0931 |

Confusion: TN=851 FP=139 FN=7 TP=26

### Top features (permutation)

| feature | importance | std |
|---|---|---|
| pca__1 | 0.2173 | 0.0331 |
| pca__2 | 0.0141 | 0.0015 |
| pca__3 | 0.0067 | 0.0009 |
| Allocation rate per person | 0.0048 | 0.0020 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0040 | 0.0063 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0035 | 0.0022 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0031 | 0.0016 |
| Non-industry income and expenditure/revenue | 0.0031 | 0.0017 |
| ROA(B) before interest and depreciation after tax | 0.0029 | 0.0024 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0023 | 0.0053 |
| pca__5 | 0.0018 | 0.0010 |
| log__Net Value Growth Rate | 0.0016 | 0.0003 |
| Debt ratio % | 0.0015 | 0.0014 |
| Total income/Total expense | 0.0014 | 0.0009 |
| Total debt/Total net worth | 0.0014 | 0.0015 |

### Top features (SHAP)

| Feature | mean abs SHAP |
|---|---|
| pca__1 | 0.1158 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0206 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0204 |
| Total income/Total expense | 0.0146 |
| Allocation rate per person | 0.0129 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0127 |
| pca__5 | 0.0096 |
| Non-industry income and expenditure/revenue | 0.0090 |
| ROA(B) before interest and depreciation after tax | 0.0080 |
| log__Net Value Growth Rate | 0.0078 |
| pca__3 | 0.0064 |
| pca__6 | 0.0062 |
| Retained Earnings to Total Assets | 0.0060 |
| pca__2 | 0.0060 |
| Total debt/Total net worth | 0.0059 |

## wide_deep

### Architecture
`{'network': 'wide_deep', 'n_features': 22, 'hidden_layers': [128, 64], 'activation': 'relu', 'dropout': 0.2, 'batch_norm': True, 'initialization': 'kaiming'}`

### Hyperparameters
`{'data': {'batch_size': 128}, 'optimizer': {'name': 'adamw', 'lr': 0.001, 'weight_decay': 0.0001}, 'scheduler': {'name': 'plateau', 'factor': 0.5, 'patience': 5}, 'loss': {'name': 'weighted_bce', 'gamma': 2.0, 'alpha': 0.25}, 'training': {'epochs': 60, 'device': 'auto', 'mixed_precision': True, 'gradient_clip': 1.0, 'log_every': 5}}`

### Metrics — val (threshold 0.653)
| Metric | Value |
|---|---|
| accuracy | 0.9189 |
| precision | 0.2642 |
| recall | 0.8485 |
| f1 | 0.4029 |
| roc_auc | 0.9389 |
| pr_auc | 0.3954 |
| balanced_accuracy | 0.8848 |
| mcc | 0.4462 |
| cohen_kappa | 0.3720 |
| log_loss | 0.2910 |
| brier_score | 0.0877 |

Confusion: TN=912 FP=78 FN=5 TP=28

### Metrics — test (threshold 0.653)
| Metric | Value |
|---|---|
| accuracy | 0.9071 |
| precision | 0.2232 |
| recall | 0.7576 |
| f1 | 0.3448 |
| roc_auc | 0.9370 |
| pr_auc | 0.3585 |
| balanced_accuracy | 0.8348 |
| mcc | 0.3790 |
| cohen_kappa | 0.3105 |
| log_loss | 0.3178 |
| brier_score | 0.0932 |

Confusion: TN=903 FP=87 FN=8 TP=25

### Top features (permutation)

| feature | importance | std |
|---|---|---|
| pca__1 | 0.4969 | 0.0306 |
| ROA(B) before interest and depreciation after tax | 0.0346 | 0.0062 |
| pca__5 | 0.0257 | 0.0009 |
| Borrowing dependency | 0.0183 | 0.0051 |
| pca__2 | 0.0178 | 0.0048 |
| Retained Earnings to Total Assets | 0.0153 | 0.0037 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0125 | 0.0009 |
| Total debt/Total net worth | 0.0082 | 0.0016 |
| Total income/Total expense | 0.0062 | 0.0034 |
| Allocation rate per person | 0.0039 | 0.0009 |
| pca__4 | 0.0034 | 0.0013 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0031 | 0.0080 |
| pca__6 | 0.0024 | 0.0010 |
| pca__7 | 0.0022 | 0.0021 |
| Net Income to Total Assets | 0.0017 | 0.0019 |

### Top features (SHAP)

| Feature | mean abs SHAP |
|---|---|
| pca__1 | 0.1192 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0361 |
| ROA(B) before interest and depreciation after tax | 0.0344 |
| Borrowing dependency | 0.0297 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0267 |
| pca__5 | 0.0255 |
| Total income/Total expense | 0.0252 |
| pca__2 | 0.0229 |
| pca__4 | 0.0220 |
| Debt ratio % | 0.0166 |
| Net Income to Total Assets | 0.0165 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0157 |
| Retained Earnings to Total Assets | 0.0131 |
| pca__7 | 0.0122 |
| pca__3 | 0.0083 |

## Leaderboard (test)

| rank | model | threshold | n_parameters | best_epoch | train_seconds | roc_auc | f1 | recall | precision | pr_auc | accuracy | balanced_accuracy | mcc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | wide_deep | 0.6533 | 11672 | 8 | 12.3800 | 0.9370 | 0.3448 | 0.7576 | 0.2232 | 0.3585 | 0.9071 | 0.8348 | 0.3790 |
| 2 | mlp | 0.6021 | 3777 | 9 | 18.2200 | 0.9321 | 0.3086 | 0.7576 | 0.1938 | 0.3219 | 0.8905 | 0.8263 | 0.3473 |
| 3 | deep_fc | 0.6381 | 50113 | 9 | 16.3500 | 0.9309 | 0.3059 | 0.7879 | 0.1898 | 0.3388 | 0.8847 | 0.8379 | 0.3506 |
| 4 | residual | 0.4215 | 103937 | 12 | 21.6700 | 0.9277 | 0.2626 | 0.7879 | 0.1576 | 0.3287 | 0.8573 | 0.8237 | 0.3110 |

## Best model

- **Network**: wide_deep
- **Best epoch**: 8
- **Test ROC-AUC**: 0.9370
- **Threshold**: 0.653 (youden)

## Figures

- `reports/deep_learning/figures/loss_mlp.png`
- `reports/deep_learning/figures/accuracy_mlp.png`
- `reports/deep_learning/figures/lr_mlp.png`
- `reports/deep_learning/figures/roc_mlp.png`
- `reports/deep_learning/figures/pr_mlp.png`
- `reports/deep_learning/figures/confusion_mlp.png`
- `reports/deep_learning/figures/calibration_mlp.png`
- `reports/deep_learning/figures/pred_dist_mlp.png`
- `reports/deep_learning/figures/importance_mlp.png`
- `reports/deep_learning/figures/loss_deep_fc.png`
- `reports/deep_learning/figures/accuracy_deep_fc.png`
- `reports/deep_learning/figures/lr_deep_fc.png`
- `reports/deep_learning/figures/roc_deep_fc.png`
- `reports/deep_learning/figures/pr_deep_fc.png`
- `reports/deep_learning/figures/confusion_deep_fc.png`
- `reports/deep_learning/figures/calibration_deep_fc.png`
- `reports/deep_learning/figures/pred_dist_deep_fc.png`
- `reports/deep_learning/figures/importance_deep_fc.png`
- `reports/deep_learning/figures/loss_residual.png`
- `reports/deep_learning/figures/accuracy_residual.png`
- `reports/deep_learning/figures/lr_residual.png`
- `reports/deep_learning/figures/roc_residual.png`
- `reports/deep_learning/figures/pr_residual.png`
- `reports/deep_learning/figures/confusion_residual.png`
- `reports/deep_learning/figures/calibration_residual.png`
- `reports/deep_learning/figures/pred_dist_residual.png`
- `reports/deep_learning/figures/importance_residual.png`
- `reports/deep_learning/figures/loss_wide_deep.png`
- `reports/deep_learning/figures/accuracy_wide_deep.png`
- `reports/deep_learning/figures/lr_wide_deep.png`
- `reports/deep_learning/figures/roc_wide_deep.png`
- `reports/deep_learning/figures/pr_wide_deep.png`
- `reports/deep_learning/figures/confusion_wide_deep.png`
- `reports/deep_learning/figures/calibration_wide_deep.png`
- `reports/deep_learning/figures/pred_dist_wide_deep.png`
- `reports/deep_learning/figures/importance_wide_deep.png`
- `reports/deep_learning/figures/comparison_roc_auc.png`
- `reports/deep_learning/figures/comparison_f1.png`
- `reports/deep_learning/figures/comparison_recall.png`
- `reports/deep_learning/figures/comparison_pr_auc.png`
