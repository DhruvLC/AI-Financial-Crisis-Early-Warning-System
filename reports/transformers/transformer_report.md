# Transformer Models Report

Generated: 2026-07-18T12:01:04.395644+00:00
Dataset: feature store v001 (target `Bankrupt?`, 22 features)

## Training summary

| Model | Status | Params | Device | Epochs | Best epoch | Train (s) | Threshold |
|---|---|---|---|---|---|---|---|
| ft_transformer | trained | 103,489 | mps | 13 | 3 | 396.34 | 0.513 (youden) |
| tab_transformer | trained | 128,461 | cpu | 13 | 3 | 496.30 | 0.469 (youden) |
| tabular_encoder | trained | 71,361 | cpu | 24 | 14 | 318.17 | 0.676 (youden) |

## ft_transformer

### Architecture
`{'network': 'ft_transformer', 'n_features': 22, 'embed_dim': 64, 'n_heads': 8, 'n_layers': 3, 'ff_dim': 128, 'dropout': 0.1}`

### Hyperparameters
`{'data': {'batch_size': 128}, 'optimizer': {'name': 'adamw', 'lr': 0.0005, 'weight_decay': 0.0001}, 'scheduler': {'name': 'cosine'}, 'loss': {'name': 'weighted_bce'}, 'training': {'epochs': 60, 'device': 'auto', 'mixed_precision': True, 'gradient_clip': 1.0, 'log_every': 5}}`

### Metrics — val (threshold 0.513)
| Metric | Value |
|---|---|
| accuracy | 0.6246 |
| precision | 0.0751 |
| recall | 0.9394 |
| f1 | 0.1390 |
| roc_auc | 0.8086 |
| pr_auc | 0.1270 |
| balanced_accuracy | 0.7768 |
| mcc | 0.1993 |
| cohen_kappa | 0.0843 |
| log_loss | 0.6707 |
| brier_score | 0.2389 |

Confusion: TN=608 FP=382 FN=2 TP=31

### Metrics — test (threshold 0.513)
| Metric | Value |
|---|---|
| accuracy | 0.6256 |
| precision | 0.0752 |
| recall | 0.9394 |
| f1 | 0.1393 |
| roc_auc | 0.8242 |
| pr_auc | 0.0939 |
| balanced_accuracy | 0.7773 |
| mcc | 0.1998 |
| cohen_kappa | 0.0847 |
| log_loss | 0.6707 |
| brier_score | 0.2389 |

Confusion: TN=609 FP=381 FN=2 TP=31

### Top features (permutation)

| feature | importance | std |
|---|---|---|
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0204 | 0.0105 |
| Non-industry income and expenditure/revenue | 0.0180 | 0.0052 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0143 | 0.0047 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0136 | 0.0057 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0103 | 0.0051 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0093 | 0.0015 |
| Total income/Total expense | 0.0037 | 0.0047 |
| Allocation rate per person | -0.0002 | 0.0055 |
| pca__3 | -0.0026 | 0.0020 |
| Total debt/Total net worth | -0.0032 | 0.0026 |
| pca__5 | -0.0038 | 0.0029 |
| pca__4 | -0.0050 | 0.0064 |
| Retained Earnings to Total Assets | -0.0098 | 0.0008 |
| log__Net Value Growth Rate | -0.0142 | 0.0025 |
| pca__6 | -0.0178 | 0.0039 |

### Top features (SHAP)

| Feature | mean abs SHAP |
|---|---|
| pca__2 | 0.0121 |
| Allocation rate per person | 0.0111 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0093 |
| Total income/Total expense | 0.0082 |
| pca__7 | 0.0073 |
| pca__5 | 0.0069 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0062 |
| ROA(B) before interest and depreciation after tax | 0.0054 |
| log__Net Value Growth Rate | 0.0053 |
| Borrowing dependency | 0.0049 |
| Total debt/Total net worth | 0.0041 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0034 |
| Non-industry income and expenditure/revenue | 0.0032 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0030 |
| pca__4 | 0.0029 |

## tab_transformer

### Architecture
`{'network': 'tab_transformer', 'n_features': 22, 'embed_dim': 32, 'n_heads': 8, 'n_layers': 3, 'ff_dim': 64, 'dropout': 0.1, 'mlp_hidden': [128, 64]}`

### Hyperparameters
`{'data': {'batch_size': 128}, 'optimizer': {'name': 'adamw', 'lr': 0.0005, 'weight_decay': 0.0001}, 'scheduler': {'name': 'cosine'}, 'loss': {'name': 'weighted_bce'}, 'training': {'epochs': 60, 'device': 'auto', 'mixed_precision': True, 'gradient_clip': 1.0, 'log_every': 5}}`

### Metrics — val (threshold 0.469)
| Metric | Value |
|---|---|
| accuracy | 0.8905 |
| precision | 0.2158 |
| recall | 0.9091 |
| f1 | 0.3488 |
| roc_auc | 0.9423 |
| pr_auc | 0.3874 |
| balanced_accuracy | 0.8995 |
| mcc | 0.4120 |
| cohen_kappa | 0.3130 |
| log_loss | 0.3499 |
| brier_score | 0.0977 |

Confusion: TN=881 FP=109 FN=3 TP=30

### Metrics — test (threshold 0.469)
| Metric | Value |
|---|---|
| accuracy | 0.8827 |
| precision | 0.1915 |
| recall | 0.8182 |
| f1 | 0.3103 |
| roc_auc | 0.9330 |
| pr_auc | 0.3368 |
| balanced_accuracy | 0.8515 |
| mcc | 0.3603 |
| cohen_kappa | 0.2723 |
| log_loss | 0.3726 |
| brier_score | 0.1039 |

Confusion: TN=876 FP=114 FN=6 TP=27

### Top features (permutation)

| feature | importance | std |
|---|---|---|
| pca__1 | 0.0052 | 0.0022 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0050 | 0.0037 |
| Retained Earnings to Total Assets | 0.0046 | 0.0013 |
| Net Income to Total Assets | 0.0039 | 0.0011 |
| pca__6 | 0.0037 | 0.0012 |
| Non-industry income and expenditure/revenue | 0.0031 | 0.0010 |
| ROA(B) before interest and depreciation after tax | 0.0027 | 0.0009 |
| Allocation rate per person | 0.0027 | 0.0017 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0018 | 0.0024 |
| pca__2 | 0.0016 | 0.0003 |
| Total income/Total expense | 0.0015 | 0.0003 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0012 | 0.0008 |
| pca__5 | 0.0008 | 0.0007 |
| pca__3 | 0.0007 | 0.0006 |
| log__Net Value Growth Rate | 0.0005 | 0.0003 |

### Top features (SHAP)

| Feature | mean abs SHAP |
|---|---|
| Borrowing dependency | 0.0269 |
| pca__1 | 0.0257 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0216 |
| Allocation rate per person | 0.0208 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0184 |
| Net Income to Total Assets | 0.0152 |
| Debt ratio % | 0.0138 |
| Retained Earnings to Total Assets | 0.0129 |
| pca__2 | 0.0117 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0110 |
| Total debt/Total net worth | 0.0110 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0086 |
| Non-industry income and expenditure/revenue | 0.0079 |
| pca__6 | 0.0075 |
| pca__3 | 0.0068 |

## tabular_encoder

### Architecture
`{'network': 'tabular_encoder', 'n_features': 22, 'embed_dim': 64, 'n_heads': 4, 'n_layers': 2, 'ff_dim': 128, 'dropout': 0.1, 'positional_embedding': True}`

### Hyperparameters
`{'data': {'batch_size': 128}, 'optimizer': {'name': 'adamw', 'lr': 0.0005, 'weight_decay': 0.0001}, 'scheduler': {'name': 'cosine'}, 'loss': {'name': 'weighted_bce'}, 'training': {'epochs': 60, 'device': 'auto', 'mixed_precision': True, 'gradient_clip': 1.0, 'log_every': 5}}`

### Metrics — val (threshold 0.676)
| Metric | Value |
|---|---|
| accuracy | 0.9159 |
| precision | 0.2613 |
| recall | 0.8788 |
| f1 | 0.4028 |
| roc_auc | 0.9437 |
| pr_auc | 0.4243 |
| balanced_accuracy | 0.8980 |
| mcc | 0.4522 |
| cohen_kappa | 0.3715 |
| log_loss | 0.2803 |
| brier_score | 0.0899 |

Confusion: TN=908 FP=82 FN=4 TP=29

### Metrics — test (threshold 0.676)
| Metric | Value |
|---|---|
| accuracy | 0.8954 |
| precision | 0.2109 |
| recall | 0.8182 |
| f1 | 0.3354 |
| roc_auc | 0.9393 |
| pr_auc | 0.4112 |
| balanced_accuracy | 0.8581 |
| mcc | 0.3824 |
| cohen_kappa | 0.2995 |
| log_loss | 0.3120 |
| brier_score | 0.1009 |

Confusion: TN=889 FP=101 FN=6 TP=27

### Top features (permutation)

| feature | importance | std |
|---|---|---|
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0206 | 0.0084 |
| pca__6 | 0.0174 | 0.0049 |
| pca__1 | 0.0147 | 0.0042 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0120 | 0.0036 |
| ROA(B) before interest and depreciation after tax | 0.0090 | 0.0036 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0081 | 0.0043 |
| Retained Earnings to Total Assets | 0.0045 | 0.0007 |
| Total income/Total expense | 0.0041 | 0.0022 |
| Net Income to Total Assets | 0.0041 | 0.0007 |
| pca__5 | 0.0030 | 0.0033 |
| pca__7 | 0.0028 | 0.0005 |
| Allocation rate per person | 0.0017 | 0.0025 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0007 | 0.0007 |
| log__Net Value Growth Rate | 0.0001 | 0.0005 |
| Total debt/Total net worth | 0.0001 | 0.0021 |

### Top features (SHAP)

| Feature | mean abs SHAP |
|---|---|
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0378 |
| Allocation rate per person | 0.0295 |
| pca__1 | 0.0259 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0238 |
| Borrowing dependency | 0.0191 |
| Total debt/Total net worth | 0.0178 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0175 |
| pca__5 | 0.0135 |
| pca__6 | 0.0118 |
| Debt ratio % | 0.0117 |
| Retained Earnings to Total Assets | 0.0110 |
| ROA(B) before interest and depreciation after tax | 0.0100 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0100 |
| Total income/Total expense | 0.0086 |
| Net Income to Total Assets | 0.0078 |

## Leaderboard (test)

| rank | model | threshold | n_parameters | best_epoch | train_seconds | roc_auc | f1 | recall | precision | pr_auc | accuracy | balanced_accuracy | mcc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tabular_encoder | 0.6759 | 71361 | 14 | 318.1700 | 0.9393 | 0.3354 | 0.8182 | 0.2109 | 0.4112 | 0.8954 | 0.8581 | 0.3824 |
| 2 | tab_transformer | 0.4695 | 128461 | 3 | 496.3000 | 0.9330 | 0.3103 | 0.8182 | 0.1915 | 0.3368 | 0.8827 | 0.8515 | 0.3603 |
| 3 | ft_transformer | 0.5126 | 103489 | 3 | 396.3400 | 0.8242 | 0.1393 | 0.9394 | 0.0752 | 0.0939 | 0.6256 | 0.7773 | 0.1998 |

## Best model

- **Network**: tabular_encoder
- **Best epoch**: 14
- **Test ROC-AUC**: 0.9393
- **Threshold**: 0.676 (youden)

## Figures

- `reports/transformers/figures/loss_ft_transformer.png`
- `reports/transformers/figures/accuracy_ft_transformer.png`
- `reports/transformers/figures/lr_ft_transformer.png`
- `reports/transformers/figures/roc_ft_transformer.png`
- `reports/transformers/figures/pr_ft_transformer.png`
- `reports/transformers/figures/confusion_ft_transformer.png`
- `reports/transformers/figures/calibration_ft_transformer.png`
- `reports/transformers/figures/pred_dist_ft_transformer.png`
- `reports/transformers/figures/importance_ft_transformer.png`
- `reports/transformers/figures/attention_heatmap_ft_transformer.png`
- `reports/transformers/figures/attention_features_ft_transformer.png`
- `reports/transformers/figures/loss_tab_transformer.png`
- `reports/transformers/figures/accuracy_tab_transformer.png`
- `reports/transformers/figures/lr_tab_transformer.png`
- `reports/transformers/figures/roc_tab_transformer.png`
- `reports/transformers/figures/pr_tab_transformer.png`
- `reports/transformers/figures/confusion_tab_transformer.png`
- `reports/transformers/figures/calibration_tab_transformer.png`
- `reports/transformers/figures/pred_dist_tab_transformer.png`
- `reports/transformers/figures/importance_tab_transformer.png`
- `reports/transformers/figures/attention_heatmap_tab_transformer.png`
- `reports/transformers/figures/attention_features_tab_transformer.png`
- `reports/transformers/figures/loss_tabular_encoder.png`
- `reports/transformers/figures/accuracy_tabular_encoder.png`
- `reports/transformers/figures/lr_tabular_encoder.png`
- `reports/transformers/figures/roc_tabular_encoder.png`
- `reports/transformers/figures/pr_tabular_encoder.png`
- `reports/transformers/figures/confusion_tabular_encoder.png`
- `reports/transformers/figures/calibration_tabular_encoder.png`
- `reports/transformers/figures/pred_dist_tabular_encoder.png`
- `reports/transformers/figures/importance_tabular_encoder.png`
- `reports/transformers/figures/attention_heatmap_tabular_encoder.png`
- `reports/transformers/figures/attention_features_tabular_encoder.png`
- `reports/transformers/figures/comparison_roc_auc.png`
- `reports/transformers/figures/comparison_f1.png`
- `reports/transformers/figures/comparison_recall.png`
- `reports/transformers/figures/comparison_pr_auc.png`
- `reports/transformers/figures/cross_family_roc_auc.png`

## Attention analysis — ft_transformer

Aggregated over 256 validation samples (mean over heads, layers, and queries).

| Layer | Mean attention entropy |
|---|---|
| layer_1 | 3.1231 |
| layer_2 | 3.1334 |
| layer_3 | 3.1352 |

### Top features by attention received

| Feature | Mean attention |
|---|---|
| pca__7 | 0.0443 |
| pca__2 | 0.0442 |
| Borrowing dependency | 0.0441 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0440 |
| Total debt/Total net worth | 0.0439 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0438 |
| pca__1 | 0.0437 |
| pca__3 | 0.0437 |
| Net Income to Total Assets | 0.0437 |
| pca__6 | 0.0436 |
| Allocation rate per person | 0.0436 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0436 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0435 |
| Retained Earnings to Total Assets | 0.0434 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0434 |

## Attention analysis — tab_transformer

Aggregated over 256 validation samples (mean over heads, layers, and queries).

| Layer | Mean attention entropy |
|---|---|
| layer_1 | 3.0759 |
| layer_2 | 3.0903 |
| layer_3 | 3.0909 |

### Top features by attention received

| Feature | Mean attention |
|---|---|
| Borrowing dependency | 0.0469 |
| Allocation rate per person | 0.0467 |
| Debt ratio % | 0.0466 |
| pca__2 | 0.0465 |
| pca__3 | 0.0464 |
| pca__5 | 0.0461 |
| ROA(B) before interest and depreciation after tax | 0.0458 |
| pca__1 | 0.0458 |
| pca__6 | 0.0455 |
| Net Income to Total Assets | 0.0454 |
| log__Net Value Growth Rate | 0.0454 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0453 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0451 |
| pca__7 | 0.0451 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0450 |

## Attention analysis — tabular_encoder

Aggregated over 256 validation samples (mean over heads, layers, and queries).

| Layer | Mean attention entropy |
|---|---|
| layer_1 | 2.9291 |
| layer_2 | 3.0177 |

### Top features by attention received

| Feature | Mean attention |
|---|---|
| Allocation rate per person | 0.0696 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0606 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0582 |
| pca__5 | 0.0542 |
| pca__1 | 0.0536 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0512 |
| Retained Earnings to Total Assets | 0.0508 |
| Borrowing dependency | 0.0485 |
| Total debt/Total net worth | 0.0447 |
| pca__6 | 0.0436 |
| Debt ratio % | 0.0418 |
| pca__2 | 0.0417 |
| ROA(B) before interest and depreciation after tax | 0.0413 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0400 |
| Non-industry income and expenditure/revenue | 0.0391 |

## Comparison with previous ML and DL models

Test-set metrics of every registered model across the classical ML, deep-learning, and transformer stages:

| family | model | roc_auc | f1 | recall | pr_auc |
|---|---|---|---|---|---|
| machine_learning | extra_trees | 0.9395 | 0.3731 | 0.7576 | 0.4380 |
| transformers | tabular_encoder | 0.9393 | 0.3354 | 0.8182 | 0.4112 |
| machine_learning | random_forest | 0.9378 | 0.2456 | 0.8485 | 0.4054 |
| deep_learning | wide_deep | 0.9370 | 0.3448 | 0.7576 | 0.3585 |
| machine_learning | xgboost | 0.9349 | 0.2500 | 0.8485 | 0.3970 |
| transformers | tab_transformer | 0.9330 | 0.3103 | 0.8182 | 0.3368 |
| deep_learning | mlp | 0.9321 | 0.3086 | 0.7576 | 0.3219 |
| deep_learning | deep_fc | 0.9309 | 0.3059 | 0.7879 | 0.3388 |
| deep_learning | residual | 0.9277 | 0.2626 | 0.7879 | 0.3287 |
| machine_learning | logistic_regression | 0.9241 | 0.2267 | 0.8485 | 0.2831 |
| machine_learning | naive_bayes | 0.9152 | 0.2640 | 0.7879 | 0.2352 |
| machine_learning | svm | 0.9100 | 0.2532 | 0.9091 | 0.2697 |
| machine_learning | mlp | 0.9046 | 0.2905 | 0.7879 | 0.2924 |
| machine_learning | knn | 0.8534 | 0.2751 | 0.7879 | 0.3048 |
| transformers | ft_transformer | 0.8242 | 0.1393 | 0.9394 | 0.0939 |
| machine_learning | decision_tree | 0.8109 | 0.3111 | 0.6364 | 0.2263 |
