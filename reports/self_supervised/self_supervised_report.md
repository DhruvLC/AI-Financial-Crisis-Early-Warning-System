# Self-Supervised Learning Report

Generated: 2026-07-19T07:25:12.514874+00:00
Dataset: feature store v001 (target `Bankrupt?`, 22 features)

## Training summary

| Model | Status | Params | Device | Epochs | Best epoch | Train (s) | Threshold |
|---|---|---|---|---|---|---|---|
| mlp | trained | 47,808 | mps | 30 | 26 | 11.84 | 0.623 (youden) |
| residual | trained | 420,672 | mps | 30 | 26 | 13.96 | 0.716 (youden) |
| transformer | trained | 21,376 | mps | 18 | 3 | 16.87 | 0.573 (youden) |

## mlp

### Architecture
`{'encoder': 'mlp', 'n_features': 22, 'hidden_dims': [256, 128], 'embedding_dim': 64, 'activation': 'relu', 'dropout': 0.1, 'batch_norm': True}`

### Hyperparameters
`{'data': {'batch_size': 256}, 'projection_head': {'hidden_dims': [128], 'projection_dim': 64, 'activation': 'relu', 'batch_norm': True}, 'loss': {'name': 'nt_xent', 'temperature': 0.5}, 'augmentations': [{'name': 'feature_masking', 'ratio': 0.15}, {'name': 'gaussian_noise', 'sigma': 0.1}], 'optimizer': {'name': 'adamw', 'lr': 0.001, 'weight_decay': 0.0001}, 'scheduler': {'name': 'cosine'}, 'training': {'epochs': 30, 'device': 'auto', 'mixed_precision': True, 'gradient_clip': 1.0, 'log_every': 5}}`

### Metrics — val (threshold 0.623)
| Metric | Value |
|---|---|
| accuracy | 0.9013 |
| precision | 0.2258 |
| recall | 0.8485 |
| f1 | 0.3567 |
| roc_auc | 0.9340 |
| pr_auc | 0.3848 |
| balanced_accuracy | 0.8758 |
| mcc | 0.4068 |
| cohen_kappa | 0.3221 |
| log_loss | 0.2994 |
| brier_score | 0.0964 |

Confusion: TN=894 FP=96 FN=5 TP=28

### Metrics — test (threshold 0.623)
| Metric | Value |
|---|---|
| accuracy | 0.8886 |
| precision | 0.1955 |
| recall | 0.7879 |
| f1 | 0.3133 |
| roc_auc | 0.9337 |
| pr_auc | 0.3010 |
| balanced_accuracy | 0.8399 |
| mcc | 0.3571 |
| cohen_kappa | 0.2758 |
| log_loss | 0.3285 |
| brier_score | 0.1035 |

Confusion: TN=883 FP=107 FN=7 TP=26

## residual

### Architecture
`{'encoder': 'residual', 'n_features': 22, 'width': 256, 'n_blocks': 3, 'embedding_dim': 64, 'activation': 'relu', 'dropout': 0.1}`

### Hyperparameters
`{'data': {'batch_size': 256}, 'projection_head': {'hidden_dims': [128], 'projection_dim': 64, 'activation': 'relu', 'batch_norm': True}, 'loss': {'name': 'nt_xent', 'temperature': 0.5}, 'augmentations': [{'name': 'feature_masking', 'ratio': 0.15}, {'name': 'gaussian_noise', 'sigma': 0.1}], 'optimizer': {'name': 'adamw', 'lr': 0.001, 'weight_decay': 0.0001}, 'scheduler': {'name': 'cosine'}, 'training': {'epochs': 30, 'device': 'auto', 'mixed_precision': True, 'gradient_clip': 1.0, 'log_every': 5}}`

### Metrics — val (threshold 0.716)
| Metric | Value |
|---|---|
| accuracy | 0.9218 |
| precision | 0.2673 |
| recall | 0.8182 |
| f1 | 0.4030 |
| roc_auc | 0.9246 |
| pr_auc | 0.4005 |
| balanced_accuracy | 0.8717 |
| mcc | 0.4403 |
| cohen_kappa | 0.3725 |
| log_loss | 0.3130 |
| brier_score | 0.1015 |

Confusion: TN=916 FP=74 FN=6 TP=27

### Metrics — test (threshold 0.716)
| Metric | Value |
|---|---|
| accuracy | 0.9101 |
| precision | 0.2294 |
| recall | 0.7576 |
| f1 | 0.3521 |
| roc_auc | 0.9376 |
| pr_auc | 0.3553 |
| balanced_accuracy | 0.8364 |
| mcc | 0.3852 |
| cohen_kappa | 0.3184 |
| log_loss | 0.3234 |
| brier_score | 0.1039 |

Confusion: TN=906 FP=84 FN=8 TP=25

## transformer

### Architecture
`{'encoder': 'transformer', 'n_features': 22, 'embed_dim': 32, 'n_heads': 4, 'n_layers': 2, 'ff_dim': 64, 'dropout': 0.1, 'embedding_dim': 64, 'positional_embedding': True}`

### Hyperparameters
`{'data': {'batch_size': 256}, 'projection_head': {'hidden_dims': [128], 'projection_dim': 64, 'activation': 'relu', 'batch_norm': True}, 'loss': {'name': 'nt_xent', 'temperature': 0.5}, 'augmentations': [{'name': 'feature_masking', 'ratio': 0.15}, {'name': 'gaussian_noise', 'sigma': 0.1}], 'optimizer': {'name': 'adamw', 'lr': 0.001, 'weight_decay': 0.0001}, 'scheduler': {'name': 'cosine'}, 'training': {'epochs': 30, 'device': 'auto', 'mixed_precision': True, 'gradient_clip': 1.0, 'log_every': 5}}`

### Metrics — val (threshold 0.573)
| Metric | Value |
|---|---|
| accuracy | 0.8768 |
| precision | 0.1921 |
| recall | 0.8788 |
| f1 | 0.3152 |
| roc_auc | 0.9358 |
| pr_auc | 0.3630 |
| balanced_accuracy | 0.8778 |
| mcc | 0.3764 |
| cohen_kappa | 0.2769 |
| log_loss | 0.3189 |
| brier_score | 0.1031 |

Confusion: TN=868 FP=122 FN=4 TP=29

### Metrics — test (threshold 0.573)
| Metric | Value |
|---|---|
| accuracy | 0.8749 |
| precision | 0.1854 |
| recall | 0.8485 |
| f1 | 0.3043 |
| roc_auc | 0.9385 |
| pr_auc | 0.3229 |
| balanced_accuracy | 0.8621 |
| mcc | 0.3608 |
| cohen_kappa | 0.2655 |
| log_loss | 0.3384 |
| brier_score | 0.1090 |

Confusion: TN=867 FP=123 FN=5 TP=28

## Leaderboard (test)

| rank | model | embedding_dim | threshold | n_parameters | best_epoch | train_seconds | roc_auc | f1 | recall | precision | pr_auc | accuracy | balanced_accuracy | mcc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | transformer | 64 | 0.5726 | 21376 | 3 | 16.8700 | 0.9385 | 0.3043 | 0.8485 | 0.1854 | 0.3229 | 0.8749 | 0.8621 | 0.3608 |
| 2 | residual | 64 | 0.7155 | 420672 | 26 | 13.9600 | 0.9376 | 0.3521 | 0.7576 | 0.2294 | 0.3553 | 0.9101 | 0.8364 | 0.3852 |
| 3 | mlp | 64 | 0.6229 | 47808 | 26 | 11.8400 | 0.9337 | 0.3133 | 0.7879 | 0.1955 | 0.3010 | 0.8886 | 0.8399 | 0.3571 |

## Best model

- **Network**: transformer
- **Best epoch**: 3
- **Test ROC-AUC**: 0.9385
- **Threshold**: 0.573 (youden)

## Figures

- `reports/self_supervised/figures/loss_mlp.png`
- `reports/self_supervised/figures/lr_mlp.png`
- `reports/self_supervised/figures/projection_pca_mlp.png`
- `reports/self_supervised/figures/projection_tsne_mlp.png`
- `reports/self_supervised/figures/similarity_mlp.png`
- `reports/self_supervised/figures/embedding_dist_mlp.png`
- `reports/self_supervised/figures/loss_residual.png`
- `reports/self_supervised/figures/lr_residual.png`
- `reports/self_supervised/figures/projection_pca_residual.png`
- `reports/self_supervised/figures/projection_tsne_residual.png`
- `reports/self_supervised/figures/similarity_residual.png`
- `reports/self_supervised/figures/embedding_dist_residual.png`
- `reports/self_supervised/figures/loss_transformer.png`
- `reports/self_supervised/figures/lr_transformer.png`
- `reports/self_supervised/figures/projection_pca_transformer.png`
- `reports/self_supervised/figures/projection_tsne_transformer.png`
- `reports/self_supervised/figures/similarity_transformer.png`
- `reports/self_supervised/figures/embedding_dist_transformer.png`
- `reports/self_supervised/figures/comparison_roc_auc.png`
- `reports/self_supervised/figures/comparison_f1.png`
- `reports/self_supervised/figures/comparison_recall.png`
- `reports/self_supervised/figures/comparison_pr_auc.png`

## Augmentation pipeline

| Augmentation | Parameters |
|---|---|
| feature_masking | `{'ratio': 0.15}` |
| gaussian_noise | `{'sigma': 0.1}` |

## Contrastive loss

`{'name': 'nt_xent', 'temperature': 0.5}`

## Representations — mlp

| Split | Path | Samples | Dim | Mean L2 norm |
|---|---|---|---|---|
| train | `models/self_supervised/representations/mlp_train.parquet` | 4773 | 64 | 6.346 |
| val | `models/self_supervised/representations/mlp_val.parquet` | 1023 | 64 | 6.432 |
| test | `models/self_supervised/representations/mlp_test.parquet` | 1023 | 64 | 6.534 |

### KNN probe — mlp

| Split | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
| val | 0.8788 | 0.1951 | 0.9677 |
| test | 0.8530 | 0.2128 | 0.9638 |

## Representations — residual

| Split | Path | Samples | Dim | Mean L2 norm |
|---|---|---|---|---|
| train | `models/self_supervised/representations/residual_train.parquet` | 4773 | 64 | 6.501 |
| val | `models/self_supervised/representations/residual_val.parquet` | 1023 | 64 | 6.511 |
| test | `models/self_supervised/representations/residual_test.parquet` | 1023 | 64 | 6.595 |

### KNN probe — residual

| Split | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
| val | 0.8688 | 0.2439 | 0.9697 |
| test | 0.8692 | 0.1667 | 0.9609 |

## Representations — transformer

| Split | Path | Samples | Dim | Mean L2 norm |
|---|---|---|---|---|
| train | `models/self_supervised/representations/transformer_train.parquet` | 4773 | 64 | 5.320 |
| val | `models/self_supervised/representations/transformer_val.parquet` | 1023 | 64 | 5.334 |
| test | `models/self_supervised/representations/transformer_test.parquet` | 1023 | 64 | 5.315 |

### KNN probe — transformer

| Split | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
| val | 0.8485 | 0.1951 | 0.9677 |
| test | 0.8723 | 0.2128 | 0.9638 |
