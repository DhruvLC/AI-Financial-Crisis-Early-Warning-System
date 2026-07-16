# Best Model Report

Generated: 2026-07-16T08:46:23.831913+00:00
Dataset: feature store v001 (target `Bankrupt?`)

## extra_trees

- Dataset version: v001
- Threshold: 0.190 (youden)
- Training time: 15.49s
- Hyperparameters: `{'n_estimators': 400, 'max_depth': None, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'n_jobs': -1}`

### val
| Metric | Value |
|---|---|
| accuracy | 0.9326 |
| precision | 0.3000 |
| recall | 0.8182 |
| f1 | 0.4390 |
| roc_auc | 0.9359 |
| pr_auc | 0.4572 |
| balanced_accuracy | 0.8773 |
| mcc | 0.4706 |
| cohen_kappa | 0.4112 |
| log_loss | 0.0923 |
| brier_score | 0.0250 |

### test
| Metric | Value |
|---|---|
| accuracy | 0.9179 |
| precision | 0.2475 |
| recall | 0.7576 |
| f1 | 0.3731 |
| roc_auc | 0.9395 |
| pr_auc | 0.4380 |
| balanced_accuracy | 0.8404 |
| mcc | 0.4032 |
| cohen_kappa | 0.3411 |
| log_loss | 0.0984 |
| brier_score | 0.0280 |

### Top features

| feature | importance |
|---|---|
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0908 |
| pca__1 | 0.0861 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0829 |
| Borrowing dependency | 0.0822 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0799 |
| Total debt/Total net worth | 0.0707 |
| Debt ratio % | 0.0669 |
| Net Income to Total Assets | 0.0546 |
| ROA(B) before interest and depreciation after tax | 0.0508 |
| Retained Earnings to Total Assets | 0.0478 |
| Total income/Total expense | 0.0392 |
| log__Net Value Growth Rate | 0.0367 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0300 |
| Non-industry income and expenditure/revenue | 0.0279 |
| Allocation rate per person | 0.0235 |
