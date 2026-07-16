# Training Report

Generated: 2026-07-16T08:46:23.829420+00:00
Dataset: feature store v001 (target `Bankrupt?`)

| Model | Status | Train (s) | CV mean | Threshold |
|---|---|---|---|---|
| logistic_regression | trained | 6.33 | 0.9266 | 0.399 (youden) |
| decision_tree | trained | 0.21 | 0.8119 | 0.830 (youden) |
| random_forest | trained | 8.81 | 0.9207 | 0.041 (youden) |
| extra_trees | trained | 15.49 | 0.9239 | 0.190 (youden) |
| xgboost | trained | 1.92 | 0.9076 | 0.003 (youden) |
| svm | trained | 1336.33 | 0.8951 | 0.028 (youden) |
| knn | trained | 193.23 | 0.8756 | 0.056 (youden) |
| naive_bayes | trained | 6.91 | 0.9207 | 0.000 (youden) |
| mlp | trained | 8.09 | 0.8406 | 0.066 (youden) |
