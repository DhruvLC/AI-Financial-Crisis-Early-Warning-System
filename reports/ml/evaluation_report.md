# Evaluation Report

Generated: 2026-07-16T08:46:23.829512+00:00
Dataset: feature store v001 (target `Bankrupt?`)


## logistic_regression

### val (threshold 0.399)
| Metric | Value |
|---|---|
| accuracy | 0.8240 |
| precision | 0.1483 |
| recall | 0.9394 |
| f1 | 0.2562 |
| roc_auc | 0.9385 |
| pr_auc | 0.3879 |
| balanced_accuracy | 0.8798 |
| mcc | 0.3329 |
| cohen_kappa | 0.2123 |
| log_loss | 0.3271 |
| brier_score | 0.1002 |

Confusion: TN=812 FP=178 FN=2 TP=31

### test (threshold 0.399)
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

Confusion: TN=804 FP=186 FN=5 TP=28

## decision_tree

### val (threshold 0.830)
| Metric | Value |
|---|---|
| accuracy | 0.9198 |
| precision | 0.2366 |
| recall | 0.6667 |
| f1 | 0.3492 |
| roc_auc | 0.8011 |
| pr_auc | 0.2470 |
| balanced_accuracy | 0.7975 |
| mcc | 0.3657 |
| cohen_kappa | 0.3167 |
| log_loss | 0.5829 |
| brier_score | 0.0747 |

Confusion: TN=919 FP=71 FN=11 TP=22

### test (threshold 0.830)
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

Confusion: TN=909 FP=81 FN=12 TP=21

## random_forest

### val (threshold 0.041)
| Metric | Value |
|---|---|
| accuracy | 0.8514 |
| precision | 0.1676 |
| recall | 0.9091 |
| f1 | 0.2830 |
| roc_auc | 0.9248 |
| pr_auc | 0.4075 |
| balanced_accuracy | 0.8793 |
| mcc | 0.3528 |
| cohen_kappa | 0.2417 |
| log_loss | 0.1167 |
| brier_score | 0.0239 |

Confusion: TN=841 FP=149 FN=3 TP=30

### test (threshold 0.041)
| Metric | Value |
|---|---|
| accuracy | 0.8319 |
| precision | 0.1436 |
| recall | 0.8485 |
| f1 | 0.2456 |
| roc_auc | 0.9378 |
| pr_auc | 0.4054 |
| balanced_accuracy | 0.8399 |
| mcc | 0.3058 |
| cohen_kappa | 0.2016 |
| log_loss | 0.0878 |
| brier_score | 0.0242 |

Confusion: TN=823 FP=167 FN=5 TP=28

## extra_trees

### val (threshold 0.190)
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

Confusion: TN=927 FP=63 FN=6 TP=27

### test (threshold 0.190)
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

Confusion: TN=914 FP=76 FN=8 TP=25

## xgboost

### val (threshold 0.003)
| Metric | Value |
|---|---|
| accuracy | 0.8524 |
| precision | 0.1648 |
| recall | 0.8788 |
| f1 | 0.2775 |
| roc_auc | 0.9313 |
| pr_auc | 0.4584 |
| balanced_accuracy | 0.8652 |
| mcc | 0.3419 |
| cohen_kappa | 0.2360 |
| log_loss | 0.1072 |
| brier_score | 0.0237 |

Confusion: TN=843 FP=147 FN=4 TP=29

### test (threshold 0.003)
| Metric | Value |
|---|---|
| accuracy | 0.8358 |
| precision | 0.1466 |
| recall | 0.8485 |
| f1 | 0.2500 |
| roc_auc | 0.9349 |
| pr_auc | 0.3970 |
| balanced_accuracy | 0.8419 |
| mcc | 0.3101 |
| cohen_kappa | 0.2063 |
| log_loss | 0.1154 |
| brier_score | 0.0285 |

Confusion: TN=827 FP=163 FN=5 TP=28

## svm

### val (threshold 0.028)
| Metric | Value |
|---|---|
| accuracy | 0.8240 |
| precision | 0.1449 |
| recall | 0.9091 |
| f1 | 0.2500 |
| roc_auc | 0.8988 |
| pr_auc | 0.2409 |
| balanced_accuracy | 0.8652 |
| mcc | 0.3212 |
| cohen_kappa | 0.2058 |
| log_loss | 0.1012 |
| brier_score | 0.0272 |

Confusion: TN=813 FP=177 FN=3 TP=30

### test (threshold 0.028)
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

Confusion: TN=816 FP=174 FN=3 TP=30

## knn

### val (threshold 0.056)
| Metric | Value |
|---|---|
| accuracy | 0.8592 |
| precision | 0.1636 |
| recall | 0.8182 |
| f1 | 0.2727 |
| roc_auc | 0.8661 |
| pr_auc | 0.3801 |
| balanced_accuracy | 0.8394 |
| mcc | 0.3261 |
| cohen_kappa | 0.2314 |
| log_loss | 0.2729 |
| brier_score | 0.0246 |

Confusion: TN=852 FP=138 FN=6 TP=27

### test (threshold 0.056)
| Metric | Value |
|---|---|
| accuracy | 0.8661 |
| precision | 0.1667 |
| recall | 0.7879 |
| f1 | 0.2751 |
| roc_auc | 0.8534 |
| pr_auc | 0.3048 |
| balanced_accuracy | 0.8283 |
| mcc | 0.3227 |
| cohen_kappa | 0.2344 |
| log_loss | 0.3090 |
| brier_score | 0.0264 |

Confusion: TN=860 FP=130 FN=7 TP=26

## naive_bayes

### val (threshold 0.000)
| Metric | Value |
|---|---|
| accuracy | 0.8641 |
| precision | 0.1768 |
| recall | 0.8788 |
| f1 | 0.2944 |
| roc_auc | 0.9345 |
| pr_auc | 0.3900 |
| balanced_accuracy | 0.8712 |
| mcc | 0.3575 |
| cohen_kappa | 0.2544 |
| log_loss | 1.2456 |
| brier_score | 0.0769 |

Confusion: TN=855 FP=135 FN=4 TP=29

### test (threshold 0.000)
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

Confusion: TN=852 FP=138 FN=7 TP=26

## mlp

### val (threshold 0.066)
| Metric | Value |
|---|---|
| accuracy | 0.8778 |
| precision | 0.1761 |
| recall | 0.7576 |
| f1 | 0.2857 |
| roc_auc | 0.8920 |
| pr_auc | 0.2957 |
| balanced_accuracy | 0.8197 |
| mcc | 0.3267 |
| cohen_kappa | 0.2463 |
| log_loss | 0.1012 |
| brier_score | 0.0261 |

Confusion: TN=873 FP=117 FN=8 TP=25

### test (threshold 0.066)
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

Confusion: TN=870 FP=120 FN=7 TP=26
