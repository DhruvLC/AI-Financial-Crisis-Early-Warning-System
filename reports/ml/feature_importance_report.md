# Feature Importance Report

Generated: 2026-07-16T08:46:23.832145+00:00
Dataset: feature store v001 (target `Bankrupt?`)


## logistic_regression — native

| feature | importance |
|---|---|
| Total income/Total expense | 0.7059 |
| Debt ratio % | 0.6449 |
| ROA(B) before interest and depreciation after tax | 0.5887 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.3426 |
| pca__7 | 0.3069 |
| pca__1 | 0.2830 |
| pca__2 | 0.2739 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.2660 |
| Borrowing dependency | 0.2612 |
| Net Income to Total Assets | 0.2461 |
| pca__4 | 0.1904 |
| Allocation rate per person | 0.1628 |
| Total debt/Total net worth | 0.1400 |
| Non-industry income and expenditure/revenue | 0.1196 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.1065 |

## logistic_regression — permutation

| feature | importance | std |
|---|---|---|
| pca__1 | 0.1507 | 0.0091 |
| ROA(B) before interest and depreciation after tax | 0.0387 | 0.0066 |
| Total income/Total expense | 0.0277 | 0.0101 |
| pca__2 | 0.0245 | 0.0065 |
| Debt ratio % | 0.0180 | 0.0035 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0155 | 0.0029 |
| pca__7 | 0.0035 | 0.0024 |
| Borrowing dependency | 0.0034 | 0.0007 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0030 | 0.0004 |
| Total debt/Total net worth | 0.0016 | 0.0012 |
| Non-industry income and expenditure/revenue | 0.0002 | 0.0010 |
| Allocation rate per person | 0.0002 | 0.0008 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0001 | 0.0005 |
| Retained Earnings to Total Assets | 0.0001 | 0.0001 |
| pca__6 | 0.0000 | 0.0000 |

## decision_tree — native

| feature | importance |
|---|---|
| pca__1 | 0.6674 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0723 |
| Allocation rate per person | 0.0509 |
| Non-industry income and expenditure/revenue | 0.0491 |
| Total income/Total expense | 0.0291 |
| Debt ratio % | 0.0205 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0135 |
| ROA(B) before interest and depreciation after tax | 0.0123 |
| pca__7 | 0.0123 |
| pca__4 | 0.0120 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0120 |
| pca__6 | 0.0114 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0107 |
| Borrowing dependency | 0.0102 |
| Retained Earnings to Total Assets | 0.0080 |

## decision_tree — permutation

| feature | importance | std |
|---|---|---|
| pca__1 | 0.2585 | 0.0231 |
| ROA(B) before interest and depreciation after tax | 0.1906 | 0.0353 |
| Debt ratio % | 0.1849 | 0.0128 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.1494 | 0.0199 |
| Non-industry income and expenditure/revenue | 0.1157 | 0.0320 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0645 | 0.0128 |
| log__Net Value Growth Rate | 0.0626 | 0.0252 |
| Retained Earnings to Total Assets | 0.0329 | 0.0150 |
| pca__6 | 0.0274 | 0.0058 |
| pca__4 | 0.0114 | 0.0077 |
| Borrowing dependency | 0.0030 | 0.0007 |
| Net Income to Total Assets | 0.0025 | 0.0008 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0000 | 0.0000 |
| Total debt/Total net worth | 0.0000 | 0.0000 |
| pca__2 | 0.0000 | 0.0000 |

## random_forest — native

| feature | importance |
|---|---|
| pca__1 | 0.1470 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.1174 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.1121 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0929 |
| Borrowing dependency | 0.0619 |
| Total debt/Total net worth | 0.0482 |
| Retained Earnings to Total Assets | 0.0433 |
| Debt ratio % | 0.0384 |
| Net Income to Total Assets | 0.0344 |
| Total income/Total expense | 0.0339 |
| log__Net Value Growth Rate | 0.0317 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0315 |
| ROA(B) before interest and depreciation after tax | 0.0285 |
| Allocation rate per person | 0.0260 |
| Non-industry income and expenditure/revenue | 0.0241 |

## random_forest — permutation

| feature | importance | std |
|---|---|---|
| pca__1 | 0.0206 | 0.0017 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0161 | 0.0027 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0135 | 0.0039 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0082 | 0.0080 |
| ROA(B) before interest and depreciation after tax | 0.0067 | 0.0054 |
| pca__6 | 0.0050 | 0.0012 |
| Non-industry income and expenditure/revenue | 0.0046 | 0.0014 |
| Retained Earnings to Total Assets | 0.0031 | 0.0022 |
| Total debt/Total net worth | 0.0013 | 0.0089 |
| Borrowing dependency | 0.0013 | 0.0070 |
| Allocation rate per person | 0.0003 | 0.0087 |
| pca__2 | -0.0000 | 0.0062 |
| pca__4 | -0.0005 | 0.0051 |
| pca__7 | -0.0045 | 0.0048 |
| pca__3 | -0.0047 | 0.0070 |

## extra_trees — native

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

## extra_trees — permutation

| feature | importance | std |
|---|---|---|
| pca__1 | 0.0111 | 0.0016 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0104 | 0.0019 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0096 | 0.0035 |
| pca__6 | 0.0054 | 0.0017 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0049 | 0.0014 |
| pca__3 | 0.0040 | 0.0016 |
| Non-industry income and expenditure/revenue | 0.0040 | 0.0011 |
| Borrowing dependency | 0.0040 | 0.0032 |
| ROA(B) before interest and depreciation after tax | 0.0036 | 0.0047 |
| Retained Earnings to Total Assets | 0.0036 | 0.0016 |
| pca__5 | 0.0028 | 0.0011 |
| Net Income to Total Assets | 0.0027 | 0.0009 |
| Total debt/Total net worth | 0.0018 | 0.0051 |
| Allocation rate per person | 0.0016 | 0.0038 |
| Total income/Total expense | 0.0009 | 0.0025 |

## xgboost — native

| feature | importance |
|---|---|
| pca__1 | 0.5302 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.1178 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0590 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0225 |
| Allocation rate per person | 0.0217 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0201 |
| log__Net Value Growth Rate | 0.0190 |
| Retained Earnings to Total Assets | 0.0171 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0163 |
| ROA(B) before interest and depreciation after tax | 0.0163 |
| pca__5 | 0.0161 |
| pca__7 | 0.0157 |
| Non-industry income and expenditure/revenue | 0.0150 |
| pca__4 | 0.0142 |
| Total debt/Total net worth | 0.0140 |

## xgboost — permutation

| feature | importance | std |
|---|---|---|
| ROA(B) before interest and depreciation after tax | 0.0334 | 0.0058 |
| pca__1 | 0.0317 | 0.0039 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0189 | 0.0045 |
| Non-industry income and expenditure/revenue | 0.0137 | 0.0007 |
| Retained Earnings to Total Assets | 0.0130 | 0.0029 |
| pca__6 | 0.0104 | 0.0031 |
| pca__2 | 0.0100 | 0.0035 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0057 | 0.0023 |
| Total debt/Total net worth | 0.0049 | 0.0026 |
| log__Net Value Growth Rate | 0.0042 | 0.0048 |
| Total income/Total expense | 0.0031 | 0.0016 |
| Borrowing dependency | 0.0030 | 0.0006 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0020 | 0.0014 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0018 | 0.0009 |
| pca__5 | 0.0017 | 0.0020 |

## svm — permutation

| feature | importance | std |
|---|---|---|
| pca__1 | 0.0820 | 0.0043 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0291 | 0.0027 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0277 | 0.0018 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0150 | 0.0038 |
| pca__5 | 0.0097 | 0.0046 |
| ROA(B) before interest and depreciation after tax | 0.0096 | 0.0037 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0095 | 0.0030 |
| Total income/Total expense | 0.0095 | 0.0053 |
| Allocation rate per person | 0.0079 | 0.0045 |
| pca__4 | 0.0079 | 0.0019 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0070 | 0.0023 |
| pca__7 | 0.0065 | 0.0021 |
| Retained Earnings to Total Assets | 0.0049 | 0.0026 |
| Borrowing dependency | 0.0028 | 0.0020 |
| Net Income to Total Assets | 0.0027 | 0.0034 |

## knn — permutation

| feature | importance | std |
|---|---|---|
| pca__1 | 0.0770 | 0.0111 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0349 | 0.0083 |
| Retained Earnings to Total Assets | 0.0224 | 0.0067 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0211 | 0.0045 |
| pca__6 | 0.0163 | 0.0022 |
| pca__2 | 0.0129 | 0.0095 |
| pca__3 | 0.0121 | 0.0124 |
| Non-industry income and expenditure/revenue | 0.0100 | 0.0080 |
| Net Income to Total Assets | 0.0051 | 0.0063 |
| pca__4 | 0.0041 | 0.0110 |
| ROA(B) before interest and depreciation after tax | 0.0023 | 0.0058 |
| Debt ratio % | 0.0008 | 0.0117 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | -0.0007 | 0.0069 |
| pca__7 | -0.0023 | 0.0048 |
| Total income/Total expense | -0.0024 | 0.0008 |

## naive_bayes — permutation

| feature | importance | std |
|---|---|---|
| Total income/Total expense | 0.0185 | 0.0078 |
| Debt ratio % | 0.0052 | 0.0024 |
| log__Net Value Growth Rate | 0.0044 | 0.0010 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0037 | 0.0013 |
| ROA(B) before interest and depreciation after tax | 0.0037 | 0.0012 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0036 | 0.0012 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0034 | 0.0015 |
| Net Income to Total Assets | 0.0034 | 0.0014 |
| pca__1 | 0.0031 | 0.0013 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0020 | 0.0013 |
| Borrowing dependency | 0.0018 | 0.0009 |
| Total debt/Total net worth | 0.0011 | 0.0010 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0010 | 0.0006 |
| pca__4 | 0.0008 | 0.0012 |
| Allocation rate per person | 0.0006 | 0.0013 |

## mlp — permutation

| feature | importance | std |
|---|---|---|
| pca__1 | 0.1551 | 0.0214 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0453 | 0.0054 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0352 | 0.0047 |
| Net Income to Total Assets | 0.0309 | 0.0062 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0224 | 0.0035 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0154 | 0.0113 |
| pca__3 | 0.0122 | 0.0047 |
| pca__5 | 0.0113 | 0.0054 |
| Total income/Total expense | 0.0071 | 0.0037 |
| Non-industry income and expenditure/revenue | 0.0063 | 0.0048 |
| Total debt/Total net worth | 0.0052 | 0.0041 |
| Retained Earnings to Total Assets | 0.0047 | 0.0027 |
| Borrowing dependency | 0.0037 | 0.0007 |
| Debt ratio % | 0.0032 | 0.0015 |
| pca__7 | 0.0025 | 0.0026 |
