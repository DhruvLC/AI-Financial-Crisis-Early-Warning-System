# SHAP Report

Generated: 2026-07-16T08:46:23.837282+00:00
Dataset: feature store v001 (target `Bankrupt?`)


## logistic_regression (500 samples)

| Feature | mean |SHAP| |
|---|---|
| pca__1 | 0.0730 |
| Debt ratio % | 0.0556 |
| ROA(B) before interest and depreciation after tax | 0.0401 |
| Total income/Total expense | 0.0387 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0370 |
| pca__2 | 0.0354 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0346 |
| Borrowing dependency | 0.0204 |
| Net Income to Total Assets | 0.0164 |
| pca__7 | 0.0146 |
| pca__4 | 0.0119 |
| Allocation rate per person | 0.0118 |
| Total debt/Total net worth | 0.0112 |
| pca__3 | 0.0080 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0067 |

## decision_tree (500 samples)

| Feature | mean |SHAP| |
|---|---|
| pca__1 | 0.2170 |
| ROA(B) before interest and depreciation after tax | 0.0575 |
| Non-industry income and expenditure/revenue | 0.0567 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0533 |
| Retained Earnings to Total Assets | 0.0526 |
| Debt ratio % | 0.0467 |
| Allocation rate per person | 0.0360 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0277 |
| Total income/Total expense | 0.0191 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0143 |
| pca__7 | 0.0131 |
| pca__4 | 0.0099 |
| log__Net Value Growth Rate | 0.0093 |
| pca__6 | 0.0075 |
| Borrowing dependency | 0.0059 |

## random_forest (500 samples)

| Feature | mean |SHAP| |
|---|---|
| pca__1 | 0.0527 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0441 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0354 |
| ROA(B) before interest and depreciation after tax | 0.0322 |
| Retained Earnings to Total Assets | 0.0295 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0289 |
| log__Net Value Growth Rate | 0.0269 |
| Total debt/Total net worth | 0.0253 |
| Total income/Total expense | 0.0241 |
| Borrowing dependency | 0.0239 |
| Non-industry income and expenditure/revenue | 0.0235 |
| Debt ratio % | 0.0195 |
| Net Income to Total Assets | 0.0187 |
| Allocation rate per person | 0.0161 |
| pca__2 | 0.0153 |

## extra_trees (500 samples)

| Feature | mean |SHAP| |
|---|---|
| pca__1 | 0.0439 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0408 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0381 |
| Borrowing dependency | 0.0361 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0358 |
| Total debt/Total net worth | 0.0287 |
| Debt ratio % | 0.0275 |
| ROA(B) before interest and depreciation after tax | 0.0257 |
| log__Net Value Growth Rate | 0.0228 |
| Net Income to Total Assets | 0.0226 |
| Total income/Total expense | 0.0224 |
| Retained Earnings to Total Assets | 0.0219 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0180 |
| Non-industry income and expenditure/revenue | 0.0156 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0132 |

## xgboost (500 samples)

| Feature | mean |SHAP| |
|---|---|
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 1.1705 |
| pca__1 | 1.1526 |
| ROA(B) before interest and depreciation after tax | 1.0299 |
| Allocation rate per person | 0.7583 |
| Total debt/Total net worth | 0.6855 |
| log__Net Value Growth Rate | 0.6421 |
| Non-industry income and expenditure/revenue | 0.5966 |
| pca__7 | 0.5122 |
| Retained Earnings to Total Assets | 0.5115 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.4817 |
| Total income/Total expense | 0.4596 |
| pca__5 | 0.4183 |
| pca__4 | 0.3806 |
| pca__6 | 0.3783 |
| Borrowing dependency | 0.3300 |

## svm (500 samples)

| Feature | mean |SHAP| |
|---|---|
| pca__1 | 0.0118 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0053 |
| pca__2 | 0.0034 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0034 |
| Total income/Total expense | 0.0031 |
| Debt ratio % | 0.0024 |
| Allocation rate per person | 0.0024 |
| pca__3 | 0.0023 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0023 |
| Total debt/Total net worth | 0.0020 |
| ROA(B) before interest and depreciation after tax | 0.0018 |
| pca__5 | 0.0015 |
| Retained Earnings to Total Assets | 0.0014 |
| Borrowing dependency | 0.0014 |
| log__Net Value Growth Rate | 0.0014 |

## knn (500 samples)

| Feature | mean |SHAP| |
|---|---|
| pca__1 | 0.0200 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0044 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0039 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0034 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0033 |
| pca__3 | 0.0026 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0022 |
| Allocation rate per person | 0.0021 |
| pca__2 | 0.0020 |
| ROA(B) before interest and depreciation after tax | 0.0020 |
| pca__4 | 0.0019 |
| Non-industry income and expenditure/revenue | 0.0015 |
| Retained Earnings to Total Assets | 0.0014 |
| Borrowing dependency | 0.0014 |
| Total debt/Total net worth | 0.0014 |

## naive_bayes (500 samples)

| Feature | mean |SHAP| |
|---|---|
| Total debt/Total net worth | 0.0149 |
| Allocation rate per person | 0.0143 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0139 |
| pca__5 | 0.0122 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0104 |
| pca__4 | 0.0102 |
| Retained Earnings to Total Assets | 0.0100 |
| Total income/Total expense | 0.0098 |
| Debt ratio % | 0.0095 |
| Net Income to Total Assets | 0.0092 |
| pca__1 | 0.0091 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0090 |
| ROA(B) before interest and depreciation after tax | 0.0077 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0073 |
| Borrowing dependency | 0.0072 |

## mlp (500 samples)

| Feature | mean |SHAP| |
|---|---|
| pca__1 | 0.0177 |
| pca__2 | 0.0176 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.0097 |
| Total debt/Total net worth | 0.0090 |
| diff__Net Income to Stockholder's Equity__minus__Borrowing dependency | 0.0083 |
| Net Income to Total Assets | 0.0062 |
| pca__5 | 0.0060 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0050 |
| pca__3 | 0.0043 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0037 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.0033 |
| Non-industry income and expenditure/revenue | 0.0033 |
| pca__7 | 0.0032 |
| log__Net Value Growth Rate | 0.0026 |
| Retained Earnings to Total Assets | 0.0025 |
