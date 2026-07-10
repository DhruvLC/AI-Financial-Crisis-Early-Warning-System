"""Pipeline stages for the AI Financial Crisis Early Warning System.

Each module maps to a stage in the end-to-end diagram:

    01_data_collection.py     -> stage 2   (Data Collection)
    02_data_prep.py           -> stage 3   (Data Preparation)
    03_features.py            -> stage 6   (Feature Engineering)
    04_models.py              -> stages 7  (Traditional ML) + Model Comparison
    05_explain.py             -> stage 13  (Explainable AI)
    06_risk_score.py          -> Output    (0-100 Financial Risk Score)

Deep-learning / transformer / self-supervised stages (8-12) are left as
stubs to fill in during phase 2.
"""
