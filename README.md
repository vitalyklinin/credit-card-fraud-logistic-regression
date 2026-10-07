# Credit Card Fraud Detection — Logistic Regression (Streamlit)

Streamlit dashboard (Hebrew, RTL) that trains a logistic regression model to detect credit card fraud,
shows its performance and lets you score a new transaction.

## Data

`credit_card_fraud_10k.csv` — 10,000 transactions, 151 frauds (1.51%). **Not included in the repo** —
place it in the project root before running.

Columns: `amount`, `transaction_hour`, `merchant_category`, `foreign_transaction`, `location_mismatch`,
`device_trust_score`, `velocity_last_24h`, `cardholder_age`, target `is_fraud`.

## Pipeline

1. Stratified train/test split (80/20) **before** any cleaning — the test set stays untouched.
2. Missing values: none in the file; `SimpleImputer` (median / most frequent) kept in the pipeline for future data.
3. Outliers: IQR (k=3) removed from the **training set only**, legit transactions only — fraud rows are kept
   (plain IQR would have removed ~10% of the frauds).
4. `StandardScaler` for numeric columns, One-Hot for `merchant_category`, `transaction_id` dropped.
5. Class imbalance: fraud class weight 10 + decision threshold chosen by 5-fold CV on the training set
   to maximize F-beta (default β=3 → priority on catching fraud).

## Results (test set, 30 frauds, β=3)

| ROC AUC | Recall | Precision | F1 |
|---|---|---|---|
| 0.994 | 0.933 (28/30) | 0.315 | 0.471 |

| β | Frauds caught | Precision | F1 |
|---|---|---|---|
| 1 | 24/30 | 0.533 | 0.640 |
| 2 | 27/30 | 0.435 | 0.587 |
| 3 | 28/30 | 0.315 | 0.471 |

## SMOTE — tested, not adopted

`experiments/smote_experiment.py`. SMOTE (inside CV, training folds only) looked better on the default split,
but over 5 other splits (150 frauds total) it caught fewer frauds:

| Method | Frauds caught | False alarms |
|---|---|---|
| **Class weight 10 (used)** | **130/150** | 177 |
| SMOTE 0.1 | 119/150 | 146 |
| SMOTE 0.3 | 126/150 | 163 |

With logistic regression, SMOTE mostly shifts the precision/recall trade-off like class weighting does;
PR-AUC was essentially unchanged.

## Run

```bash
pip install -r requirements.txt
streamlit run fraud_app.py
```
