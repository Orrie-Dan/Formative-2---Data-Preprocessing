# Multimodal Authentication

Customer authentication and product-recommendation pipeline combining **social profile**, **transaction**, **face image**, and **voice audio** signals.

## Project structure

```
multimodal-authentication/
├── data/
│   ├── customer_social_profiles.csv
│   ├── customer_transactions.csv
│   ├── merged_dataset.csv           # platform-level join (notebook 01)
│   ├── modeling_dataset.csv         # one row per customer (notebook 01)
│   ├── image_features.csv
│   └── audio_features.csv
├── images/
├── audio/
├── models/
│   ├── product_model.pkl
│   └── product_label_encoder.pkl
├── reports/
│   ├── evaluation_report.md
│   └── confusion_matrix.png
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   └── 02_product_recommendation.ipynb
├── scripts/
│   └── _build_notebooks.py          # regenerates notebook templates if needed
├── archive/                         # obsolete scripts (not part of the pipeline)
├── app.py
├── process_images.py
├── requirements.txt
└── README.md
```

## Setup

```bash
cd multimodal-authentication
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## How to reproduce results

Run notebooks **in order** from the `notebooks/` directory (relative paths assume that CWD):

1. **Preprocessing** — `notebooks/01_data_preprocessing.ipynb`
2. **Train & evaluate** — `notebooks/02_product_recommendation.ipynb`
3. **(Optional) Image features** — `python process_images.py`
4. **Demo app** — `streamlit run app.py`

Expected artifacts after steps 1–2:

| Artifact | Produced by |
|----------|-------------|
| `data/merged_dataset.csv` | Notebook 01 |
| `data/modeling_dataset.csv` | Notebook 01 |
| `models/product_model.pkl` | Notebook 02 |
| `models/product_label_encoder.pkl` | Notebook 02 |
| `reports/evaluation_report.md` | Notebook 02 |
| `reports/confusion_matrix.png` | Notebook 02 |

## Preprocessing workflow (Notebook 01)

1. **Load** social + transaction CSVs with `encoding="utf-8"`, `sep=","`; fail early if required columns are missing.
2. **Validate** before cleaning: dimensions, missing values, duplicate rows, duplicate `customer_id`s.
3. **Clean**: normalize IDs (strip + lowercase), standardize categoricals, coerce numerics/timestamps, drop bad rows, winsorize `amount` at 1st/99th percentile.
4. **Feature engineering (customer-level TX)**: RFM (`total_spent`, `average_spent`, `purchase_count`, `days_since_last_purchase`), `fraud_rate`, `product_category` = mode of TX `category`.
5. **Merge**: attach TX features + social aggregates (`average_engagement`, `platform_count`, `main_platform`) to social rows.
6. **Export**:
   - `merged_dataset.csv` — **platform-level** (intentional; multiple rows per customer when they have multiple social platforms). Used by the Streamlit auth demo.
   - `modeling_dataset.csv` — **one row per customer** (mean engagement / interest, mode sentiment, same TX aggregates). Used for recommendation training.

## Feature engineering pipeline

| Feature group | Source | Used for |
|---------------|--------|----------|
| RFM + `fraud_rate` | Transactions | Auth heuristic (`app.py`); stored in both CSVs |
| `product_category` | Mode of TX `category` | Recommendation **target** |
| Social scores / platform aggregates | Profiles | Recommendation **features** + auth heuristic |

Recency uses a fixed reference date (`2025-04-01`) so runs are reproducible.

## Training process (Notebook 02)

1. Load `data/modeling_dataset.csv` (falls back to deduped `merged_dataset.csv` if missing).
2. Drop leakage columns that share a TX source with the target: `total_spent`, `average_spent`, `purchase_count`, `days_since_last_purchase`, `fraud_rate`, `last_purchase`.
3. Build a `ColumnTransformer` + `RandomForestClassifier` pipeline (`class_weight="balanced"`, `n_estimators=200`).
4. Stratified holdout split (80/20) with an assertion that customer IDs do not overlap train/test.
5. Fit on train; evaluate on holdout + 2-fold stratified CV.

## Evaluation process

Holdout metrics written to `reports/evaluation_report.md`:

- Accuracy, Precision (macro), Recall (macro), F1 (macro), F1 (weighted)
- Multiclass **ROC-AUC** (One-vs-Rest, macro) from `predict_proba`
- **Log loss** from predicted probabilities
- Classification report + confusion matrix (`reports/confusion_matrix.png`)
- Stratified K-Fold CV (`n_splits=2`; rarest class has only 2 customers)
- Feature importances (top 25)

If ROC-AUC or log loss cannot be computed (e.g. a class absent from the holdout fold), the report documents the reason instead of failing.

## Run the demo app

```bash
streamlit run app.py
```

The app uses `fraud_rate` and social scores from `merged_dataset.csv` for a heuristic auth decision. Customer filters are case-insensitive so raw TX/profile IDs still match lowercased merged IDs.

## Data overview

| File | Description |
|------|-------------|
| `customer_social_profiles.csv` | Social identity and engagement attributes |
| `customer_transactions.csv` | Transaction history with fraud flags |
| `merged_dataset.csv` | Platform-level joined view (auth / inspection) |
| `modeling_dataset.csv` | Customer-level frame for recommendation training |
| `image_features.csv` | Extracted face / emotion image features |
| `audio_features.csv` | Extracted MFCC / pitch audio features |

## Modeling notes

- **Merge IDs:** always strip + lowercase before joins.
- **Target:** `product_category` = mode of transaction `category`.
- **Leakage control:** RFM / `fraud_rate` stay in the CSVs for authentication, but are **excluded from the recommendation model** because they share a source with the target.
- **Leakage control:** training uses **one row per customer** (`modeling_dataset.csv`); stratified holdout + `StratifiedKFold(n_splits=2)`.
- **Imbalance:** `class_weight="balanced"` on `RandomForestClassifier`.
- **Platform vs customer grain:** `merged_dataset.csv` is intentionally platform-level; do not train on it without aggregating/deduping.

## Limitations

- Recommendation accuracy is modest (small N, many classes, social-only features by design).
- Audio `customer_id` values (`C001`…) are not yet linked to tabular IDs (`a178`); the app falls back to demo `person1` audio.
- Image features in this repo are demo-scoped (`person1`).
- CV uses `n_splits=2` because the rarest product class has only two customers.
