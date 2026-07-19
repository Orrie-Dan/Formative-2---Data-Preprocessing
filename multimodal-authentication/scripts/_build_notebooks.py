"""One-shot builder for remediated assignment notebooks."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def _src(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    if not lines:
        return []
    return [line + "\n" for line in lines[:-1]] + [lines[-1] + "\n"]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(text),
    }


def write_nb(path: Path, cells: list[dict]) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13.0"},
        },
        "cells": cells,
    }
    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {path} ({len(cells)} cells)")


def build_01() -> None:
    cells = [
        md(
            """# 01 — Data Preprocessing & Merge

Builds a reproducible tabular merge of social profiles and transactions.

**Decisions**
- Normalize `customer_id` (strip + lowercase) **before** any merge.
- Social categoricals: strip + lowercase; missing text → `unknown`.
- Transactions: coerce amount/timestamp; drop non-positive amounts and bad timestamps.
- Soft outlier winsorize on `amount` at 1st/99th percentile.
- RFM features: `total_spent`, `average_spent`, `purchase_count`, `days_since_last_purchase`, plus `fraud_rate`.
- `product_category` = mode of TX `category` (kept in export; notebook 02 must not train with same-source spend features).
- Recency uses a fixed reference date (`2025-04-01`) for reproducibility.
- `merged_dataset.csv` is **intentionally platform-level** (one row per social-platform row).
- `modeling_dataset.csv` is the **customer-level** training frame (one row per customer, no ID duplication)."""
        ),
        code(
            """from pathlib import Path

import pandas as pd

DATA_DIR = Path("../data")
SOCIAL_PATH = DATA_DIR / "customer_social_profiles.csv"
TX_PATH = DATA_DIR / "customer_transactions.csv"
OUT_PATH = DATA_DIR / "merged_dataset.csv"
MODELING_PATH = DATA_DIR / "modeling_dataset.csv"

REQUIRED_SOCIAL = [
    "customer_id",
    "social_media_platform",
    "engagement_score",
    "purchase_interest_score",
    "review_sentiment",
]
REQUIRED_TX = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "amount",
    "category",
    "is_fraud",
]"""
        ),
        md("## 1. Load with validation"),
        code(
            """def load_csv(path: Path, required_columns: list[str], name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found at {path.resolve()}")
    try:
        df = pd.read_csv(path, encoding="utf-8", sep=",")
    except Exception as exc:
        raise RuntimeError(f"Failed to read {name} from {path}") from exc

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"{name} is empty")

    print(f"[OK] Loaded {name}")
    print(f"  path: {path.resolve()}")
    print(f"  shape: {df.shape}")
    print(f"  dtypes:\\n{df.dtypes.to_string()}")
    return df


def report_data_quality(df: pd.DataFrame, name: str, id_col: str = "customer_id") -> None:
    \"\"\"Pre-preprocessing validation summary (does not mutate).\"\"\"
    n_dup_rows = int(df.duplicated().sum())
    n_dup_ids = int(df[id_col].duplicated().sum()) if id_col in df.columns else -1
    print(f"\\n=== Data quality: {name} ===")
    print(f"  dimensions: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  missing values:\\n{df.isnull().sum().to_string()}")
    print(f"  duplicate rows: {n_dup_rows}")
    if id_col in df.columns:
        print(f"  duplicate {id_col} values: {n_dup_ids}")
        print(f"  unique {id_col}: {df[id_col].nunique()}")


social_df = load_csv(SOCIAL_PATH, REQUIRED_SOCIAL, "Social profiles")
transactions_df = load_csv(TX_PATH, REQUIRED_TX, "Transactions")

report_data_quality(social_df, "Social profiles")
report_data_quality(transactions_df, "Transactions")
print("\\nSocial head:")
print(social_df.head())
print("\\nTransactions head:")
print(transactions_df.head())"""
        ),
        md("## 2. Clean & standardize"),
        code(
            """# --- IDs first (must happen before any merge) ---
social_df["customer_id"] = social_df["customer_id"].astype(str).str.strip().str.lower()
transactions_df["customer_id"] = (
    transactions_df["customer_id"].astype(str).str.strip().str.lower()
)

# --- Social categoricals ---
for col in ["social_media_platform", "review_sentiment"]:
    social_df[col] = social_df[col].astype(str).str.strip().str.lower()

social_df["engagement_score"] = pd.to_numeric(
    social_df["engagement_score"], errors="coerce"
)
social_df["purchase_interest_score"] = pd.to_numeric(
    social_df["purchase_interest_score"], errors="coerce"
)

n_social_before = len(social_df)
social_df = social_df.drop_duplicates()
print(f"Social duplicates removed: {n_social_before - len(social_df)}")

social_df["social_media_platform"] = social_df["social_media_platform"].fillna("unknown")
social_df["review_sentiment"] = social_df["review_sentiment"].fillna("unknown")
social_df["engagement_score"] = social_df["engagement_score"].fillna(
    social_df["engagement_score"].median()
)
social_df["purchase_interest_score"] = social_df["purchase_interest_score"].fillna(
    social_df["purchase_interest_score"].median()
)

assert social_df["customer_id"].notna().all(), "Null customer_id in social_df"
assert social_df.isnull().sum().sum() == 0, social_df.isnull().sum()

# --- Transactions ---
n_tx_before = len(transactions_df)
transactions_df = transactions_df.drop_duplicates()
print(f"TX duplicates removed: {n_tx_before - len(transactions_df)}")

transactions_df["amount"] = pd.to_numeric(transactions_df["amount"], errors="coerce")
transactions_df["timestamp"] = pd.to_datetime(
    transactions_df["timestamp"], errors="coerce"
)
transactions_df["category"] = (
    transactions_df["category"].astype(str).str.strip().str.lower()
)

bad_amount = transactions_df["amount"].isna() | (transactions_df["amount"] <= 0)
bad_ts = transactions_df["timestamp"].isna()
print(f"Dropping TX with bad amount: {int(bad_amount.sum())}")
print(f"Dropping TX with bad timestamp: {int(bad_ts.sum())}")
transactions_df = transactions_df.loc[~bad_amount & ~bad_ts].copy()

lo, hi = transactions_df["amount"].quantile([0.01, 0.99])
n_out = int(((transactions_df["amount"] < lo) | (transactions_df["amount"] > hi)).sum())
transactions_df["amount"] = transactions_df["amount"].clip(lo, hi)
print(f"Amounts winsorized (1-99%): {n_out} values clipped to [{lo:.2f}, {hi:.2f}]")

assert transactions_df["customer_id"].notna().all()
assert (transactions_df["amount"] > 0).all()
assert transactions_df["timestamp"].notna().all()
print("Clean social:", social_df.shape, "| Clean TX:", transactions_df.shape)"""
        ),
        md("## 3. Feature engineering (RFM + fraud + preferred category)"),
        code(
            """transaction_features = (
    transactions_df.groupby("customer_id", as_index=False)
    .agg(
        total_spent=("amount", "sum"),
        average_spent=("amount", "mean"),
        purchase_count=("amount", "count"),
        last_purchase=("timestamp", "max"),
        fraud_rate=("is_fraud", "mean"),
    )
)

# Fixed reference date so recency is reproducible across runs
ref_day = pd.Timestamp("2025-04-01")
transaction_features["days_since_last_purchase"] = (
    ref_day - transaction_features["last_purchase"]
).dt.days

product_category = (
    transactions_df.groupby("customer_id")["category"]
    .agg(lambda s: s.value_counts().index[0])
    .reset_index(name="product_category")
)
transaction_features = transaction_features.merge(
    product_category, on="customer_id", how="left"
)

required_tx_feats = [
    "total_spent",
    "average_spent",
    "purchase_count",
    "days_since_last_purchase",
    "fraud_rate",
    "product_category",
]
missing_feats = [c for c in required_tx_feats if c not in transaction_features.columns]
assert not missing_feats, missing_feats
assert transaction_features[required_tx_feats].isnull().sum().sum() == 0, (
    transaction_features[required_tx_feats].isnull().sum()
)

print("Transaction feature rows:", len(transaction_features))
print("Target class distribution (customer-level):")
print(transaction_features["product_category"].value_counts())
print(transaction_features.head())"""
        ),
        md(
            """## 4. Social aggregates + merge

`merged_df` stays **platform-level** (one social-profile row per platform × customer) for authentication / inspection.
A separate **customer-level** `modeling_df` is built next so recommendation training never duplicates IDs."""
        ),
        code(
            """engagement = (
    social_df.groupby("customer_id", as_index=False)
    .agg(average_engagement=("engagement_score", "mean"))
)
platform_count = (
    social_df.groupby("customer_id")["social_media_platform"]
    .nunique()
    .reset_index(name="platform_count")
)
main_platform = (
    social_df.groupby("customer_id")["social_media_platform"]
    .agg(lambda s: s.value_counts().index[0])
    .reset_index(name="main_platform")
)

n_before = len(social_df)
merged_df = social_df.merge(transaction_features, on="customer_id", how="left")
merged_df = merged_df.merge(engagement, on="customer_id", how="left")
merged_df = merged_df.merge(platform_count, on="customer_id", how="left")
merged_df = merged_df.merge(main_platform, on="customer_id", how="left")
merged_df = merged_df.drop_duplicates()

assert len(merged_df) == n_before, "Merge changed social row count unexpectedly"

tx_cols = [
    "total_spent",
    "average_spent",
    "purchase_count",
    "days_since_last_purchase",
    "fraud_rate",
]
match_rate = merged_df["total_spent"].notna().mean()
print(f"TX match rate (non-null total_spent): {match_rate:.1%}")
assert match_rate >= 0.5, (
    f"Suspiciously low TX match rate ({match_rate:.1%}) — check customer_id casing"
)

# Customers with no TX: neutral defaults
merged_df["total_spent"] = merged_df["total_spent"].fillna(0.0)
merged_df["average_spent"] = merged_df["average_spent"].fillna(0.0)
merged_df["purchase_count"] = merged_df["purchase_count"].fillna(0).astype(int)
merged_df["days_since_last_purchase"] = merged_df["days_since_last_purchase"].fillna(-1)
merged_df["fraud_rate"] = merged_df["fraud_rate"].fillna(0.0)
merged_df["product_category"] = merged_df["product_category"].fillna("unknown")

assert merged_df[tx_cols].isnull().sum().sum() == 0, merged_df[tx_cols].isnull().sum()
assert (merged_df["customer_id"] == merged_df["customer_id"].str.lower()).all()

print("Platform-level merged shape:", merged_df.shape)
print("Unique customers in merged:", merged_df["customer_id"].nunique())
print("Nulls:\\n", merged_df.isnull().sum())
print(merged_df.head())"""
        ),
        md(
            """## 5. Customer-level modeling dataset

Aggregates social signals to **one row per `customer_id`** for leakage-safe train/test splits.
RFM / `fraud_rate` / `product_category` are already customer-level from Step 3."""
        ),
        code(
            """modeling_df = (
    social_df.groupby("customer_id", as_index=False)
    .agg(
        engagement_score=("engagement_score", "mean"),
        purchase_interest_score=("purchase_interest_score", "mean"),
        review_sentiment=("review_sentiment", lambda s: s.value_counts().index[0]),
    )
)
modeling_df = modeling_df.merge(engagement, on="customer_id", how="left")
modeling_df = modeling_df.merge(platform_count, on="customer_id", how="left")
modeling_df = modeling_df.merge(main_platform, on="customer_id", how="left")
modeling_df = modeling_df.merge(transaction_features, on="customer_id", how="left")

modeling_df["total_spent"] = modeling_df["total_spent"].fillna(0.0)
modeling_df["average_spent"] = modeling_df["average_spent"].fillna(0.0)
modeling_df["purchase_count"] = modeling_df["purchase_count"].fillna(0).astype(int)
modeling_df["days_since_last_purchase"] = modeling_df["days_since_last_purchase"].fillna(-1)
modeling_df["fraud_rate"] = modeling_df["fraud_rate"].fillna(0.0)
modeling_df["product_category"] = modeling_df["product_category"].fillna("unknown")

assert modeling_df["customer_id"].is_unique, "modeling_df must be one row per customer"
assert modeling_df["product_category"].notna().all()
print("Customer-level modeling shape:", modeling_df.shape)
print("Class distribution:")
print(modeling_df["product_category"].value_counts())
print(modeling_df.head())"""
        ),
        md("## 6. Export"),
        code(
            """DATA_DIR.mkdir(parents=True, exist_ok=True)
merged_df.to_csv(OUT_PATH, index=False, encoding="utf-8")
modeling_df.to_csv(MODELING_PATH, index=False, encoding="utf-8")

reloaded = pd.read_csv(OUT_PATH, encoding="utf-8", sep=",")
assert list(reloaded.columns) == list(merged_df.columns)
assert reloaded["total_spent"].notna().all()
assert (reloaded["total_spent"] >= 0).all()

reloaded_model = pd.read_csv(MODELING_PATH, encoding="utf-8", sep=",")
assert reloaded_model["customer_id"].is_unique
assert len(reloaded_model) == merged_df["customer_id"].nunique()

print(f"Saved platform-level -> {OUT_PATH.resolve()} ({len(reloaded)} rows)")
print(f"Saved modeling      -> {MODELING_PATH.resolve()} ({len(reloaded_model)} rows)")"""
        ),
    ]
    write_nb(NB_DIR / "01_data_preprocessing.ipynb", cells)


def build_02() -> None:
    cells = [
        md(
            """# 02 — Product Category Recommendation

Trains a scikit-learn Pipeline to predict preferred `product_category` from **social signals only**.

**Anti-leakage**
- Target = mode of transaction `category` (created in notebook 01).
- Transaction RFM / fraud aggregates share that same TX source → **excluded from `X`**.
- Training uses `data/modeling_dataset.csv` (**one row per customer**) so no customer can appear in both train and test.

**Imbalance:** `class_weight="balanced"` on `RandomForestClassifier` (inverse class frequency).

**Evaluation:** holdout metrics (incl. ROC-AUC OvR + log loss) + confusion matrix + Stratified K-Fold CV (`n_splits=2` because smallest class has 2 customers) + Markdown report under `reports/`."""
        ),
        code(
            """from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

DATA_PATH = Path("../data/modeling_dataset.csv")
FALLBACK_PATH = Path("../data/merged_dataset.csv")
MODEL_PATH = Path("../models/product_model.pkl")
ENCODER_PATH = Path("../models/product_label_encoder.pkl")
REPORT_DIR = Path("../reports")
REPORT_PATH = REPORT_DIR / "evaluation_report.md"
CM_PATH = REPORT_DIR / "confusion_matrix.png"
TARGET = "product_category"
RANDOM_STATE = 42

# Same-source as TARGET — keep in CSV for auth, exclude from classifier inputs
TRANSACTION_LEAK_COLS = [
    "total_spent",
    "average_spent",
    "purchase_count",
    "days_since_last_purchase",
    "fraud_rate",
    "last_purchase",
]"""
        ),
        md("## 1. Load & validate"),
        code(
            """def load_modeling_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8", sep=",")
    print(f"[OK] Loaded {path.name}  shape={df.shape}")
    print(f"  dtypes:\\n{df.dtypes.to_string()}")
    return df


if DATA_PATH.exists():
    df = load_modeling_frame(DATA_PATH)
    source_name = DATA_PATH.name
elif FALLBACK_PATH.exists():
    print(
        f"WARNING: {DATA_PATH} missing — falling back to {FALLBACK_PATH.name} "
        "and deduplicating to one row per customer."
    )
    df = load_modeling_frame(FALLBACK_PATH)
    source_name = FALLBACK_PATH.name
else:
    raise FileNotFoundError(
        f"Missing {DATA_PATH.resolve()}. Run notebooks/01_data_preprocessing.ipynb first."
    )

required = ["customer_id", TARGET, "engagement_score", "purchase_interest_score"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"{source_name} missing columns: {missing}")

df["customer_id"] = df["customer_id"].astype(str).str.strip().str.lower()
df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()

n_rows = len(df)
if df["customer_id"].is_unique:
    model_df = df.reset_index(drop=True)
else:
    model_df = (
        df.sort_values(["customer_id", "engagement_score"], ascending=[True, False])
        .drop_duplicates(subset=["customer_id"], keep="first")
        .reset_index(drop=True)
    )
    print(f"Deduped platform-level rows {n_rows} -> {len(model_df)} customers")

assert model_df["customer_id"].is_unique
assert model_df[TARGET].notna().all()
print(f"Modeling rows: {len(model_df)} unique customers (source={source_name})")
print("Class distribution:")
print(model_df[TARGET].value_counts())
print("\\nMissing values:")
print(model_df.isnull().sum())"""
        ),
        md("## 2. Features / target (no TX leakage)"),
        code(
            """drop_cols = [TARGET, "customer_id"] + [
    c for c in TRANSACTION_LEAK_COLS if c in model_df.columns
]
X = model_df.drop(columns=[c for c in drop_cols if c in model_df.columns])
y_raw = model_df[TARGET]

assert not any(c in X.columns for c in TRANSACTION_LEAK_COLS), X.columns.tolist()
assert len(X) == model_df["customer_id"].nunique()

label = LabelEncoder()
y = label.fit_transform(y_raw)

categorical_columns = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
numeric_columns = [c for c in X.columns if c not in categorical_columns]
print("Categorical:", categorical_columns)
print("Numeric:", numeric_columns)
print("Classes:", list(label.classes_))"""
        ),
        md("## 3. Pipeline, split, train"),
        code(
            """numeric_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)
categorical_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_columns),
        ("cat", categorical_pipe, categorical_columns),
    ]
)

# class_weight='balanced' up-weights rare product categories (Travel/Wholesale, etc.)
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
        ),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Customer-level frame ⇒ IDs cannot overlap across splits
train_ids = set(model_df.loc[X_train.index, "customer_id"])
test_ids = set(model_df.loc[X_test.index, "customer_id"])
assert train_ids.isdisjoint(test_ids), "Customer leakage detected in holdout split"

model.fit(X_train, y_train)
print("Train size:", len(X_train), "Test size:", len(X_test))"""
        ),
        md("## 4. Holdout evaluation"),
        code(
            """y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
labels_idx = list(range(len(label.classes_)))

accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(
    y_test, y_pred, average="macro", labels=labels_idx, zero_division=0
)
recall_macro = recall_score(
    y_test, y_pred, average="macro", labels=labels_idx, zero_division=0
)
f1_macro = f1_score(
    y_test, y_pred, average="macro", labels=labels_idx, zero_division=0
)
f1_weighted = f1_score(
    y_test, y_pred, average="weighted", labels=labels_idx, zero_division=0
)

# Multiclass ROC-AUC (One-vs-Rest) + log loss from predicted probabilities
roc_auc_macro = None
roc_auc_note = ""
present_test = np.unique(y_test)
missing_in_test = sorted(set(labels_idx) - set(present_test.tolist()))
try:
    if len(present_test) < 2:
        raise ValueError("Holdout set has fewer than 2 classes.")
    # Subset + renormalize so column count matches classes present in y_test
    # (full 12-col proba fails when a rare class is absent from the holdout).
    y_proba_present = y_proba[:, present_test]
    row_sums = y_proba_present.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Predicted probabilities for present classes sum to zero.")
    y_proba_present = y_proba_present / row_sums
    roc_auc_macro = float(
        roc_auc_score(
            y_test,
            y_proba_present,
            multi_class="ovr",
            average="macro",
            labels=present_test,
        )
    )
    if np.isnan(roc_auc_macro) or np.isinf(roc_auc_macro):
        raise ValueError(
            "ROC-AUC is undefined (NaN/Inf) — typically when a holdout class has "
            "too few positives/negatives for a stable OvR curve."
        )
    if missing_in_test:
        missing_names = [label.classes_[i] for i in missing_in_test]
        roc_auc_note = (
            f"OvR macro AUC over the {len(present_test)} classes present in holdout "
            f"(absent from holdout: {missing_names}). Probabilities renormalized "
            "over present classes only."
        )
except ValueError as exc:
    roc_auc_macro = None
    roc_auc_note = (
        f"ROC-AUC (OvR) not reported: {exc} "
        "With small stratified holdouts, rare classes may be missing or unpaired."
    )

log_loss_value = None
log_loss_note = ""
try:
    log_loss_value = float(log_loss(y_test, y_proba, labels=labels_idx))
    if np.isnan(log_loss_value) or np.isinf(log_loss_value):
        raise ValueError("Log loss is undefined (NaN/Inf).")
except ValueError as exc:
    log_loss_value = None
    log_loss_note = f"Log loss could not be computed: {exc}."

clf_report = classification_report(
    y_test,
    y_pred,
    labels=labels_idx,
    target_names=label.classes_,
    zero_division=0,
)

print(f"Accuracy:           {accuracy:.4f}")
print(f"Precision (macro):  {precision_macro:.4f}")
print(f"Recall (macro):     {recall_macro:.4f}")
print(f"F1 (macro):         {f1_macro:.4f}")
print(f"F1 (weighted):      {f1_weighted:.4f}")
if roc_auc_macro is not None:
    print(f"ROC-AUC (OvR macro): {roc_auc_macro:.4f}")
    if roc_auc_note:
        print(f"  ({roc_auc_note})")
else:
    print(roc_auc_note or "ROC-AUC (OvR macro): N/A")
if log_loss_value is not None:
    print(f"Log loss:           {log_loss_value:.4f}")
    if log_loss_note:
        print(f"  ({log_loss_note})")
else:
    print(log_loss_note or "Log loss: N/A")
print()
print(clf_report)

cm = confusion_matrix(y_test, y_pred, labels=labels_idx)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label.classes_,
    yticklabels=label.classes_,
    ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (holdout)")
plt.tight_layout()
REPORT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(CM_PATH, dpi=120, bbox_inches="tight")
plt.show()
print(f"Saved confusion matrix -> {CM_PATH.resolve()}")"""
        ),
        md(
            """## 5. Stratified K-Fold cross-validation

`n_splits=2` because the rarest class (travel) has only 2 customers — larger `k` cannot stratify that class."""
        ),
        code(
            """cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
cv_results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=["accuracy", "f1_macro"],
    return_train_score=False,
)

acc_scores = cv_results["test_accuracy"]
f1_scores = cv_results["test_f1_macro"]
print("Fold accuracy:", np.round(acc_scores, 4))
print(f"Mean accuracy: {acc_scores.mean():.4f}  std: {acc_scores.std():.4f}")
print("Fold F1-macro:", np.round(f1_scores, 4))
print(f"Mean F1-macro: {f1_scores.mean():.4f}  std: {f1_scores.std():.4f}")"""
        ),
        md("## 6. Feature importance"),
        code(
            """feature_names = model.named_steps["preprocessor"].get_feature_names_out()
importances = (
    pd.Series(
        model.named_steps["classifier"].feature_importances_,
        index=feature_names,
    )
    .sort_values(ascending=False)
)
print(importances.head(20).to_string())"""
        ),
        md("## 7. Export evaluation report + persist model"),
        code(
            """REPORT_DIR.mkdir(parents=True, exist_ok=True)

def _fmt(v):
    return f"{v:.4f}" if v is not None else "N/A"

roc_auc_section = f"| ROC-AUC (OvR macro) | {_fmt(roc_auc_macro)} |\\n"
if roc_auc_note:
    roc_auc_section += f"\\n> {roc_auc_note}\\n"
log_loss_section = f"| Log loss | {_fmt(log_loss_value)} |\\n"
if log_loss_note:
    log_loss_section += f"\\n> {log_loss_note}\\n"

importance_md = "\\n".join(
    f"| `{name}` | {val:.4f} |" for name, val in importances.head(25).items()
)

report = f\"\"\"# Product Recommendation — Evaluation Report

Generated by `notebooks/02_product_recommendation.ipynb`.

## Dataset statistics

| Item | Value |
|------|-------|
| Source file | `{source_name}` |
| Rows (customers) | {len(model_df)} |
| Features used | {X.shape[1]} |
| Classes | {len(label.classes_)} |
| Train / test size | {len(X_train)} / {len(X_test)} |
| Random state | {RANDOM_STATE} |

### Class distribution

```
{model_df[TARGET].value_counts().to_string()}
```

## Holdout metrics

| Metric | Value |
|--------|-------|
| Accuracy | {_fmt(accuracy)} |
| Precision (macro) | {_fmt(precision_macro)} |
| Recall (macro) | {_fmt(recall_macro)} |
| F1 (macro) | {_fmt(f1_macro)} |
| F1 (weighted) | {_fmt(f1_weighted)} |
{roc_auc_section}{log_loss_section}

## Cross-validation (StratifiedKFold, n_splits=2)

| Metric | Fold scores | Mean | Std |
|--------|-------------|------|-----|
| Accuracy | {np.round(acc_scores, 4).tolist()} | {acc_scores.mean():.4f} | {acc_scores.std():.4f} |
| F1 (macro) | {np.round(f1_scores, 4).tolist()} | {f1_scores.mean():.4f} | {f1_scores.std():.4f} |

## Classification report

```
{clf_report}
```

## Confusion matrix

![Confusion matrix](confusion_matrix.png)

```
{pd.DataFrame(cm, index=label.classes_, columns=label.classes_).to_string()}
```

## Feature importance (top 25)

| Feature | Importance |
|---------|------------|
{importance_md}

## Notes

- Features are **social signals only**; RFM / `fraud_rate` are excluded to avoid target leakage.
- Training frame is **one row per customer** (`modeling_dataset.csv`).
- Multiclass ROC-AUC uses One-vs-Rest (`multi_class='ovr'`, macro average).
- Log loss uses predicted class probabilities from `RandomForestClassifier.predict_proba`.
\"\"\"

REPORT_PATH.write_text(report, encoding="utf-8")
print(f"Saved evaluation report -> {REPORT_PATH.resolve()}")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(label, ENCODER_PATH)
print(f"Saved model -> {MODEL_PATH.resolve()}")
print(f"Saved label encoder -> {ENCODER_PATH.resolve()}")"""
        ),
    ]
    write_nb(NB_DIR / "02_product_recommendation.ipynb", cells)


def build_03() -> None:
    cells = [
        md(
            """# 03 — Facial Recognition Model

Trains a face-identity classifier on the 128-d `face_recognition` embeddings produced by
`process_images.py` (`data/image_features.csv`).

**Decisions**
- Features: `Feature1`..`Feature128` embedding columns.
- Target: `Person` — this is an **identification** model ("which known person is this?"),
  not open-set verification.
- **Group-aware split**: each `(Person, Emotion)` pair is one physical photo; its 4 augmented
  variants (original/grayscale/flip/rotate) are near-duplicates. Splitting at the row level
  would leak a transformed copy of a training image into the test set, so every split groups
  all variants of a photo onto the same side via `StratifiedGroupKFold`.
- **Model**: `StandardScaler` + `LogisticRegression` — a large Random Forest overfits fast on
  ~24 rows / 128 dimensions; a simpler linear model in embedding space generalizes better here.
- **Threshold-based unknown handling**: predictions with max class probability below
  `UNKNOWN_THRESHOLD` are rejected as `UNKNOWN` instead of forced onto the closest known
  person. The threshold is persisted so the Task 6 CLI demo can reuse it.
- **Distance-based novelty guard**: softmax confidence alone is not a reliable unknown-input
  detector — a linear classifier can extrapolate *confidently* far outside the training
  distribution (this was verified empirically against the voiceprint model in Task 6; the
  same weakness applies here). So a sample is only accepted if it is *also* within
  `DISTANCE_MARGIN` × the largest observed intra-class distance from its predicted class's
  centroid (in scaled embedding space); otherwise it is rejected as `UNKNOWN` regardless of
  confidence.
- **Small-N caveat**: only a handful of source photos per person — metrics are directional,
  not production-grade."""
        ),
        code(
            """from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = Path("../data/image_features.csv")
MODEL_PATH = Path("../models/face_recognition_model.pkl")
ENCODER_PATH = Path("../models/face_label_encoder.pkl")
CONFIG_PATH = Path("../models/face_recognition_config.json")
REPORT_DIR = Path("../reports")
REPORT_PATH = REPORT_DIR / "face_recognition_report.md"
CM_PATH = REPORT_DIR / "face_recognition_confusion_matrix.png"
FEATURE_PREFIX = "Feature"
RANDOM_STATE = 42
N_SPLITS = 3
UNKNOWN_THRESHOLD = 0.6
DISTANCE_MARGIN = 1.5  # multiplier on max observed intra-class distance (novelty guard)"""
        ),
        md("## 1. Load & validate"),
        code(
            """df = pd.read_csv(DATA_PATH, encoding="utf-8", sep=",")
print(f"[OK] Loaded {DATA_PATH.name}  shape={df.shape}")

required = ["Person", "Emotion"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"{DATA_PATH.name} missing columns: {missing}")

feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
if not feature_cols:
    raise ValueError(f"No columns starting with '{FEATURE_PREFIX}' in {DATA_PATH.name}")

assert df[feature_cols].isnull().sum().sum() == 0, "Embeddings contain nulls"
assert df["Person"].nunique() >= 2, "Need at least 2 known people to train a classifier"

print("People:", df["Person"].value_counts().to_dict())
print("Feature columns:", len(feature_cols))
print(df[["Person", "Image", "Emotion"]].head())"""
        ),
        md(
            """## 2. Group-aware labels

Each `(Person, Emotion)` pair is one physical photo; its augmented variants must stay
together on one side of any split."""
        ),
        code(
            """df["group"] = df["Person"].astype(str) + "_" + df["Emotion"].astype(str)

label = LabelEncoder()
y = label.fit_transform(df["Person"])
X = df[feature_cols].to_numpy()
groups = df["group"].to_numpy()

print("Classes:", list(label.classes_))
print("Unique (Person, Emotion) photos:", df["group"].nunique())"""
        ),
        md("## 3. Model pipeline"),
        code(
            """def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )"""
        ),
        md(
            """## 4. Grouped, stratified cross-validation

`StratifiedGroupKFold` keeps every photo's 4 variants together while still balancing people
across folds. Each fold asserts no photo group appears on both sides of the split."""
        ),
        code(
            """cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_accuracy, fold_f1 = [], []
holdout_idx = None  # keep fold 0's split for the detailed holdout report below

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
    assert set(groups[train_idx]).isdisjoint(groups[test_idx]), (
        f"Photo leakage across fold {fold} split"
    )

    pipe = build_pipeline()
    pipe.fit(X[train_idx], y[train_idx])
    preds = pipe.predict(X[test_idx])

    acc = accuracy_score(y[test_idx], preds)
    f1 = f1_score(y[test_idx], preds, average="macro", zero_division=0)
    fold_accuracy.append(acc)
    fold_f1.append(f1)
    print(
        f"Fold {fold}: accuracy={acc:.4f}  f1_macro={f1:.4f}  "
        f"train={len(train_idx)} test={len(test_idx)}"
    )

    if holdout_idx is None:
        holdout_idx = (train_idx, test_idx)

fold_accuracy = np.array(fold_accuracy)
fold_f1 = np.array(fold_f1)
print(f"\\nMean accuracy: {fold_accuracy.mean():.4f}  std: {fold_accuracy.std():.4f}")
print(f"Mean F1 (macro): {fold_f1.mean():.4f}  std: {fold_f1.std():.4f}")"""
        ),
        md("## 5. Holdout evaluation (fold 0) + threshold-based unknown handling"),
        code(
            """train_idx, test_idx = holdout_idx
model = build_pipeline()
model.fit(X[train_idx], y[train_idx])

y_test = y[test_idx]
y_proba = model.predict_proba(X[test_idx])
y_pred = model.predict(X[test_idx])
labels_idx = list(range(len(label.classes_)))

accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro", labels=labels_idx, zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average="weighted", labels=labels_idx, zero_division=0)

log_loss_value = None
log_loss_note = ""
try:
    log_loss_value = float(log_loss(y_test, y_proba, labels=labels_idx))
    if np.isnan(log_loss_value) or np.isinf(log_loss_value):
        raise ValueError("Log loss is undefined (NaN/Inf).")
except ValueError as exc:
    log_loss_value = None
    log_loss_note = f"Log loss could not be computed: {exc}."

clf_report = classification_report(
    y_test, y_pred, labels=labels_idx, target_names=label.classes_, zero_division=0
)

# Threshold-based rejection: below UNKNOWN_THRESHOLD confidence => treat as UNKNOWN
max_proba = y_proba.max(axis=1)
predicted_labels = np.where(
    max_proba >= UNKNOWN_THRESHOLD, label.inverse_transform(y_pred), "UNKNOWN"
)
n_rejected = int((predicted_labels == "UNKNOWN").sum())

print(f"Accuracy:      {accuracy:.4f}")
print(f"F1 (macro):    {f1_macro:.4f}")
print(f"F1 (weighted): {f1_weighted:.4f}")
print(f"Log loss:      {log_loss_value:.4f}" if log_loss_value is not None else log_loss_note)
print(
    f"\\nUNKNOWN_THRESHOLD={UNKNOWN_THRESHOLD} -> {n_rejected}/{len(test_idx)} holdout "
    "predictions rejected as UNKNOWN (all rows here are known people; expected low/zero)"
)
print()
print(clf_report)

cm = confusion_matrix(y_test, y_pred, labels=labels_idx)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label.classes_,
    yticklabels=label.classes_,
    ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Facial Recognition — Confusion Matrix (holdout fold)")
plt.tight_layout()
REPORT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(CM_PATH, dpi=120, bbox_inches="tight")
plt.show()
print(f"Saved confusion matrix -> {CM_PATH.resolve()}")"""
        ),
        md(
            """## 6. Fit final model on all data + persist

The holdout/CV folds above are for evaluation only; the deployed model is refit on every
available row so the CLI demo (Task 6) has access to all known people/photos."""
        ),
        code(
            """final_model = build_pipeline()
final_model.fit(X, y)

# Softmax/sigmoid confidence from a linear classifier is not a reliable unknown-
# input detector: LogisticRegression extrapolates *confidently* far outside the
# training distribution (verified empirically — random noise scored >0.99 "known
# person" confidence). A distance-to-centroid novelty check catches this: reject
# as UNKNOWN if the scaled sample is farther from its predicted class centroid
# than any training point was, times a safety margin.
scaler = final_model.named_steps["scaler"]
X_scaled = scaler.transform(X)
centroids: dict[str, list[float]] = {}
max_intra_class_distance: dict[str, float] = {}
for class_idx, class_name in enumerate(label.classes_):
    class_points = X_scaled[y == class_idx]
    centroid = class_points.mean(axis=0)
    centroids[class_name] = centroid.tolist()
    max_intra_class_distance[class_name] = float(
        np.linalg.norm(class_points - centroid, axis=1).max()
    )

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(final_model, MODEL_PATH)
joblib.dump(label, ENCODER_PATH)

config = {
    "unknown_threshold": UNKNOWN_THRESHOLD,
    "feature_columns": feature_cols,
    "classes": list(label.classes_),
    "centroids": centroids,
    "max_intra_class_distance": max_intra_class_distance,
    "distance_margin": DISTANCE_MARGIN,
}
CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

print(f"Saved model   -> {MODEL_PATH.resolve()}")
print(f"Saved encoder -> {ENCODER_PATH.resolve()}")
print(f"Saved config  -> {CONFIG_PATH.resolve()}")
print("Max intra-class distance (scaled space):", max_intra_class_distance)"""
        ),
        md("## 7. Export evaluation report"),
        code(
            """def _fmt(v):
    return f"{v:.4f}" if v is not None else "N/A"

log_loss_section = f"\\n> {log_loss_note}\\n" if log_loss_note else ""

report = f\"\"\"# Facial Recognition — Evaluation Report

Generated by `notebooks/03_facial_recognition.ipynb`.

## Dataset statistics

| Item | Value |
|------|-------|
| Source file | `{DATA_PATH.name}` |
| Rows (image variants) | {len(df)} |
| Unique source photos (groups) | {df['group'].nunique()} |
| Known people | {len(label.classes_)} ({', '.join(label.classes_)}) |
| Embedding dimensions | {len(feature_cols)} |
| CV folds | {N_SPLITS} (StratifiedGroupKFold, grouped by Person+Emotion photo) |
| Unknown threshold | {UNKNOWN_THRESHOLD} |

## Cross-validation (grouped, every photo rotates through test)

| Metric | Fold scores | Mean | Std |
|--------|-------------|------|-----|
| Accuracy | {np.round(fold_accuracy, 4).tolist()} | {fold_accuracy.mean():.4f} | {fold_accuracy.std():.4f} |
| F1 (macro) | {np.round(fold_f1, 4).tolist()} | {fold_f1.mean():.4f} | {fold_f1.std():.4f} |

## Holdout metrics (fold 0)

| Metric | Value |
|--------|-------|
| Accuracy | {_fmt(accuracy)} |
| F1 (macro) | {_fmt(f1_macro)} |
| F1 (weighted) | {_fmt(f1_weighted)} |
| Log loss | {_fmt(log_loss_value)} |
{log_loss_section}
## Classification report (holdout fold)

```
{clf_report}
```

## Confusion matrix

![Confusion matrix](face_recognition_confusion_matrix.png)

## Unknown / unauthorized handling

Predictions with max class probability below `{UNKNOWN_THRESHOLD}` are rejected as `UNKNOWN`
rather than forced to the nearest known class. On this holdout fold (every row belongs to a
known person), {n_rejected}/{len(test_idx)} predictions were rejected — expected to be low or
zero, since none of these test rows are genuinely unauthorized. True unauthorized-attempt
behavior (a face never seen during training) is exercised in the Task 6 CLI demo, not in this
dataset.

## Notes

- **Identification, not open-set verification**: the model classifies "which known person",
  with unknown-identity rejection handled via the probability threshold above rather than
  learned from negative examples.
- **Group-aware evaluation**: augmented variants (grayscale/flip/rotate) of the same photo
  are kept together on one side of every split, so metrics reflect generalization to a new
  photo rather than a transformed duplicate of a training image.
- **Small-N caveat**: only {df['group'].nunique()} source photos across {len(label.classes_)}
  people. Metrics are directional, not production-grade.
\"\"\"

REPORT_PATH.write_text(report, encoding="utf-8")
print(f"Saved evaluation report -> {REPORT_PATH.resolve()}")"""
        ),
    ]
    write_nb(NB_DIR / "03_facial_recognition.ipynb", cells)


def build_04() -> None:
    cells = [
        md(
            """# 04 — Voiceprint Verification Model

Trains a voice-identity classifier on the MFCC / spectral features produced by
`process_audio.py` (`data/audio_features.csv`).

**Decisions**
- Features: `MFCC_1`..`MFCC_13`, `SpectralRolloff`, `SpectralCentroid`, `RMSEnergy`,
  `ZeroCrossingRate`.
- Target: `Person` — an **identification** model ("whose voice is this?"), not open-set
  verification — same framing as notebook 03.
- **Group-aware split**: each `(Person, Phrase)` pair is one physical recording; its 4
  augmented variants (original/pitch-shift/time-stretch/noise) are near-duplicates. Splits
  group all variants of a recording onto the same side via `StratifiedGroupKFold`. Only 2
  recordings per person exist, so `N_SPLITS=2` (each fold holds out exactly 1 recording's
  4 rows per person).
- **Model**: `StandardScaler` + `LogisticRegression`, same architecture as the facial
  recognition model, for consistency across modalities.
- **Threshold-based unknown handling**: same `UNKNOWN_THRESHOLD` mechanism as notebook 03 —
  max class probability below threshold => `UNKNOWN`.
- **Distance-based novelty guard**: discovered necessary while testing the Task 6 CLI demo —
  synthetic white-noise audio was classified as `person1` with 0.99 confidence, because a
  linear classifier extrapolates confidently far outside its training distribution instead
  of expressing uncertainty. A sample is now only accepted if it is *also* within
  `DISTANCE_MARGIN` × the largest observed intra-class distance from its predicted class's
  centroid (in scaled feature space); otherwise it is rejected as `UNKNOWN` regardless of
  confidence.
- **Small-N caveat**: only 2 source recordings per person (4 groups total) — metrics are
  directional, not production-grade."""
        ),
        code(
            """from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = Path("../data/audio_features.csv")
MODEL_PATH = Path("../models/voice_verification_model.pkl")
ENCODER_PATH = Path("../models/voice_label_encoder.pkl")
CONFIG_PATH = Path("../models/voice_verification_config.json")
REPORT_DIR = Path("../reports")
REPORT_PATH = REPORT_DIR / "voice_verification_report.md"
CM_PATH = REPORT_DIR / "voice_verification_confusion_matrix.png"
NON_FEATURE_COLS = {"Person", "Audio", "Phrase", "SourcePath"}
RANDOM_STATE = 42
N_SPLITS = 2
UNKNOWN_THRESHOLD = 0.6
DISTANCE_MARGIN = 1.5  # multiplier on max observed intra-class distance (novelty guard)"""
        ),
        md("## 1. Load & validate"),
        code(
            """df = pd.read_csv(DATA_PATH, encoding="utf-8", sep=",")
print(f"[OK] Loaded {DATA_PATH.name}  shape={df.shape}")

required = ["Person", "Phrase"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"{DATA_PATH.name} missing columns: {missing}")

feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
if not feature_cols:
    raise ValueError(f"No feature columns found in {DATA_PATH.name}")

assert df[feature_cols].isnull().sum().sum() == 0, "Audio features contain nulls"
assert df["Person"].nunique() >= 2, "Need at least 2 known people to train a classifier"

print("People:", df["Person"].value_counts().to_dict())
print("Feature columns:", len(feature_cols))
print(df[["Person", "Audio", "Phrase"]].head(20))"""
        ),
        md(
            """## 2. Group-aware labels

Each `(Person, Phrase)` pair is one physical recording; its augmented variants must stay
together on one side of any split."""
        ),
        code(
            """df["group"] = df["Person"].astype(str) + "_" + df["Phrase"].astype(str)

label = LabelEncoder()
y = label.fit_transform(df["Person"])
X = df[feature_cols].to_numpy()
groups = df["group"].to_numpy()

print("Classes:", list(label.classes_))
print("Unique (Person, Phrase) recordings:", df["group"].nunique())"""
        ),
        md("## 3. Model pipeline"),
        code(
            """def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )"""
        ),
        md(
            """## 4. Grouped, stratified cross-validation

`StratifiedGroupKFold` keeps every recording's 4 variants together while balancing people
across folds. Each fold asserts no recording group appears on both sides of the split."""
        ),
        code(
            """cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_accuracy, fold_f1 = [], []
holdout_idx = None  # keep fold 0's split for the detailed holdout report below

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
    assert set(groups[train_idx]).isdisjoint(groups[test_idx]), (
        f"Recording leakage across fold {fold} split"
    )

    pipe = build_pipeline()
    pipe.fit(X[train_idx], y[train_idx])
    preds = pipe.predict(X[test_idx])

    acc = accuracy_score(y[test_idx], preds)
    f1 = f1_score(y[test_idx], preds, average="macro", zero_division=0)
    fold_accuracy.append(acc)
    fold_f1.append(f1)
    print(
        f"Fold {fold}: accuracy={acc:.4f}  f1_macro={f1:.4f}  "
        f"train={len(train_idx)} test={len(test_idx)}"
    )

    if holdout_idx is None:
        holdout_idx = (train_idx, test_idx)

fold_accuracy = np.array(fold_accuracy)
fold_f1 = np.array(fold_f1)
print(f"\\nMean accuracy: {fold_accuracy.mean():.4f}  std: {fold_accuracy.std():.4f}")
print(f"Mean F1 (macro): {fold_f1.mean():.4f}  std: {fold_f1.std():.4f}")"""
        ),
        md("## 5. Holdout evaluation (fold 0) + threshold-based unknown handling"),
        code(
            """train_idx, test_idx = holdout_idx
model = build_pipeline()
model.fit(X[train_idx], y[train_idx])

y_test = y[test_idx]
y_proba = model.predict_proba(X[test_idx])
y_pred = model.predict(X[test_idx])
labels_idx = list(range(len(label.classes_)))

accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro", labels=labels_idx, zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average="weighted", labels=labels_idx, zero_division=0)

log_loss_value = None
log_loss_note = ""
try:
    log_loss_value = float(log_loss(y_test, y_proba, labels=labels_idx))
    if np.isnan(log_loss_value) or np.isinf(log_loss_value):
        raise ValueError("Log loss is undefined (NaN/Inf).")
except ValueError as exc:
    log_loss_value = None
    log_loss_note = f"Log loss could not be computed: {exc}."

clf_report = classification_report(
    y_test, y_pred, labels=labels_idx, target_names=label.classes_, zero_division=0
)

# Threshold-based rejection: below UNKNOWN_THRESHOLD confidence => treat as UNKNOWN
max_proba = y_proba.max(axis=1)
predicted_labels = np.where(
    max_proba >= UNKNOWN_THRESHOLD, label.inverse_transform(y_pred), "UNKNOWN"
)
n_rejected = int((predicted_labels == "UNKNOWN").sum())

print(f"Accuracy:      {accuracy:.4f}")
print(f"F1 (macro):    {f1_macro:.4f}")
print(f"F1 (weighted): {f1_weighted:.4f}")
print(f"Log loss:      {log_loss_value:.4f}" if log_loss_value is not None else log_loss_note)
print(
    f"\\nUNKNOWN_THRESHOLD={UNKNOWN_THRESHOLD} -> {n_rejected}/{len(test_idx)} holdout "
    "predictions rejected as UNKNOWN (all rows here are known people; expected low/zero)"
)
print()
print(clf_report)

cm = confusion_matrix(y_test, y_pred, labels=labels_idx)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label.classes_,
    yticklabels=label.classes_,
    ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Voiceprint Verification — Confusion Matrix (holdout fold)")
plt.tight_layout()
REPORT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(CM_PATH, dpi=120, bbox_inches="tight")
plt.show()
print(f"Saved confusion matrix -> {CM_PATH.resolve()}")"""
        ),
        md(
            """## 6. Fit final model on all data + persist

The holdout/CV folds above are for evaluation only; the deployed model is refit on every
available row so the CLI demo (Task 6) has access to all known people/recordings."""
        ),
        code(
            """final_model = build_pipeline()
final_model.fit(X, y)

# Softmax/sigmoid confidence from a linear classifier is not a reliable unknown-
# input detector: LogisticRegression extrapolates *confidently* far outside the
# training distribution (verified empirically — random noise scored >0.99 "known
# person" confidence). A distance-to-centroid novelty check catches this: reject
# as UNKNOWN if the scaled sample is farther from its predicted class centroid
# than any training point was, times a safety margin.
scaler = final_model.named_steps["scaler"]
X_scaled = scaler.transform(X)
centroids: dict[str, list[float]] = {}
max_intra_class_distance: dict[str, float] = {}
for class_idx, class_name in enumerate(label.classes_):
    class_points = X_scaled[y == class_idx]
    centroid = class_points.mean(axis=0)
    centroids[class_name] = centroid.tolist()
    max_intra_class_distance[class_name] = float(
        np.linalg.norm(class_points - centroid, axis=1).max()
    )

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(final_model, MODEL_PATH)
joblib.dump(label, ENCODER_PATH)

config = {
    "unknown_threshold": UNKNOWN_THRESHOLD,
    "feature_columns": feature_cols,
    "classes": list(label.classes_),
    "centroids": centroids,
    "max_intra_class_distance": max_intra_class_distance,
    "distance_margin": DISTANCE_MARGIN,
}
CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

print(f"Saved model   -> {MODEL_PATH.resolve()}")
print(f"Saved encoder -> {ENCODER_PATH.resolve()}")
print(f"Saved config  -> {CONFIG_PATH.resolve()}")
print("Max intra-class distance (scaled space):", max_intra_class_distance)"""
        ),
        md("## 7. Export evaluation report"),
        code(
            """def _fmt(v):
    return f"{v:.4f}" if v is not None else "N/A"

log_loss_section = f"\\n> {log_loss_note}\\n" if log_loss_note else ""

report = f\"\"\"# Voiceprint Verification — Evaluation Report

Generated by `notebooks/04_voiceprint_verification.ipynb`.

## Dataset statistics

| Item | Value |
|------|-------|
| Source file | `{DATA_PATH.name}` |
| Rows (audio variants) | {len(df)} |
| Unique source recordings (groups) | {df['group'].nunique()} |
| Known people | {len(label.classes_)} ({', '.join(label.classes_)}) |
| Feature dimensions | {len(feature_cols)} |
| CV folds | {N_SPLITS} (StratifiedGroupKFold, grouped by Person+Phrase recording) |
| Unknown threshold | {UNKNOWN_THRESHOLD} |

## Cross-validation (grouped, every recording rotates through test)

| Metric | Fold scores | Mean | Std |
|--------|-------------|------|-----|
| Accuracy | {np.round(fold_accuracy, 4).tolist()} | {fold_accuracy.mean():.4f} | {fold_accuracy.std():.4f} |
| F1 (macro) | {np.round(fold_f1, 4).tolist()} | {fold_f1.mean():.4f} | {fold_f1.std():.4f} |

## Holdout metrics (fold 0)

| Metric | Value |
|--------|-------|
| Accuracy | {_fmt(accuracy)} |
| F1 (macro) | {_fmt(f1_macro)} |
| F1 (weighted) | {_fmt(f1_weighted)} |
| Log loss | {_fmt(log_loss_value)} |
{log_loss_section}
## Classification report (holdout fold)

```
{clf_report}
```

## Confusion matrix

![Confusion matrix](voice_verification_confusion_matrix.png)

## Unknown / unauthorized handling

Predictions with max class probability below `{UNKNOWN_THRESHOLD}` are rejected as `UNKNOWN`
rather than forced to the nearest known class. On this holdout fold (every row belongs to a
known person), {n_rejected}/{len(test_idx)} predictions were rejected — expected to be low or
zero, since none of these test rows are genuinely unauthorized. True unauthorized-attempt
behavior (a voice never seen during training) is exercised in the Task 6 CLI demo, not in
this dataset.

## Notes

- **Identification, not open-set verification**: the model classifies "which known person",
  with unknown-identity rejection handled via the probability threshold above rather than
  learned from negative examples.
- **Group-aware evaluation**: augmented variants (pitch-shift/time-stretch/noise) of the
  same recording are kept together on one side of every split, so metrics reflect
  generalization to a new recording rather than a transformed duplicate of a training clip.
- **Small-N caveat**: only {df['group'].nunique()} source recordings across
  {len(label.classes_)} people. Metrics are directional, not production-grade.
\"\"\"

REPORT_PATH.write_text(report, encoding="utf-8")
print(f"Saved evaluation report -> {REPORT_PATH.resolve()}")"""
        ),
    ]
    write_nb(NB_DIR / "04_voiceprint_verification.ipynb", cells)


if __name__ == "__main__":
    build_01()
    build_02()
    build_03()
    build_04()
