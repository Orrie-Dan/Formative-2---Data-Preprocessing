"""Prepare aligned tabular CSVs and build merged_dataset.csv (Phase 2 / Task 1)."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"


def prepare_profiles() -> pd.DataFrame:
    src = DATA / "customer_social_profiles - customer_social_profiles.csv"
    dest = DATA / "customer_social_profiles.csv"

    if dest.exists():
        profiles = pd.read_csv(dest)
    elif src.exists():
        profiles = pd.read_csv(src)
    else:
        raise FileNotFoundError("No social profiles CSV found in data/")

    if "customer_id_new" in profiles.columns:
        profiles = profiles.rename(columns={"customer_id_new": "customer_id"})

    profiles.to_csv(dest, index=False)
    return profiles


def prepare_transactions(customer_ids: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    merchants = [
        ("Pick n Pay", "Groceries", "Card"),
        ("Woolworths", "Retail", "Card"),
        ("Takealot", "Ecommerce", "Online"),
        ("Uber", "Transport", "App"),
        ("Shell", "Fuel", "Card"),
        ("Dischem", "Health", "Card"),
        ("Mr Price", "Clothing", "Card"),
        ("Nandos", "Food", "Card"),
        ("Makro", "Wholesale", "Card"),
        ("Travelstart", "Travel", "Online"),
        ("Unknown Vendor", "Other", "Online"),
        ("Crypto Exchange", "Finance", "Online"),
        ("Gift Cards Online", "Other", "Online"),
        ("Wire Transfer", "Finance", "Online"),
    ]
    safe_locations = [
        "Cape Town",
        "Johannesburg",
        "Durban",
        "Pretoria",
        "Soweto",
        "Bloemfontein",
        "Port Elizabeth",
    ]
    fraud_locations = ["Unknown", "Lagos", "Moscow"]
    devices = ["Mobile", "Desktop"]
    suspicious = {
        "Unknown Vendor",
        "Crypto Exchange",
        "Gift Cards Online",
        "Wire Transfer",
    }

    rows = []
    tid = 1
    base = datetime(2025, 1, 1)

    for cid in customer_ids:
        n_tx = int(rng.integers(2, 7))
        for _ in range(n_tx):
            merchant, category, channel = merchants[int(rng.integers(0, len(merchants)))]
            is_suspicious = merchant in suspicious
            if is_suspicious:
                amount = float(rng.uniform(5000, 25000))
                is_fraud = bool(rng.random() < 0.7)
                location = str(rng.choice(fraud_locations))
            else:
                amount = float(rng.uniform(50, 3500))
                is_fraud = False
                location = str(rng.choice(safe_locations))

            status = "Declined" if is_fraud else "Approved"
            ts = base + timedelta(
                days=int(rng.integers(0, 90)),
                hours=int(rng.integers(0, 24)),
                minutes=int(rng.integers(0, 60)),
            )
            rows.append(
                {
                    "transaction_id": f"T{tid:04d}",
                    "customer_id": cid,
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "amount": round(amount, 2),
                    "currency": "ZAR",
                    "merchant": merchant,
                    "category": category,
                    "channel": channel,
                    "device_type": str(rng.choice(devices)),
                    "location": location,
                    "status": status,
                    "is_fraud": is_fraud,
                }
            )
            tid += 1

    transactions = pd.DataFrame(rows)
    transactions.to_csv(DATA / "customer_transactions.csv", index=False)
    return transactions


def merge_and_engineer(profiles: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    # Step 3 — Merge
    merged = pd.merge(profiles, transactions, on="customer_id", how="inner")

    # Step 4 — Clean
    merged = merged.drop_duplicates()

    if "timestamp" in merged.columns:
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")

    if "amount" in merged.columns:
        merged["amount"] = pd.to_numeric(merged["amount"], errors="coerce")

    if "engagement_score" in merged.columns:
        merged["engagement_score"] = pd.to_numeric(
            merged["engagement_score"], errors="coerce"
        )

    if "purchase_interest_score" in merged.columns:
        merged["purchase_interest_score"] = pd.to_numeric(
            merged["purchase_interest_score"], errors="coerce"
        )

    if "is_fraud" in merged.columns:
        merged["is_fraud"] = (
            merged["is_fraud"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .fillna(False)
            .astype(bool)
        )

    numeric_cols = merged.select_dtypes(include=["number"]).columns
    merged[numeric_cols] = merged[numeric_cols].fillna(0)

    object_cols = merged.select_dtypes(include=["object"]).columns
    merged[object_cols] = merged[object_cols].fillna("Unknown")

    # Step 5 — Feature engineering (customer-level)
    sentiment_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    merged["sentiment_score"] = (
        merged["review_sentiment"].map(sentiment_map).fillna(1).astype(int)
    )

    customer_features = (
        merged.groupby("customer_id", as_index=False)
        .agg(
            purchase_count=("transaction_id", "nunique"),
            average_spent=("amount", "mean"),
            total_spent=("amount", "sum"),
            fraud_count=("is_fraud", "sum"),
            avg_engagement_score=("engagement_score", "mean"),
            avg_purchase_interest=("purchase_interest_score", "mean"),
            avg_sentiment_score=("sentiment_score", "mean"),
            platform_count=("social_media_platform", "nunique"),
            primary_platform=("social_media_platform", lambda s: s.mode().iloc[0]),
        )
        .round(
            {
                "average_spent": 2,
                "total_spent": 2,
                "avg_engagement_score": 2,
                "avg_purchase_interest": 2,
                "avg_sentiment_score": 2,
            }
        )
    )

    customer_features["fraud_rate"] = (
        customer_features["fraud_count"] / customer_features["purchase_count"]
    ).round(3)

    # Proxy social engagement features (dataset has no likes/comments columns)
    customer_features["likes"] = (
        customer_features["avg_engagement_score"] * 12
    ).round(0).astype(int)
    customer_features["comments"] = (
        customer_features["avg_engagement_score"] * 3
    ).round(0).astype(int)
    customer_features["followers"] = (
        customer_features["avg_engagement_score"] * 40
        + customer_features["platform_count"] * 100
    ).round(0).astype(int)

    return merged, customer_features


def main() -> None:
    """DEPRECATED entrypoint.

    Canonical merge lives in notebooks/01_data_preprocessing.ipynb.
    This script keeps legacy customer-level engineering but writes to a
    *separate* filename so it cannot overwrite the notebook deliverable.
    """
    print(
        "WARNING: prepare_merged_data.py is legacy. "
        "Use notebooks/01_data_preprocessing.ipynb to produce "
        "data/merged_dataset.csv for modeling and app.py."
    )
    profiles = prepare_profiles()
    customer_ids = sorted(profiles["customer_id"].unique())
    transactions = prepare_transactions(customer_ids)

    print("profiles:", profiles.shape)
    print("transactions:", transactions.shape)
    print(
        "common customer_id count:",
        len(set(profiles["customer_id"]) & set(transactions["customer_id"])),
    )

    row_level, customer_level = merge_and_engineer(profiles, transactions)

    row_level.to_csv(DATA / "merged_dataset_row_level.csv", index=False)
    # Do NOT overwrite merged_dataset.csv (notebook 01 is source of truth)
    customer_level.to_csv(DATA / "merged_dataset_legacy_customer.csv", index=False)

    print("row-level merged:", row_level.shape)
    print(
        "legacy customer-level written to merged_dataset_legacy_customer.csv:",
        customer_level.shape,
    )
    print(customer_level.head())


if __name__ == "__main__":
    main()
