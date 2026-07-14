"""
Multimodal Authentication Demo App

Loads tabular, image, and audio feature data to score a simple
authentication risk decision for a selected customer.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"
AUDIO_DIR = BASE_DIR / "audio"


@st.cache_data
def load_data():
    required = {
        "profiles": DATA_DIR / "customer_social_profiles.csv",
        "transactions": DATA_DIR / "customer_transactions.csv",
        "merged": DATA_DIR / "merged_dataset.csv",
        "image_features": DATA_DIR / "image_features.csv",
        "audio_features": DATA_DIR / "audio_features.csv",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing data files: "
            + ", ".join(missing)
            + ". Run notebooks/01_data_preprocessing.ipynb first for merged_dataset.csv."
        )

    profiles = pd.read_csv(required["profiles"], encoding="utf-8", sep=",")
    transactions = pd.read_csv(required["transactions"], encoding="utf-8", sep=",")
    merged = pd.read_csv(required["merged"], encoding="utf-8", sep=",")
    image_features = pd.read_csv(required["image_features"], encoding="utf-8", sep=",")
    audio_features = pd.read_csv(required["audio_features"], encoding="utf-8", sep=",")
    return profiles, transactions, merged, image_features, audio_features


def _row_float(row: pd.Series, *keys: str, default: float) -> float:
    """Read the first present numeric column (supports notebook + script schemas)."""
    for key in keys:
        if key in row.index and pd.notna(row.get(key)):
            return float(row[key])
    return float(default)


def _sentiment_score(row: pd.Series) -> float:
    """Numeric sentiment on 0–2 scale used by the risk heuristic."""
    if "avg_sentiment_score" in row.index and pd.notna(row.get("avg_sentiment_score")):
        return float(row["avg_sentiment_score"])
    if "review_sentiment" in row.index and pd.notna(row.get("review_sentiment")):
        sentiment_map = {"negative": 0.0, "neutral": 1.0, "positive": 2.0}
        return sentiment_map.get(str(row["review_sentiment"]).strip().lower(), 1.0)
    return 1.0


def authentication_score(row: pd.Series) -> tuple[float, str, str]:
    """Heuristic multimodal risk score in [0, 1]. Higher = more risk."""
    score = 0.15
    score += 0.45 * _row_float(row, "fraud_rate", default=0.0)
    score += max(
        0.0,
        (3.0 - _row_float(row, "avg_purchase_interest", "purchase_interest_score", default=3.0))
        / 3.0,
    ) * 0.15
    score += max(
        0.0,
        (
            60.0
            - _row_float(
                row,
                "avg_engagement_score",
                "average_engagement",
                "engagement_score",
                default=60.0,
            )
        )
        / 60.0,
    ) * 0.15
    if _sentiment_score(row) < 0.75:
        score += 0.1
    score = min(max(score, 0.0), 1.0)

    if score < 0.25:
        decision = "APPROVED"
        label = "legitimate"
    elif score < 0.5:
        decision = "CHALLENGE (face + voice)"
        label = "review"
    else:
        decision = "DENIED"
        label = "suspicious"

    return score, decision, label


def filter_by_customer(df: pd.DataFrame, customer_id: str) -> pd.DataFrame:
    """Case-insensitive customer_id filter (merged IDs are lowercased; raw CSVs are not)."""
    return df[
        df["customer_id"].astype(str).str.strip().str.lower()
        == str(customer_id).strip().lower()
    ]


def main():
    st.set_page_config(page_title="Multimodal Authentication", layout="wide")
    st.title("Multimodal Authentication")
    st.caption("Social profile + transactions + face + voice signals")

    profiles, transactions, merged, image_features, audio_features = load_data()

    # One entry per customer (merged CSV is platform-level and repeats IDs)
    customer_ids = sorted(
        merged["customer_id"].astype(str).str.lower().unique().tolist()
    )
    selected = st.sidebar.selectbox("Customer", customer_ids)

    row = merged[merged["customer_id"] == selected].iloc[0]
    score, decision, label = authentication_score(row)

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk score", f"{score:.2f}")
    col2.metric("Auth label", label)
    col3.metric("Decision", decision)

    st.subheader("Customer profile")
    st.dataframe(filter_by_customer(profiles, selected), use_container_width=True)

    st.subheader("Engineered tabular features")
    st.dataframe(merged[merged["customer_id"] == selected], use_container_width=True)

    st.subheader("Recent transactions")
    st.dataframe(
        filter_by_customer(transactions, selected),
        use_container_width=True,
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Image modality")
        # Prefer person1 demo embeddings produced in Phase 3
        if "Person" in image_features.columns:
            person_imgs = image_features[image_features["Person"] == "person1"]
            preview_cols = [c for c in ["Person", "Image", "Emotion"] if c in person_imgs.columns]
            preview_cols += [c for c in person_imgs.columns if c.startswith("Feature")][:5]
            st.dataframe(person_imgs[preview_cols], use_container_width=True)
        else:
            person_imgs = image_features[image_features.get("customer_id", pd.Series(dtype=str)) == selected]
            st.dataframe(person_imgs, use_container_width=True)

        img_path = IMAGES_DIR / "person1" / "neutral.jpg"
        if img_path.exists():
            st.image(str(img_path), caption="person1 / neutral.jpg", width=224)
        else:
            st.info("No images found under images/person1/")

    with right:
        st.subheader("Audio modality")
        person_aud = audio_features[audio_features["customer_id"] == selected]
        st.dataframe(person_aud, use_container_width=True)

        person_id = (
            person_aud["person_id"].iloc[0] if not person_aud.empty else "person1"
        )
        wav_path = AUDIO_DIR / person_id / "yes_approve.wav"
        if wav_path.exists():
            st.audio(str(wav_path))
        else:
            st.info("No linked audio for this customer yet (demo uses person1).")
            demo = AUDIO_DIR / "person1" / "yes_approve.wav"
            if demo.exists():
                st.audio(str(demo))

    st.subheader("Full merged dataset")
    st.dataframe(merged, use_container_width=True)


if __name__ == "__main__":
    main()
