"""
Task 6 — System Demonstration (command-line app).

Chains face recognition -> voice verification -> product recommendation,
denying access at whichever step fails. Run:

    python cli_app.py demo                          # canned authorized + unauthorized scenarios
    python cli_app.py transaction --face F --voice V # a specific face/voice pair
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import soundfile as sf

from process_audio import SAMPLE_RATE, extract_audio_features, load_audio
from process_images import extract_encoding

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# No real link exists yet between biometric identities (person1/person2) and the
# tabular customer_id space — this mapping is an explicit, documented stand-in
# until Task 1/2/3 data collection is unified under shared customer IDs.
PERSON_TO_CUSTOMER = {"person1": "a178", "person2": "a190"}

# Must match the drop list in notebooks/02_product_recommendation.ipynb — these
# share a transaction source with the target and were excluded from training.
TRANSACTION_LEAK_COLS = [
    "total_spent",
    "average_spent",
    "purchase_count",
    "days_since_last_purchase",
    "fraud_rate",
    "last_purchase",
]

UNAUTH_IMAGE_PATH = BASE_DIR / "images" / "unauthorized" / "unknown_face.jpg"
UNAUTH_AUDIO_PATH = BASE_DIR / "audio" / "unauthorized" / "unknown_voice.wav"


def ensure_unauthorized_samples() -> None:
    """Create synthetic negative-test fixtures (random noise, no real identity)."""
    UNAUTH_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not UNAUTH_IMAGE_PATH.exists():
        rng = np.random.default_rng(seed=7)
        noise_img = rng.integers(0, 256, size=(240, 240, 3), dtype=np.uint8)
        cv2.imwrite(str(UNAUTH_IMAGE_PATH), noise_img)

    UNAUTH_AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not UNAUTH_AUDIO_PATH.exists():
        rng = np.random.default_rng(seed=11)
        noise_wave = (0.2 * rng.standard_normal(SAMPLE_RATE * 2)).astype(np.float32)
        sf.write(str(UNAUTH_AUDIO_PATH), noise_wave, SAMPLE_RATE)


class _ThresholdVerifier:
    """Shared accept/reject logic for face and voice verifiers.

    Softmax confidence alone is not a reliable unknown-input detector — a linear
    classifier extrapolates *confidently* far outside its training distribution
    (verified empirically: random noise scored >0.99 "known person" confidence).
    So a candidate is only accepted if it clears BOTH the confidence threshold
    AND a distance-to-centroid novelty check in the model's scaled feature space.
    """

    model_filename: str
    encoder_filename: str
    config_filename: str

    def __init__(self) -> None:
        self.model = joblib.load(MODELS_DIR / self.model_filename)
        self.label = joblib.load(MODELS_DIR / self.encoder_filename)
        config = json.loads((MODELS_DIR / self.config_filename).read_text())
        self.threshold = config["unknown_threshold"]
        self.feature_columns = config["feature_columns"]
        self.centroids = {k: np.array(v) for k, v in config["centroids"].items()}
        self.max_intra_class_distance = config["max_intra_class_distance"]
        self.distance_margin = config["distance_margin"]

    def _classify(self, feature_row: list[float]) -> tuple[str | None, float]:
        proba = self.model.predict_proba([feature_row])[0]
        idx = int(np.argmax(proba))
        confidence = float(proba[idx])
        class_name = str(self.label.inverse_transform([idx])[0])

        if confidence < self.threshold:
            return None, confidence

        scaled = self.model.named_steps["scaler"].transform([feature_row])[0]
        distance = float(np.linalg.norm(scaled - self.centroids[class_name]))
        allowed = self.max_intra_class_distance[class_name] * self.distance_margin
        if distance > allowed:
            return None, confidence

        return class_name, confidence


class FaceVerifier(_ThresholdVerifier):
    model_filename = "face_recognition_model.pkl"
    encoder_filename = "face_label_encoder.pkl"
    config_filename = "face_recognition_config.json"

    def verify(self, image_path: Path) -> tuple[str | None, float]:
        encoding = extract_encoding(image_path)
        if encoding is None:
            return None, 0.0
        return self._classify(list(encoding))


class VoiceVerifier(_ThresholdVerifier):
    model_filename = "voice_verification_model.pkl"
    encoder_filename = "voice_label_encoder.pkl"
    config_filename = "voice_verification_config.json"

    def verify(self, audio_path: Path) -> tuple[str | None, float]:
        y, sr = load_audio(audio_path)
        features = extract_audio_features(y, sr)
        row = [features[c] for c in self.feature_columns]
        return self._classify(row)


class ProductRecommender:
    def __init__(self) -> None:
        self.model = joblib.load(MODELS_DIR / "product_model.pkl")
        self.label = joblib.load(MODELS_DIR / "product_label_encoder.pkl")
        self.df = pd.read_csv(DATA_DIR / "modeling_dataset.csv", encoding="utf-8", sep=",")

    def predict(self, customer_id: str) -> str:
        row = self.df[self.df["customer_id"].str.lower() == customer_id.lower()]
        if row.empty:
            raise ValueError(f"No modeling row found for customer_id={customer_id}")
        drop_cols = ["customer_id", "product_category"] + TRANSACTION_LEAK_COLS
        X = row.drop(columns=[c for c in drop_cols if c in row.columns])
        pred_idx = self.model.predict(X)[0]
        return str(self.label.inverse_transform([pred_idx])[0])


def run_transaction(
    face_path: Path,
    voice_path: Path,
    face_verifier: FaceVerifier,
    voice_verifier: VoiceVerifier,
    recommender: ProductRecommender,
) -> None:
    print(f"  Face input:  {face_path}")
    print(f"  Voice input: {voice_path}")

    print("\n[1/3] Face recognition...")
    face_person, face_conf = face_verifier.verify(face_path)
    if face_person is None:
        print(f"  ACCESS DENIED — face not recognized (confidence={face_conf:.2f})")
        return
    print(f"  MATCH: {face_person} (confidence={face_conf:.2f})")

    print("\n[2/3] Voice verification...")
    voice_person, voice_conf = voice_verifier.verify(voice_path)
    if voice_person is None:
        print(f"  ACCESS DENIED — voice not recognized (confidence={voice_conf:.2f})")
        return
    print(f"  MATCH: {voice_person} (confidence={voice_conf:.2f})")

    if voice_person != face_person:
        print(
            f"  ACCESS DENIED — voice identity ({voice_person}) does not confirm "
            f"face identity ({face_person})"
        )
        return

    print(f"\n[3/3] Identity confirmed as {face_person}. Running product recommendation...")
    customer_id = PERSON_TO_CUSTOMER.get(face_person)
    if customer_id is None:
        print(f"  No linked customer_id for {face_person}; cannot run product model.")
        return

    prediction = recommender.predict(customer_id)
    print(f"  ACCESS GRANTED — recommended product category: '{prediction}'")


def cmd_transaction(args: argparse.Namespace) -> None:
    face_verifier = FaceVerifier()
    voice_verifier = VoiceVerifier()
    recommender = ProductRecommender()
    run_transaction(
        Path(args.face), Path(args.voice), face_verifier, voice_verifier, recommender
    )


def cmd_demo(_: argparse.Namespace) -> None:
    ensure_unauthorized_samples()
    face_verifier = FaceVerifier()
    voice_verifier = VoiceVerifier()
    recommender = ProductRecommender()

    scenarios = [
        (
            "Scenario A — Authorized transaction (person1 face + person1 voice)",
            BASE_DIR / "images" / "person1" / "neutral.jpg",
            BASE_DIR / "audio" / "person1" / "yes_approve.wav",
        ),
        (
            "Scenario B — Unauthorized attempt: unknown face (synthetic, no known match)",
            UNAUTH_IMAGE_PATH,
            BASE_DIR / "audio" / "person1" / "yes_approve.wav",
        ),
        (
            "Scenario C — Unauthorized attempt: unknown voice (synthetic, no known match)",
            BASE_DIR / "images" / "person1" / "neutral.jpg",
            UNAUTH_AUDIO_PATH,
        ),
        (
            "Scenario D — Identity mismatch (person1 face + person2 voice)",
            BASE_DIR / "images" / "person1" / "neutral.jpg",
            BASE_DIR / "audio" / "person2" / "yes_approve.wav",
        ),
    ]

    for title, face_path, voice_path in scenarios:
        print("=" * 70)
        print(title)
        print("=" * 70)
        run_transaction(face_path, voice_path, face_verifier, voice_verifier, recommender)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal authentication CLI demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser(
        "demo", help="Run canned authorized + unauthorized scenarios"
    )
    demo_parser.set_defaults(func=cmd_demo)

    tx_parser = subparsers.add_parser(
        "transaction", help="Run one face/voice pair through the pipeline"
    )
    tx_parser.add_argument("--face", required=True, help="Path to a face image")
    tx_parser.add_argument("--voice", required=True, help="Path to a voice recording")
    tx_parser.set_defaults(func=cmd_transaction)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
