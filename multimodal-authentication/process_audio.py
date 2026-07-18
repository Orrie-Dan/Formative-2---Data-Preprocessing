"""
Phase 3 / Task 3 — Audio processing & voiceprint feature pipeline.

Loads two voice-command recordings per member ("Yes, approve" and
"Confirm transaction"), plots waveforms/spectrograms, applies
augmentations (pitch shift, time stretch, background noise), extracts
MFCC / spectral roll-off / energy features, and writes
data/audio_features.csv for the Voiceprint Verification Model
(notebooks/03_voice_verification.ipynb).
"""

from __future__ import annotations

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
AUG_DIR = AUDIO_DIR / "augmented"
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports" / "audio"

SAMPLE_RATE = 22050

# customer_id follows the "C0xx" scheme referenced in the README (kept
# distinct from tabular a-prefixed IDs until the two datasets are linked).
PERSONS = {
    "person1": "C001",
    "person2": "C002",
    "person3": "C003",
}
PHRASES = {
    "yes_approve": "Yes, approve",
    "confirm_transaction": "Confirm transaction",
}


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio: {path}")
    return y, sr


def plot_waveform(y: np.ndarray, sr: int, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 2.5))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(f"Waveform — {title}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_spectrogram(y: np.ndarray, sr: int, title: str, out_path: Path) -> None:
    stft = librosa.stft(y)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    img = librosa.display.specshow(db, sr=sr, x_axis="time", y_axis="log", ax=ax)
    ax.set_title(f"Spectrogram — {title}")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def augment_pitch(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def augment_time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    return librosa.effects.time_stretch(y, rate=rate)


def augment_background_noise(y: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """Add Gaussian noise at a target signal-to-noise ratio."""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(y**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=y.shape)
    return (y + noise).astype(np.float32)


def build_augmented_variants(y: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    return {
        "original": y,
        "pitch_up": augment_pitch(y, sr, n_steps=3),
        "pitch_down": augment_pitch(y, sr, n_steps=-3),
        "stretch_fast": augment_time_stretch(y, rate=1.2),
        "stretch_slow": augment_time_stretch(y, rate=0.8),
        "noise_low": augment_background_noise(y, snr_db=30, seed=1),
        "noise_high": augment_background_noise(y, snr_db=15, seed=2),
    }


def extract_features(y: np.ndarray, sr: int) -> dict[str, float]:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)

    features: dict[str, float] = {}
    for i in range(mfcc.shape[0]):
        features[f"mfcc{i + 1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc{i + 1}_std"] = float(np.std(mfcc[i]))
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))
    features["energy_mean"] = float(np.mean(rmse))
    features["energy_std"] = float(np.std(rmse))
    return features


def build_feature_table(show: bool = False) -> pd.DataFrame:
    rows = []

    for person, customer_id in PERSONS.items():
        person_dir = AUDIO_DIR / person
        for phrase, text in PHRASES.items():
            src = person_dir / f"{phrase}.wav"
            y, sr = load_audio(src)

            plot_waveform(y, sr, f"{person}/{phrase}", REPORT_DIR / "waveforms" / f"{person}_{phrase}.png")
            plot_spectrogram(y, sr, f"{person}/{phrase}", REPORT_DIR / "spectrograms" / f"{person}_{phrase}.png")

            variants = build_augmented_variants(y, sr)
            out_dir = AUG_DIR / person
            out_dir.mkdir(parents=True, exist_ok=True)

            for aug_name, variant in variants.items():
                filename = f"{phrase}_{aug_name}.wav"
                out_path = out_dir / filename
                sf.write(str(out_path), variant, sr)

                feats = extract_features(variant, sr)
                row = {
                    "customer_id": customer_id,
                    "person_id": person,
                    "phrase": phrase,
                    "text": text,
                    "augmentation": aug_name,
                    "file": str(out_path.relative_to(BASE_DIR)),
                }
                row.update(feats)
                rows.append(row)

    return pd.DataFrame(rows)


def main(show: bool = False) -> pd.DataFrame:
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    features = build_feature_table(show=show)
    out_csv = DATA_DIR / "audio_features.csv"
    features.to_csv(out_csv, index=False)

    print(f"Saved {len(features)} rows -> {out_csv}")
    print("Persons:", list(PERSONS))
    print("Augmentations per phrase:", len(build_augmented_variants(*load_audio(AUDIO_DIR / 'person1' / 'yes_approve.wav'))))
    return features


if __name__ == "__main__":
    main(show=False)
