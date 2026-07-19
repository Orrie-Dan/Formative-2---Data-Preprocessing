"""
Phase 3 / Task 3 — Audio processing pipeline.

Loads person voice-command recordings, displays waveforms/spectrograms,
applies augmentations, extracts MFCC/spectral features, and writes
data/audio_features.csv.
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
PLOTS_DIR = BASE_DIR / "reports" / "audio_plots"
PERSONS = ["person1", "person2"]
PHRASES = ["yes_approve", "confirm_transaction"]
SAMPLE_RATE = 22050
N_MFCC = 13


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio: {path}")
    return y, sr


def save_waveform_and_spectrogram(y: np.ndarray, sr: int, person: str, label: str) -> None:
    """Persist waveform + spectrogram PNGs for a clip (original or augmented)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 2.5))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(f"{person} / {label} — waveform")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{person}_{label}_waveform.png", dpi=110)
    plt.close(fig)

    stft = librosa.stft(y)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    img = librosa.display.specshow(db, sr=sr, x_axis="time", y_axis="hz", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"{person} / {label} — spectrogram")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{person}_{label}_spectrogram.png", dpi=110)
    plt.close(fig)


def augment_and_save(
    person: str, phrase: str, y: np.ndarray, sr: int
) -> dict[str, np.ndarray]:
    """Create pitch-shift, time-stretch, and background-noise variants and save them.

    Returns {label: waveform} including the original, keyed the same way the
    saved .wav filenames are, so callers can iterate features/plots consistently.
    """
    out_dir = AUG_DIR / person
    out_dir.mkdir(parents=True, exist_ok=True)

    pitch_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)
    stretched = librosa.effects.time_stretch(y, rate=0.85)

    rng = np.random.default_rng(seed=abs(hash((person, phrase))) % (2**32))
    noisy = y + 0.005 * rng.standard_normal(len(y)).astype(np.float32)

    variants = {
        phrase: y,
        f"{phrase}_pitch": pitch_shifted,
        f"{phrase}_stretch": stretched,
        f"{phrase}_noise": noisy,
    }

    for label, wav in variants.items():
        path = out_dir / f"{label}.wav"
        sf.write(str(path), wav, sr)

    return variants


def extract_audio_features(y: np.ndarray, sr: int) -> dict[str, float]:
    """MFCCs (time-averaged) + spectral roll-off + RMS energy + centroid + ZCR."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)

    features = {f"MFCC_{i + 1}": float(v) for i, v in enumerate(mfcc.mean(axis=1))}
    features["SpectralRolloff"] = float(rolloff.mean())
    features["SpectralCentroid"] = float(centroid.mean())
    features["RMSEnergy"] = float(rms.mean())
    features["ZeroCrossingRate"] = float(zcr.mean())
    return features


def build_feature_table(show: bool = False) -> pd.DataFrame:
    rows = []

    for person in PERSONS:
        person_dir = AUDIO_DIR / person
        for phrase in PHRASES:
            src = person_dir / f"{phrase}.wav"
            y, sr = load_audio(src)

            variants = augment_and_save(person, phrase, y, sr)

            for label, wav in variants.items():
                if show:
                    save_waveform_and_spectrogram(wav, sr, person, label)

                row = {
                    "Person": person,
                    "Audio": f"{label}.wav",
                    "Phrase": phrase,
                    "SourcePath": str(
                        (AUG_DIR / person / f"{label}.wav").relative_to(BASE_DIR)
                    ),
                }
                row.update(extract_audio_features(wav, sr))
                rows.append(row)

    return pd.DataFrame(rows)


def main(show: bool = True) -> pd.DataFrame:
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    features = build_feature_table(show=show)
    out_csv = DATA_DIR / "audio_features.csv"
    features.to_csv(out_csv, index=False)

    print(f"Saved {len(features)} rows -> {out_csv}")
    print("Columns:", list(features.columns))
    return features


if __name__ == "__main__":
    main(show=True)
