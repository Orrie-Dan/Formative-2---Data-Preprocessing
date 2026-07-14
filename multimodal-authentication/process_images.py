"""
Phase 3 / Task 2 — Image processing pipeline.

Loads person emotion images, applies augmentations, extracts 128-d
face_recognition embeddings, and writes data/image_features.csv.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import face_recognition
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
AUG_DIR = IMAGES_DIR / "augmented"
DATA_DIR = BASE_DIR / "data"
PERSONS = ["person1"]
EMOTIONS = ["neutral", "smile", "surprised"]


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def show_image(img_bgr: np.ndarray, title: str = "") -> None:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(3, 3))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def augment_and_save(person: str, emotion: str, img: np.ndarray) -> list[Path]:
    """Create grayscale, flip, and rotate variants and save them."""
    out_dir = AUG_DIR / person
    out_dir.mkdir(parents=True, exist_ok=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    flip = cv2.flip(img, 1)
    rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    saved = []
    variants = {
        f"{emotion}.jpg": img,
        f"{emotion}_gray.jpg": gray_bgr,
        f"{emotion}_flip.jpg": flip,
        f"{emotion}_rotate.jpg": rotate,
    }
    for name, variant in variants.items():
        path = out_dir / name
        cv2.imwrite(str(path), variant)
        saved.append(path)
    return saved


def extract_encoding(path: Path) -> np.ndarray | None:
    """Return a 128-d face embedding, or None if no face is detected.

    Rotated images often fail HOG detection, so we also try upright
    orientations derived from the same file.
    """
    image = face_recognition.load_image_file(str(path))
    candidates = [image]

    # Try all 90-degree orientations (helps rotate augmentations)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for rotate_flag in (
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_180,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
    ):
        rotated = cv2.rotate(bgr, rotate_flag)
        candidates.append(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

    for candidate in candidates:
        encodings = face_recognition.face_encodings(candidate)
        if encodings:
            return encodings[0]

        locations = face_recognition.face_locations(
            candidate, number_of_times_to_upsample=1
        )
        encodings = face_recognition.face_encodings(
            candidate, known_face_locations=locations
        )
        if encodings:
            return encodings[0]

    return None


def build_feature_table(show: bool = False) -> pd.DataFrame:
    rows = []

    for person in PERSONS:
        person_dir = IMAGES_DIR / person
        for emotion in EMOTIONS:
            src = person_dir / f"{emotion}.jpg"
            img = load_image(src)
            if show:
                show_image(img, f"{person} / {emotion}")

            saved_paths = augment_and_save(person, emotion, img)

            for path in saved_paths:
                encoding = extract_encoding(path)
                if encoding is None:
                    print(f"No face detected in {path}, skipping")
                    continue

                row = {
                    "Person": person,
                    "Image": path.name,
                    "Emotion": emotion,
                    "SourcePath": str(path.relative_to(BASE_DIR)),
                }
                for i, value in enumerate(encoding, start=1):
                    row[f"Feature{i}"] = float(value)
                rows.append(row)

    return pd.DataFrame(rows)


def main(show: bool = False) -> pd.DataFrame:
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    features = build_feature_table(show=show)
    out_csv = DATA_DIR / "image_features.csv"
    features.to_csv(out_csv, index=False)

    print(f"Saved {len(features)} rows -> {out_csv}")
    print("Columns:", list(features.columns[:5]), "...", f"Feature128 present={('Feature128' in features.columns)}")
    return features


if __name__ == "__main__":
    main(show=False)
