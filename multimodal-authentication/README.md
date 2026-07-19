# Multimodal Authentication

Customer authentication and product-recommendation pipeline combining **social profile**, **transaction**, **face image**, and **voice audio** signals: face recognition + voiceprint verification gate a product-recommendation model, denying access whenever either biometric check fails.

## Project structure

```
multimodal-authentication/
├── data/
│   ├── customer_social_profiles.csv
│   ├── customer_transactions.csv
│   ├── merged_dataset.csv           # platform-level join (notebook 01)
│   ├── modeling_dataset.csv         # one row per customer (notebook 01)
│   ├── image_features.csv           # face embeddings (process_images.py)
│   └── audio_features.csv           # MFCC / spectral features (process_audio.py)
├── images/
│   ├── person1/ person2/            # raw neutral/smile/surprised photos
│   ├── augmented/                   # grayscale/flip/rotate variants (generated)
│   └── unauthorized/                # synthetic no-match test fixture (generated)
├── audio/
│   ├── person1/ person2/            # raw "yes_approve" / "confirm_transaction" clips
│   ├── augmented/                   # pitch-shift/time-stretch/noise variants (generated)
│   └── unauthorized/                # synthetic no-match test fixture (generated)
├── models/
│   ├── product_model.pkl / product_label_encoder.pkl
│   ├── face_recognition_model.pkl / face_label_encoder.pkl / face_recognition_config.json
│   └── voice_verification_model.pkl / voice_label_encoder.pkl / voice_verification_config.json
├── reports/
│   ├── evaluation_report.md + confusion_matrix.png                       # product model
│   ├── face_recognition_report.md + face_recognition_confusion_matrix.png
│   ├── voice_verification_report.md + voice_verification_confusion_matrix.png
│   └── audio_plots/                 # waveform + spectrogram PNGs per recording
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_product_recommendation.ipynb
│   ├── 03_facial_recognition.ipynb
│   └── 04_voiceprint_verification.ipynb
├── scripts/
│   └── _build_notebooks.py          # regenerates all four notebooks from source
├── archive/                         # obsolete scripts (not part of the pipeline)
├── app.py                           # Streamlit data/heuristic viewer
├── process_images.py                # Task 2: image augmentation + embeddings
├── process_audio.py                 # Task 3: audio augmentation + features
├── cli_app.py                       # Task 6: face -> voice -> product CLI demo
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

Run in order (notebooks assume CWD = `notebooks/`; scripts assume CWD = `multimodal-authentication/`):

1. **Preprocessing** — `notebooks/01_data_preprocessing.ipynb`
2. **Image features** — `python process_images.py` (writes `data/image_features.csv`)
3. **Audio features** — `python process_audio.py` (writes `data/audio_features.csv`)
4. **Product recommendation** — `notebooks/02_product_recommendation.ipynb`
5. **Facial recognition** — `notebooks/03_facial_recognition.ipynb`
6. **Voiceprint verification** — `notebooks/04_voiceprint_verification.ipynb`
7. **CLI demo** — `python cli_app.py demo`
8. **(Optional) Data viewer** — `streamlit run app.py`

Notebooks are generated from `scripts/_build_notebooks.py` — regenerate with `python scripts/_build_notebooks.py` after editing that file, then re-execute each notebook.

Expected artifacts:

| Artifact | Produced by |
|----------|-------------|
| `data/merged_dataset.csv`, `data/modeling_dataset.csv` | Notebook 01 |
| `data/image_features.csv` | `process_images.py` |
| `data/audio_features.csv` | `process_audio.py` |
| `models/product_model.pkl`, `product_label_encoder.pkl` | Notebook 02 |
| `models/face_recognition_model.pkl`, `face_label_encoder.pkl`, `face_recognition_config.json` | Notebook 03 |
| `models/voice_verification_model.pkl`, `voice_label_encoder.pkl`, `voice_verification_config.json` | Notebook 04 |
| `reports/evaluation_report.md`, `face_recognition_report.md`, `voice_verification_report.md` | Notebooks 02/03/04 |

## Preprocessing workflow (Notebook 01)

1. **Load** social + transaction CSVs with `encoding="utf-8"`, `sep=","`; fail early if required columns are missing.
2. **Validate** before cleaning: dimensions, missing values, duplicate rows, duplicate `customer_id`s.
3. **Clean**: normalize IDs (strip + lowercase), standardize categoricals, coerce numerics/timestamps, drop bad rows, winsorize `amount` at 1st/99th percentile.
4. **Feature engineering (customer-level TX)**: RFM (`total_spent`, `average_spent`, `purchase_count`, `days_since_last_purchase`), `fraud_rate`, `product_category` = mode of TX `category`.
5. **Merge**: attach TX features + social aggregates (`average_engagement`, `platform_count`, `main_platform`) to social rows.
6. **Export**:
   - `merged_dataset.csv` — **platform-level** (intentional; multiple rows per customer when they have multiple social platforms). Used by the Streamlit auth demo.
   - `modeling_dataset.csv` — **one row per customer** (mean engagement / interest, mode sentiment, same TX aggregates). Used for recommendation training and by the CLI demo's product step.

## Image processing (Task 2 — `process_images.py`)

1. **Load** each person's `neutral.jpg` / `smile.jpg` / `surprised.jpg` from `images/<person>/`.
2. **Augment**: grayscale, horizontal flip, 90° rotation — saved to `images/augmented/<person>/`.
3. **Extract**: 128-d `face_recognition` embedding per image (original + 3 augmented variants), retrying upright orientations when a rotated crop fails face detection.
4. **Export** to `data/image_features.csv` (`Person`, `Image`, `Emotion`, `SourcePath`, `Feature1`…`Feature128`).

## Audio processing (Task 3 — `process_audio.py`)

1. **Load** each person's `yes_approve.wav` / `confirm_transaction.wav` from `audio/<person>/` (resampled to 22.05 kHz mono).
2. **Augment**: pitch shift, time stretch, background noise — saved to `audio/augmented/<person>/`.
3. **Display**: waveform + spectrogram PNGs per clip (original + 3 augmented variants) saved to `reports/audio_plots/`.
4. **Extract**: 13 MFCCs (time-averaged) + spectral roll-off + spectral centroid + RMS energy + zero-crossing rate.
5. **Export** to `data/audio_features.csv` (`Person`, `Audio`, `Phrase`, `SourcePath`, feature columns).

## Feature engineering pipeline

| Feature group | Source | Used for |
|---------------|--------|----------|
| RFM + `fraud_rate` | Transactions | Auth heuristic (`app.py`); stored in both CSVs |
| `product_category` | Mode of TX `category` | Recommendation **target** |
| Social scores / platform aggregates | Profiles | Recommendation **features** + auth heuristic |
| Face embeddings (`Feature1..128`) | `process_images.py` | Facial recognition **features** |
| MFCC / spectral stats | `process_audio.py` | Voiceprint verification **features** |

Recency uses a fixed reference date (`2025-04-01`) so runs are reproducible.

## Product recommendation model (Notebook 02)

1. Load `data/modeling_dataset.csv` (falls back to deduped `merged_dataset.csv` if missing).
2. Drop leakage columns that share a TX source with the target: `total_spent`, `average_spent`, `purchase_count`, `days_since_last_purchase`, `fraud_rate`, `last_purchase`.
3. Build a `ColumnTransformer` + `RandomForestClassifier` pipeline (`class_weight="balanced"`, `n_estimators=200`).
4. Stratified holdout split (80/20) with an assertion that customer IDs do not overlap train/test.
5. Fit on train; evaluate on holdout + 2-fold stratified CV.

Metrics (`reports/evaluation_report.md`): Accuracy, Precision/Recall/F1 (macro + weighted), multiclass ROC-AUC (OvR), log loss, confusion matrix, feature importances. Accuracy is modest (~12%) — small N (84 customers), 12 classes, social-only features by design.

## Facial recognition model (Notebook 03) & voiceprint verification model (Notebook 04)

Both models answer "which known person is this?" (identification), not open-set verification, and share the same design:

- **Group-aware split**: augmented variants of the same source photo/recording are near-duplicates, so `StratifiedGroupKFold` keeps every variant of one photo/recording on the same side of a split — a plain row-level split would leak a transformed copy of a training sample into the test set.
- **Model**: `StandardScaler` + `LogisticRegression` — more appropriate than a large Random Forest for this feature scale and sample size.
- **Unknown/unauthorized rejection** uses two checks together, both required to accept a match:
  1. **Confidence threshold** — max class probability must be ≥ `unknown_threshold` (0.6).
  2. **Distance-to-centroid novelty guard** — the sample must also be within `distance_margin` (1.5×) of the largest observed intra-class distance from its predicted class's centroid, in scaled feature space. This was added after empirically finding that confidence alone is unreliable: `LogisticRegression` extrapolates *confidently* far outside its training distribution (random noise audio scored 99% confidence for a known person). Both thresholds are persisted in `models/*_config.json` for reuse by `cli_app.py`.

Reports: `reports/face_recognition_report.md`, `reports/voice_verification_report.md` (Accuracy, F1, log loss, CV scores, confusion matrices). Both currently show near-perfect scores — expected given only 2 known people and a handful of source photos/recordings each; metrics are directional, not production-grade.

## Run the CLI demo (Task 6 — `cli_app.py`)

```bash
python cli_app.py demo                            # canned authorized + unauthorized scenarios
python cli_app.py transaction --face F --voice V  # a specific face/voice pair
```

`demo` runs four scenarios: an authorized transaction (face + voice match, product model called), an unauthorized face (synthetic no-match image, denied at step 1), an unauthorized voice (synthetic no-match audio, denied at step 2), and an identity mismatch (valid face + a *different* known person's voice, denied because voice doesn't confirm the face's identity).

`PERSON_TO_CUSTOMER` in `cli_app.py` is an explicit, documented placeholder mapping biometric identities (`person1`/`person2`) to tabular `customer_id`s (`a178`/`a190`) — no real link between the two data sources exists yet.

## Run the Streamlit data viewer

```bash
streamlit run app.py
```

The app uses `fraud_rate` and social scores from `merged_dataset.csv` for a heuristic auth decision (separate from, and simpler than, the trained face/voice models used by `cli_app.py`). Customer filters are case-insensitive so raw TX/profile IDs still match lowercased merged IDs.

## Data overview

| File | Description |
|------|--------------|
| `customer_social_profiles.csv` | Social identity and engagement attributes |
| `customer_transactions.csv` | Transaction history with fraud flags |
| `merged_dataset.csv` | Platform-level joined view (auth / inspection) |
| `modeling_dataset.csv` | Customer-level frame for recommendation training |
| `image_features.csv` | Face embeddings per person/emotion/augmentation |
| `audio_features.csv` | MFCC/spectral features per person/phrase/augmentation |

## Modeling notes

- **Merge IDs:** always strip + lowercase before joins.
- **Target:** `product_category` = mode of transaction `category`.
- **Leakage control (product model):** RFM / `fraud_rate` stay in the CSVs for authentication, but are excluded from the recommendation model because they share a source with the target; training uses one row per customer + `StratifiedKFold(n_splits=2)`.
- **Leakage control (face/voice models):** splits are grouped by source photo/recording (see above), not by row, so augmented near-duplicates never span train/test.
- **Imbalance:** `class_weight="balanced"` on all three classifiers.
- **Platform vs customer grain:** `merged_dataset.csv` is intentionally platform-level; do not train on it without aggregating/deduping.

## Limitations

- Recommendation accuracy is modest (small N, many classes, social-only features by design).
- Facial/voice models cover only `person1`/`person2` — `images/person3/` and `audio/person3/` contain a third member's raw samples but aren't yet run through `process_images.py`/`process_audio.py` (their `PERSONS` lists still only include `person1`/`person2`).
- `PERSON_TO_CUSTOMER` in `cli_app.py` is a placeholder identity-to-customer mapping, not a real linkage.
- Face/voice models are **identification**, not open-set verification — unknown-person rejection relies on the confidence + distance-novelty checks described above rather than being learned from negative examples.
- All three model reports are based on very small sample sizes (single/double digits per class) — metrics are directional, not production-grade.
