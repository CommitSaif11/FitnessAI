# FitnessAI

A pose-based fitness exercise classifier and rep counter. Uses MediaPipe for pose landmark extraction,
a classical ML model (XGBoost / RandomForest) for exercise classification, and a Streamlit app for a
live demo with real-time rep counting and rule-based form feedback.

Currently supports three exercises: **bicep curls**, **shoulder press**, and **squats**.

## How it works

```
video (webcam or upload)
  -> MediaPipe Pose landmarks
  -> joint angle signals (elbow, shoulder, hip, knee)
  -> sliding-window statistical features (mean/std/min/max/percentiles)
  -> XGBoost classifier -> predicted exercise
  -> rule-based state machine -> rep count + form feedback
```

## Project structure

```
scripts/
  extract_landmarks_from_videos.py  # video -> per-frame joint angle CSVs (data/landmarks_v2)
  build_features.py                 # per-frame angles -> windowed statistical features
  normalize_labels.py                # label cleanup helper
  train_models.py                    # trains the RandomForest/XGBoost classifier
  auto_label_from_videos.py          # (WIP) auto-labeling helper

utils/
  rep_counter.py     # per-exercise rep-counting state machines (angle thresholds + hysteresis)
  feedback.py         # rule-based form feedback
  sessions.py          # session bookkeeping helpers

demo/
  fitnessai_streamlit.py   # Streamlit app: webcam/video upload, live prediction, rep counts, feedback

data/
  features_v2/    # windowed training features (tracked in git; small)
  raw/, landmarks/, landmarks_v2/, features/   # regenerable, not tracked in git

models/
  xgb_.../   # trained model artifacts (model, label encoder, feature names)

train_models.ipynb   # notebook version of model training/exploration
```

## Setup

```bash
pip install -r requirements.txt
```

Python 3.8 is required (pinned MediaPipe/protobuf versions depend on it).

## Regenerating data (optional)

Raw videos and extracted landmarks are not committed to git (large, regenerable). If you have your own
video dataset under `data/raw/`:

```bash
python scripts/extract_landmarks_from_videos.py
python scripts/build_features.py
```

## Training

```bash
python scripts/train_models.py --features data/features_v2/train.csv --out models/model.joblib
```

## Running the demo

```bash
streamlit run demo/fitnessai_streamlit.py
```

Point the sidebar "Model file" field at a trained model directory under `models/`, choose Webcam or
Upload video mode, and press Start.

## Roadmap

- [ ] Deep learning (LSTM) sequence model as an alternative to the window-stats + XGBoost approach
- [ ] Live LLM-generated coaching feedback (Groq) replacing/augmenting the rule-based feedback strip
- [ ] Browser-based webcam capture (streamlit-webrtc) for a cloud-deployed live demo
