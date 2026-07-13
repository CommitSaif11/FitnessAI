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
  raw/, landmarks/, landmarks_v2/, features/   # regenerable, NOT tracked in git (see .gitignore)

models/
  xgb_.../   # trained model artifacts (model, label encoder, feature names)

train_models.ipynb   # notebook version of model training/exploration
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed (`pip install uv`, or see uv's install docs). uv manages the
  Python version for you — **you do NOT need Python 3.8 installed system-wide.**
- This project is pinned to **Python 3.8** because of MediaPipe/protobuf version constraints.

## Setup (using uv)

From the project root (`D:\python fitness ai trial`):

```bash
# 1. Install a managed Python 3.8 (uv downloads it; no system install needed)
uv python install 3.8

# 2. Create the virtual environment using that Python version
uv venv .venv --python 3.8

# 3. Install all dependencies into .venv
uv pip install -r requirements.txt --python .venv
```

To run any script or the app, either activate the venv first:

```bash
# Git Bash / WSL
source .venv/Scripts/activate

# PowerShell
.venv\Scripts\Activate.ps1
```

...or prefix commands with `uv run --python .venv` (no activation needed), e.g.:

```bash
uv run --python .venv streamlit run demo/fitnessai_streamlit.py
```

If you previously had a manually-installed Python 3.8 on your machine solely for this project, you can
now uninstall it — everything runs through uv's managed Python + `.venv` instead.

## Environment variables (Groq LLM key)

Live and end-of-session AI coaching feedback uses the [Groq API](https://console.groq.com/) (free tier,
`llama-3.1-8b-instant` model).

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Put your key in `.env`:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

`.env` is git-ignored — never commit real API keys. If a key is ever pasted into a chat, shared publicly,
or committed by accident, rotate it immediately in the Groq console.

### What the Groq integration does

In the demo sidebar, "Enable Groq AI coaching (live + summary)" is on by default whenever `GROQ_API_KEY`
is set (a warning shows and it auto-disables if the key is missing, so the app still runs fine without it
using only the rule-based feedback strip). When enabled:

- **Live coaching**: every ~5 seconds during a session, recent rep count and form-issue tallies for the
  active exercise are sent to Groq, which returns a one-sentence coaching cue shown in a purple strip
  below the rule-based feedback. If the call fails or is slow, it degrades gracefully (rule-based feedback
  keeps working regardless).
- **Session summary**: when you press Stop, accumulated rep counts and form-issue tallies across all
  three exercises are sent to Groq for a short natural-language summary (what went well, what to improve),
  shown at the bottom of the page.

## Regenerating data (optional)

Raw videos and extracted landmarks are not committed to git (large, regenerable). If you have your own
video dataset under `data/raw/`:

```bash
uv run --python .venv python scripts/extract_landmarks_from_videos.py
uv run --python .venv python scripts/build_features.py
```

## Training

```bash
uv run --python .venv python scripts/train_models.py --features data/features_v2/train.csv --out models/model.joblib
```

## Running the demo

```bash
uv run --python .venv streamlit run demo/fitnessai_streamlit.py
```

Point the sidebar "Model file" field at a trained model directory under `models/` (e.g.
`models/xgb_20251104_082114/xgb_model.joblib`), choose Webcam or Upload video mode, and press Start.

## Roadmap

- [ ] Deep learning (LSTM) sequence model as an alternative to the window-stats + XGBoost approach
- [x] Live + end-of-session LLM coaching feedback (Groq)
- [ ] Browser-based webcam capture (streamlit-webrtc) for a cloud-deployed live demo
