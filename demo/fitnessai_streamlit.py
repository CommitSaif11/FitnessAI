import os
import time
from collections import deque, Counter
import tempfile
from pathlib import Path
import json

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import joblib
import mediapipe as mp

# ---------------- Shared angle logic (must match training) ---------------- #

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

ANGLE_COLS = [
    "angle_elbow_L","angle_elbow_R",
    "angle_shoulder_L","angle_shoulder_R",
    "angle_hip_L","angle_hip_R",
    "angle_knee_L","angle_knee_R",
]

def angle_deg(a, b, c):
    a, b, c = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32), np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    nba = ba / (np.linalg.norm(ba) + 1e-6)
    nbc = bc / (np.linalg.norm(bc) + 1e-6)
    cosang = np.clip(np.dot(nba, nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def get_xy(landmarks, name):
    lm = landmarks[name.value]
    return (lm.x, lm.y)

def compute_core_angles(landmarks):
    POSE_LM = mp_pose.PoseLandmark
    L = lambda n: get_xy(landmarks, n)
    shL, elL, wrL = L(POSE_LM.LEFT_SHOULDER), L(POSE_LM.LEFT_ELBOW), L(POSE_LM.LEFT_WRIST)
    shR, elR, wrR = L(POSE_LM.RIGHT_SHOULDER), L(POSE_LM.RIGHT_ELBOW), L(POSE_LM.RIGHT_WRIST)
    hiL, knL, anL = L(POSE_LM.LEFT_HIP), L(POSE_LM.LEFT_KNEE), L(POSE_LM.LEFT_ANKLE)
    hiR, knR, anR = L(POSE_LM.RIGHT_HIP), L(POSE_LM.RIGHT_KNEE), L(POSE_LM.RIGHT_ANKLE)

    ang_elbow_L = angle_deg(shL, elL, wrL)
    ang_elbow_R = angle_deg(shR, elR, wrR)
    ang_shoulder_L = angle_deg(hiL, shL, elL)
    ang_shoulder_R = angle_deg(hiR, shR, elR)
    ang_hip_L = angle_deg(shL, hiL, knL)
    ang_hip_R = angle_deg(shR, hiR, knR)
    ang_knee_L = angle_deg(hiL, knL, anL)
    ang_knee_R = angle_deg(hiR, knR, anR)

    return {
        "angle_elbow_L": ang_elbow_L,
        "angle_elbow_R": ang_elbow_R,
        "angle_shoulder_L": ang_shoulder_L,
        "angle_shoulder_R": ang_shoulder_R,
        "angle_hip_L": ang_hip_L,
        "angle_hip_R": ang_hip_R,
        "angle_knee_L": ang_knee_L,
        "angle_knee_R": ang_knee_R,
    }

def stats_feature_vector(window_angles, feature_names):
    # window_angles: list of dicts (one per frame),
    # feature_names: list of expected columns as saved in training meta
    # We compute stats per angle to match training: mean,std,min,max,p10,p90,range
    df = pd.DataFrame(window_angles)
    feats = {}
    for col in ANGLE_COLS:
        v = df[col].astype(float).to_numpy()
        feats[f"{col}_mean"] = float(np.nanmean(v))
        feats[f"{col}_std"] = float(np.nanstd(v))
        feats[f"{col}_min"] = float(np.nanmin(v))
        feats[f"{col}_max"] = float(np.nanmax(v))
        feats[f"{col}_p10"] = float(np.nanpercentile(v, 10))
        feats[f"{col}_p90"] = float(np.nanpercentile(v, 90))
        feats[f"{col}_range"] = feats[f"{col}_max"] - feats[f"{col}_min"]
    # Reorder and fill any missing with 0.0 (shouldn't happen if consistent)
    x = np.array([feats.get(k, 0.0) for k in feature_names], dtype=np.float32).reshape(1, -1)
    return x

# ---------------- Load model and meta ---------------- #

@st.cache_resource
def load_model_and_meta(model_path: str, meta_path: str):
    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    classes_ = meta["classes_"]
    window = meta.get("window", 30) or 30
    return model, feature_names, classes_, window

# ---------------- Streamlit UI ---------------- #

st.set_page_config(page_title="Fitness AI Demo", layout="wide")
st.title("Fitness AI — Live Exercise Classifier")
st.caption("Mediapipe Pose + RandomForest | Webcam + Video Upload")

colL, colR = st.columns([2,1])

with colR:
    st.header("Controls")
    model_path = st.text_input("Model file", value="models/model.joblib")
    meta_path = st.text_input("Meta file", value="models/model_meta.json")
    mode = st.radio("Mode", ["Webcam", "Upload video"], index=0)
    smooth_k = st.slider("Smoothing (votes)", 1, 9, 5, step=2)
    conf_bar = st.checkbox("Show per-class confidences", value=True)
    start_btn = st.button("Start")
    stop_btn = st.button("Stop")

with colL:
    frame_slot = st.empty()
    info_slot = st.empty()
    conf_slot = st.empty()

if "running" not in st.session_state:
    st.session_state.running = False
if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

# Try to load model
try:
    model, feature_names, classes_, WINDOW = load_model_and_meta(model_path, meta_path)
except Exception as e:
    st.error(f"Load model/meta failed: {e}")
    st.stop()

def predict_stream(capture):
    votes = deque(maxlen=smooth_k)
    angle_buffer = deque(maxlen=WINDOW)
    fps_t = time.time()
    fps_count = 0
    fps_val = 0.0

    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while st.session_state.running:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = pose.process(frame_rgb)

            if res.pose_landmarks:
                # Draw skeleton
                mp_draw.draw_landmarks(
                    frame_bgr, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )
                angles = compute_core_angles(res.pose_landmarks.landmark)
                angle_buffer.append(angles)

                if len(angle_buffer) >= WINDOW:
                    x = stats_feature_vector(list(angle_buffer)[-WINDOW:], feature_names)
                    probs = model.predict_proba(x)[0]
                    pred_idx = int(np.argmax(probs))
                    pred_label = classes_[pred_idx]
                    votes.append(pred_label)
                    # Smooth by majority vote
                    final = Counter(votes).most_common(1)[0][0]
                    conf = float(np.max(probs))

                    # Display
                    info_slot.markdown(
                        f"### Prediction: **{final}**  |  Confidence: **{conf:.2f}**  |  FPS: **{fps_val:.1f}**"
                    )
                    if conf_bar:
                        conf_df = pd.DataFrame({"class": classes_, "prob": probs}).sort_values("prob", ascending=False)
                        conf_slot.bar_chart(conf_df.set_index("class"))

            # FPS calc
            fps_count += 1
            if (time.time() - fps_t) >= 1.0:
                fps_val = fps_count / (time.time() - fps_t)
                fps_count = 0
                fps_t = time.time()

            # Show frame
            frame_slot.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
        return
    try:
        predict_stream(cap)
    finally:
        cap.release()

def run_uploaded_video(file):
    # Save to temp file for OpenCV
    tmpdir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmpdir, "upload.mp4")
    with open(tmp_path, "wb") as f:
        f.write(file.read())
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("Cannot open uploaded video")
        return
    try:
        predict_stream(cap)
    finally:
        cap.release()

# Main run
if st.session_state.running:
    if mode == "Webcam":
        run_webcam()
    else:
        up = st.file_uploader("Upload a .mp4 file", type=["mp4", "mov", "mkv"])
        if up is not None:
            run_uploaded_video(up)
        else:
            st.info("Upload a video to start.")
else:
    st.info("Press Start to begin.")