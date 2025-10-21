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


# Ensure project root is on sys.path so "utils" is importable when running from demo/
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# New utils
from utils.rep_counter import default_counters, BicepCurlCounter
from utils.feedback import feedback_for
from utils.sessions import SessionLogger, FrameLog

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
    # window_angles: list of dicts (one per frame)
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
st.caption("Mediapipe Pose + RandomForest | Webcam + Video Upload + Reps + Feedback")

colL, colR = st.columns([2.5,1])

with colR:
    st.header("Controls")
    model_path = st.text_input("Model file", value="models/model.joblib")
    meta_path = st.text_input("Meta file", value="models/model_meta.json")

    mode = st.radio("Mode", ["Webcam", "Upload video"], index=0)
    exercise_mode = st.radio("Exercise mode", ["Auto (from model)", "Force"], index=0)
    forced_ex = st.selectbox("Force exercise", ["bicep_curl", "shoulder_press", "squats"], index=0)

    st.subheader("Runtime")
    smooth_k = st.slider("Smoothing (votes)", 1, 11, 5, step=2)
    min_conf = st.slider("Min confidence to accept label", 0.0, 1.0, 0.50, step=0.05)
    frame_stride = st.slider("Process every Nth frame", 1, 4, 2)
    pose_complexity = st.selectbox("Pose model complexity", [0,1], index=0)

    st.subheader("Features")
    enable_count = st.checkbox("Enable rep counting", value=True)
    enable_feedback = st.checkbox("Enable feedback tips", value=True)
    enable_logging = st.checkbox("Save session log on Stop", value=True)

    st.subheader("Biceps (elbow angles)")
    curl_side = st.selectbox("Counting side", ["auto","left","right"], index=0)
    curl_top = st.slider("Top (squeeze) ≤", 30, 90, 60)
    curl_bottom = st.slider("Bottom (extension) ≥", 120, 180, 150)

    st.subheader("Shoulder press (elbow)")
    sp_top = st.slider("Top (lockout) ≤", 10, 60, 25)
    sp_bottom = st.slider("Bottom (down) ≥", 70, 140, 95)

    st.subheader("Squats (knee/hip)")
    sq_top = st.slider("Top (stand) ≥", 150, 180, 160)
    sq_bottom = st.slider("Bottom (depth) ≤", 60, 110, 85)

    start_btn = st.button("Start")
    stop_btn = st.button("Stop")
    reset_btn = st.button("Reset counters")

with colL:
    frame_slot = st.empty()
    info_slot = st.empty()
    count_slot = st.empty()
    conf_slot = st.empty()
    tips_slot = st.empty()
    download_slot = st.empty()

if "running" not in st.session_state:
    st.session_state.running = False
if "counters" not in st.session_state:
    st.session_state.counters = None
if "logger" not in st.session_state:
    st.session_state.logger = None

if start_btn:
    st.session_state.running = True
    # Initialize counters with UI thresholds
    st.session_state.counters = default_counters(side_for_curl=curl_side)
    # Update thresholds from UI
    st.session_state.counters["bicep_curl"].th.top_low = curl_top
    st.session_state.counters["bicep_curl"].th.bottom_high = curl_bottom
    st.session_state.counters["shoulder_press"].th.top_low = sp_top
    st.session_state.counters["shoulder_press"].th.bottom_high = sp_bottom
    st.session_state.counters["squats"].th.top_low = sq_top
    st.session_state.counters["squats"].th.bottom_high = sq_bottom

    st.session_state.logger = SessionLogger()
if reset_btn and st.session_state.counters:
    for c in st.session_state.counters.values():
        c.reset()
if stop_btn:
    st.session_state.running = False

# Try to load model
try:
    model, feature_names, classes_, WINDOW = load_model_and_meta(model_path, meta_path)
except Exception as e:
    st.error(f"Load model/meta failed: {e}")
    st.stop()

ANGLE_WINDOW_SEC = 2.0  # for feedback; roughly 2 seconds worth of angles

def predict_stream(capture, run_mode: str):
    votes = deque(maxlen=smooth_k)
    angle_buffer = deque(maxlen=WINDOW)
    feedback_window = deque(maxlen=max(WINDOW, int(ANGLE_WINDOW_SEC*30)))  # ~2s
    fps_t = time.time()
    fps_count = 0
    fps_val = 0.0

    counters = st.session_state.counters
    logger = st.session_state.logger

    with mp_pose.Pose(static_image_mode=False, model_complexity=pose_complexity,
                      enable_segmentation=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        frame_idx = 0
        while st.session_state.running:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % frame_stride != 0:
                # Still show the frame even if skipped for processing
                frame_slot.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = pose.process(frame_rgb)

            raw_label = None
            final_label = None
            conf = 0.0
            angles = None
            tips = []

            if res.pose_landmarks:
                # Draw skeleton for visualization
                mp_draw.draw_landmarks(
                    frame_bgr, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )
                angles = compute_core_angles(res.pose_landmarks.landmark)
                angle_buffer.append(angles)
                feedback_window.append(angles)

                if len(angle_buffer) >= WINDOW:
                    x = stats_feature_vector(list(angle_buffer)[-WINDOW:], feature_names)
                    probs = model.predict_proba(x)[0]
                    raw_idx = int(np.argmax(probs))
                    raw_label = classes_[raw_idx]
                    conf = float(np.max(probs))

                    # Apply min confidence gate
                    if conf >= min_conf:
                        votes.append(raw_label)
                    # Smooth with majority vote
                    if len(votes) > 0:
                        final_label = Counter(votes).most_common(1)[0][0]
                    else:
                        final_label = None

                # Rep counting + feedback
                counts_view = {"bicep_curl": 0, "shoulder_press": 0, "squats": 0}
                active_ex = None
                if exercise_mode.startswith("Force"):
                    active_ex = forced_ex
                else:
                    active_ex = final_label or raw_label

                if enable_count and active_ex in {"bicep_curl","shoulder_press","squats"} and counters:
                    c = counters[active_ex]
                    stt = c.update(angles)
                    # reflect latest totals for display
                    for ex, ctr in counters.items():
                        counts_view[ex] = ctr.get_stats().count

                # Feedback (based on recent window)
                if enable_feedback and active_ex in {"bicep_curl","shoulder_press","squats"}:
                    tips = [f"{msg}" for msg, sev in feedback_for(active_ex, list(feedback_window))]
                    # Show up to 2 tips
                    tips = tips[:2]

                
                # Info UI
                if exercise_mode.startswith("Force"):
                    title_label = f"{forced_ex} (forced)"
                else:
                    title_label = final_label if final_label else (raw_label if raw_label else "—")
                info_slot.markdown(
                    f"### Prediction: **{title_label}**  |  Confidence: **{conf:.2f}**  |  FPS: **{fps_val:.1f}**"
                )

                count_slot.write(pd.DataFrame([{
                    "bicep_curl": counters['bicep_curl'].get_stats().count if counters else 0,
                    "shoulder_press": counters['shoulder_press'].get_stats().count if counters else 0,
                    "squats": counters['squats'].get_stats().count if counters else 0,
                }]).T.rename(columns={0:"reps"}))

                if tips:
                    tips_slot.info(" • " + "  |  ".join(tips))
                else:
                    tips_slot.empty()

                if conf > 0 and final_label is not None and conf_slot:
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

            # Logging (after display)
            if st.session_state.running and enable_logging and logger:
                logger.log(FrameLog(
                    ts=time.time(),
                    mode=run_mode,
                    raw_label=raw_label,
                    final_label=final_label,
                    confidence=conf,
                    angles=angles or {},
                    counts={
                        "bicep_curl": counters['bicep_curl'].get_stats().count if counters else 0,
                        "shoulder_press": counters['shoulder_press'].get_stats().count if counters else 0,
                        "squats": counters['squats'].get_stats().count if counters else 0,
                    },
                    feedback=tips
                ))

def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
        return
    try:
        predict_stream(cap, run_mode="Webcam")
    finally:
        cap.release()

def run_uploaded_video():
    up = st.file_uploader("Upload a video", type=["mp4","mov","mkv"])
    if up is None:
        st.info("Upload a video to start.")
        return
    tmpdir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmpdir, "upload.mp4")
    with open(tmp_path, "wb") as f:
        f.write(up.read())
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("Cannot open uploaded video")
        return
    try:
        predict_stream(cap, run_mode="Upload")
    finally:
        cap.release()

# Main run switch
if st.session_state.running:
    if mode == "Webcam":
        run_webcam()
    else:
        run_uploaded_video()
else:
    st.info("Press Start to begin.")
    download_slot.empty()

# On Stop: allow saving/downloading the session log
if (not st.session_state.running) and st.session_state.get("logger") and enable_logging:
    try:
        df = st.session_state.logger.to_df()
        if not df.empty:
            st.success(f"Session captured: {len(df)} frames")
            st.dataframe(df.tail(5))
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            download_slot.download_button(
                "Download session CSV",
                data=csv_bytes,
                file_name="session_latest.csv",
                mime="text/csv"
            )
            # Also write to disk
            out_dir = Path("data/sessions")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts_name = time.strftime("%Y%m%d_%H%M%S")
            path = out_dir / f"session_{ts_name}.csv"
            df.to_csv(path, index=False)
            st.caption(f"Saved to {path}")
        else:
            st.info("No frames recorded in this session.")
    except Exception as e:
        st.warning(f"Could not finalize session log: {e}")