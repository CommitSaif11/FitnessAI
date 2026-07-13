#!/usr/bin/env python3
# Minimal, stable demo: clear feedback + rep counters at the top
# - Works on Python 3.8
# - Always-visible feedback strip (HTML, no flicker)
# - Always-visible rep counters (HTML, no duplicates)
# - Simple built-in rep counters with easy threshold sliders
# - Counts in both Webcam and Upload modes (so it's obvious in the demo)
# - Matches the XGBoost notebook feature schema

import os
import time
from collections import deque, Counter
from pathlib import Path
import json

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import joblib
import mediapipe as mp
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---------------- Groq AI coaching ---------------- #

GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_LIVE_INTERVAL_SEC = 5.0  # min seconds between live coaching calls (rate limit + latency control)

@st.cache_resource
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

def groq_live_tip(client, exercise: str, rep_count: int, issue_counts: dict) -> str:
    issues_text = ", ".join(f"{k} ({v}x)" for k, v in issue_counts.items()) or "none noted"
    prompt = (
        f"You are a concise, encouraging fitness coach watching someone do {exercise.replace('_', ' ')} live. "
        f"They've completed {rep_count} reps so far. Recent form issues detected: {issues_text}. "
        "In ONE short sentence (max 15 words), give a single actionable coaching cue or encouragement. "
        "No preamble, no markdown, just the sentence."
    )
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=40,
    )
    return resp.choices[0].message.content.strip()

def groq_session_summary(client, session_stats: dict) -> str:
    lines = []
    for ex, stats in session_stats.items():
        if stats["reps"] == 0 and not stats["issues"]:
            continue
        issues_text = ", ".join(f"{k} ({v}x)" for k, v in stats["issues"].items()) or "none noted"
        lines.append(f"{ex.replace('_', ' ')}: {stats['reps']} reps, form issues: {issues_text}")
    if not lines:
        return "No exercise activity recorded this session."
    prompt = (
        "You are a fitness coach reviewing a just-finished workout session. Here is the data:\n"
        + "\n".join(lines)
        + "\n\nWrite a short (3-4 sentence) friendly summary: what went well, the main thing to "
        "improve next time, and one word of encouragement. No markdown, plain text."
    )
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()

# ---------------- Load model artifacts ---------------- #

@st.cache_resource
def load_xgb_artifacts(model_path: str):
    model_path = Path(model_path)
    model = joblib.load(model_path)
    le_path = model_path.with_name("xgb_label_encoder.joblib")
    feat_path = model_path.with_name("xgb_feature_names.json")
    if not le_path.exists() or not feat_path.exists():
        raise FileNotFoundError(f"Expected {le_path.name} and {feat_path.name} next to {model_path.name}")
    le = joblib.load(le_path)
    with open(feat_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    classes_ = getattr(le, "classes_", None)
    if classes_ is None:
        raise ValueError("Label encoder missing classes_.")
    window_len = 30  # 2.0 s @ 15 fps
    return model, feature_names, classes_, window_len

# ---------------- Pose + features (match the notebook) ---------------- #

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

SIGNAL_COLS = [
    "elbow_L","elbow_R","shoulder_L","shoulder_R",
    "hip_L","hip_R","knee_L","knee_R","wrist_y_rel_L","wrist_y_rel_R","hip_y"
]

def angle_deg(a, b, c):
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2 + 1e-12), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def get_xy(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x, lm.y], dtype=float)

def normalize_frame(landmarks):
    LS = get_xy(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    RS = get_xy(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    LH = get_xy(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value)
    RH = get_xy(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value)
    center = (LH + RH) / 2.0
    scale1 = (np.linalg.norm(LS - LH) + np.linalg.norm(RS - RH)) / 2.0
    scale2 = np.linalg.norm(LS - RS)
    scale = scale1 if scale1 > 1e-6 else (scale2 if scale2 > 1e-6 else 1.0)
    return center, scale

def compute_core_signals(landmarks):
    center, scale = normalize_frame(landmarks)
    def nxy(idx): return (get_xy(landmarks, idx) - center) / scale

    LS = nxy(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    RS = nxy(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    LE = nxy(mp_pose.PoseLandmark.LEFT_ELBOW.value)
    RE = nxy(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
    LW = nxy(mp_pose.PoseLandmark.LEFT_WRIST.value)
    RW = nxy(mp_pose.PoseLandmark.RIGHT_WRIST.value)
    LH = nxy(mp_pose.PoseLandmark.LEFT_HIP.value)
    RH = nxy(mp_pose.PoseLandmark.RIGHT_HIP.value)
    LK = nxy(mp_pose.PoseLandmark.LEFT_KNEE.value)
    RK = nxy(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    LA = nxy(mp_pose.PoseLandmark.LEFT_ANKLE.value)
    RA = nxy(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

    elbow_L = angle_deg(LS, LE, LW)
    elbow_R = angle_deg(RS, RE, RW)
    shoulder_L = angle_deg(LH, LS, LE)
    shoulder_R = angle_deg(RH, RS, RE)
    hip_L = angle_deg(LS, LH, LK)
    hip_R = angle_deg(RS, RH, RK)
    knee_L = angle_deg(LH, LK, LA)
    knee_R = angle_deg(RH, RK, RA)

    wrist_y_rel_L = LW[1] - LS[1]
    wrist_y_rel_R = RW[1] - RS[1]
    hip_y = (LH[1] + RH[1]) / 2.0

    return {
        "elbow_L": elbow_L, "elbow_R": elbow_R,
        "shoulder_L": shoulder_L, "shoulder_R": shoulder_R,
        "hip_L": hip_L, "hip_R": hip_R,
        "knee_L": knee_L, "knee_R": knee_R,
        "wrist_y_rel_L": wrist_y_rel_L,
        "wrist_y_rel_R": wrist_y_rel_R,
        "hip_y": hip_y,
    }

def window_stats(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, ptp=np.nan, d_mean=np.nan, d_max=np.nan)
    d = np.abs(np.diff(arr))
    return dict(
        mean=np.nanmean(arr),
        std=np.nanstd(arr),
        min=np.nanmin(arr),
        max=np.nanmax(arr),
        ptp=(np.nanmax(arr) - np.nanmin(arr)),
        d_mean=(np.nanmean(d) if d.size else 0.0),
        d_max=(np.nanmax(d) if d.size else 0.0),
    )

def build_feature_vector(window_signals, feature_names):
    df = pd.DataFrame(window_signals)
    feats = {}
    for col in SIGNAL_COLS:
        st_ = window_stats(df[col].values)
        for k, v in st_.items():
            feats[f"{col}_{k}"] = float(v)
    upper = np.nansum(np.abs(np.diff(df[["elbow_L","elbow_R","shoulder_L","shoulder_R"]].values, axis=0)))
    lower = np.nansum(np.abs(np.diff(df[["hip_L","hip_R","knee_L","knee_R"]].values, axis=0)))
    feats["motion_ratio_upper_lower"] = float(upper / max(lower, 1e-6))
    x = np.array([feats.get(k, 0.0) for k in feature_names], dtype=np.float32).reshape(1, -1)
    return x

# ---------------- Minimal feedback rules (with movement guard) ---------------- #

def simple_feedback(exercise, win):
    # If almost no movement in the recent window, show neutral prompt
    if not win:
        return "Move naturally — collecting data...", "info"
    arr = lambda k: np.array([float(d.get(k, np.nan)) for d in win], dtype=float)
    ex = (exercise or "").lower()
    if ex == "biceps_curl":
        elbows = np.nanmean(np.vstack([arr("elbow_L"), arr("elbow_R")]), axis=0)
        if elbows.size == 0 or (np.nanmax(elbows) - np.nanmin(elbows)) < 12:
            return "Start curling — bend and extend the elbows.", "info"
        msg = []
        if np.nanmax(elbows) < 140: msg.append("Extend fully at bottom")
        if np.nanmin(elbows) > 75:  msg.append("Squeeze higher at top")
        return (" | ".join(msg) if msg else "Nice curls — steady pace"), ("warn" if msg else "info")
    if ex == "shoulder_press":
        eL, eR = arr("elbow_L"), arr("elbow_R")
        elbows = np.nanmean(np.vstack([eL, eR]), axis=0)
        if elbows.size == 0 or (np.nanmax(elbows) - np.nanmin(elbows)) < 12:
            return "Start pressing — down then press overhead.", "info"
        wrists_up = (arr("wrist_y_rel_L") < -0.02).mean() + (arr("wrist_y_rel_R") < -0.02).mean()
        msg = []
        if np.nanmin(elbows) > 45: msg.append("Lock out overhead")
        if wrists_up < 0.8: msg.append("Drive hands above shoulders")
        return (" | ".join(msg) if msg else "Strong presses — smooth"), ("warn" if msg else "info")
    if ex in ("squat", "squats"):
        kL, kR = arr("knee_L"), arr("knee_R")
        knees = np.nanmean(np.vstack([kL, kR]), axis=0) if kL.size and kR.size else (kL if kL.size else kR)
        if knees.size == 0 or (np.nanmax(knees) - np.nanmin(knees)) < 12:
            return "Start squatting — sit down then stand up.", "info"
        depth_ok = np.nanmin(knees) <= 95
        return ("Go deeper (hip below knee)" if not depth_ok else "Good depth — stand tall"), ("warn" if not depth_ok else "info")
    return "Move naturally — collecting data...", "info"

# ---------------- Simple rep counters with adjustable thresholds ---------------- #

class SimpleCounter:
    def __init__(self, top_low, bottom_high, invert=False):
        # invert=True means we use value = 180 - angle (for squats)
        self.top_low = float(top_low)
        self.bottom_high = float(bottom_high)
        self.invert = invert
        self.state = "down"
        self.count = 0

    def set_thresholds(self, top_low, bottom_high):
        self.top_low = float(top_low)
        self.bottom_high = float(bottom_high)

    def update(self, value):
        v = float(value)
        if self.invert:
            v = 180.0 - v
        # very small values may be NaN or unrealistic; ignore
        if not np.isfinite(v):
            return
        # Top -> Down cycle counts one rep
        if v <= self.top_low and self.state != "up":
            self.state = "up"
        elif v >= self.bottom_high and self.state == "up":
            self.count += 1
            self.state = "down"

# ---------------- UI ---------------- #

st.set_page_config(page_title="Fitness AI — Minimal Demo", layout="wide")
st.title("Fitness AI — Minimal Demo (Feedback + Reps)")

def find_default_model() -> str:
    candidates = sorted(Path("models").glob("*/xgb_model.joblib"))
    return str(candidates[-1]) if candidates else "models/xgb_YYYYmmdd_HHMMSS/xgb_model.joblib"

# Sidebar controls
model_path = st.sidebar.text_input("Model file", value=find_default_model())
mode = st.sidebar.radio("Mode", ["Webcam", "Upload video"], index=0)
exercise_mode = st.sidebar.radio("Exercise mode", ["Auto (from model)", "Force"], index=1)
forced_ex = st.sidebar.selectbox("Force exercise", ["biceps_curl", "shoulder_press", "squats"], index=1)  # default press per your screenshot
min_conf = st.sidebar.slider("Min confidence (Auto)", 0.0, 1.0, 0.55, 0.05)
frame_stride = st.sidebar.slider("Process every Nth frame", 1, 4, 2)
pose_complexity = st.sidebar.selectbox("Pose model complexity", [0, 1], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("Rep thresholds (tune if reps don't count):")
curl_top = st.sidebar.slider("Curl top elbow ≤", 40, 100, 70)
curl_bottom = st.sidebar.slider("Curl bottom elbow ≥", 120, 180, 140)
press_top = st.sidebar.slider("Press lockout elbow ≤", 30, 80, 55)
press_bottom = st.sidebar.slider("Press bottom elbow ≥", 70, 140, 95)
sq_stand = st.sidebar.slider("Squat stand knee ≥", 150, 180, 160)
sq_depth = st.sidebar.slider("Squat depth knee ≤", 60, 110, 100)

groq_client = get_groq_client()
st.sidebar.markdown("---")
use_groq = st.sidebar.checkbox("Enable Groq AI coaching (live + summary)", value=groq_client is not None)
if use_groq and groq_client is None:
    st.sidebar.warning("GROQ_API_KEY not set in .env — AI coaching disabled.")
    use_groq = False

start = st.sidebar.button("Start")
stop = st.sidebar.button("Stop")

# Top area: feedback strip + AI coach + reps row + video + confidences
feedback_slot = st.empty()
ai_slot = st.empty()
reps_slot = st.empty()
frame_slot = st.empty()
conf_slot = st.empty()
summary_slot = st.empty()

def fresh_counters():
    return {
        "bicep_curl": SimpleCounter(top_low=curl_top, bottom_high=curl_bottom, invert=False),
        "shoulder_press": SimpleCounter(top_low=press_top, bottom_high=press_bottom, invert=False),
        "squats": SimpleCounter(top_low=(180 - sq_stand), bottom_high=(180 - sq_depth), invert=True),
    }

def fresh_session_stats():
    return {ex: {"reps": 0, "issues": Counter()} for ex in ("bicep_curl", "shoulder_press", "squats")}

if "running" not in st.session_state:
    st.session_state.running = False
if "counters" not in st.session_state:
    st.session_state.counters = fresh_counters()
if "session_stats" not in st.session_state:
    st.session_state.session_stats = fresh_session_stats()
if "last_groq_call" not in st.session_state:
    st.session_state.last_groq_call = 0.0
if "needs_summary" not in st.session_state:
    st.session_state.needs_summary = False

if start:
    st.session_state.running = True
    st.session_state.needs_summary = False
    # reset counters + session stats for a clean demo
    st.session_state.counters = fresh_counters()
    st.session_state.session_stats = fresh_session_stats()
if stop:
    st.session_state.running = False
    st.session_state.needs_summary = True

# Load model once
try:
    model, feature_names, classes_, WINDOW = load_xgb_artifacts(model_path)
except Exception as e:
    st.error(f"Load model/meta failed: {e}")
    st.stop()

def set_feedback(text, sev):
    # Colored strip using HTML
    colors = {"crit": "#C62828", "warn": "#EF6C00", "info": "#1565C0", "ok": "#2E7D32"}
    bg = colors.get(sev, "#1565C0")
    html = f"""
    <div style="background:{bg}; color:white; padding:10px 14px; border-radius:6px; font-size:18px; font-weight:600;">
      {text}
    </div>
    """
    feedback_slot.markdown(html, unsafe_allow_html=True)

def show_reps(counts):
    html = f"""
    <div style="display:flex; gap:24px; margin:8px 0 12px 0;">
      <div style="background:#263238; color:#fff; padding:8px 12px; border-radius:6px; font-weight:700;">Bicep curl: {counts.get('bicep_curl', 0)}</div>
      <div style="background:#004d40; color:#fff; padding:8px 12px; border-radius:6px; font-weight:700;">Shoulder press: {counts.get('shoulder_press', 0)}</div>
      <div style="background:#1b5e20; color:#fff; padding:8px 12px; border-radius:6px; font-weight:700;">Squats: {counts.get('squats', 0)}</div>
    </div>
    """
    reps_slot.markdown(html, unsafe_allow_html=True)

def run_capture(capture):
    votes = deque(maxlen=7)
    signals_buf = deque(maxlen=WINDOW)
    feedback_win = deque(maxlen=30)  # ~1s buffer only for feedback
    counters = st.session_state.counters

    # Update thresholds live from sliders (so you can adjust during demo)
    counters["bicep_curl"].set_thresholds(curl_top, curl_bottom)
    counters["shoulder_press"].set_thresholds(press_top, press_bottom)
    counters["squats"].set_thresholds(180 - sq_stand, 180 - sq_depth)

    with mp_pose.Pose(static_image_mode=False, model_complexity=int(pose_complexity),
                      enable_segmentation=False, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        frame_idx = 0
        set_feedback("Move naturally — collecting data...", "info")
        show_reps({k: c.count for k, c in counters.items()})

        while st.session_state.running:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % frame_stride != 0:
                frame_slot.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = pose.process(frame_rgb)

            raw_label = None
            conf = 0.0
            proba = None

            if res.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame_bgr, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )
                sig = compute_core_signals(res.pose_landmarks.landmark)
                signals_buf.append(sig)
                feedback_win.append(sig)

                # Prediction only for Auto mode
                if len(signals_buf) >= WINDOW:
                    x = build_feature_vector(list(signals_buf)[-WINDOW:], feature_names)
                    proba = model.predict_proba(x)[0]
                    raw_label = classes_[int(np.argmax(proba))]
                    conf = float(np.max(proba))
                    if conf >= min_conf:
                        votes.append(raw_label)

                # Choose active exercise
                if exercise_mode == "Force":
                    active = forced_ex
                else:
                    active = votes[-1] if len(votes) else None
                    if active == "squat":
                        active = "squats"

                # Show feedback (top strip)
                fb_label = (active if active else ("biceps_curl" if forced_ex == "biceps_curl" else None))
                if fb_label:
                    # Map UI "squats" to model "squat" for rules
                    rule_label = "squat" if fb_label == "squats" else fb_label
                    msg, sev = simple_feedback(rule_label, list(feedback_win))
                    set_feedback(msg, sev)
                    if fb_label in st.session_state.session_stats and sev == "warn":
                        for issue in msg.split(" | "):
                            st.session_state.session_stats[fb_label]["issues"][issue] += 1

                # Count reps if we know what to count
                if active in ("bicep_curl", "shoulder_press", "squats"):
                    if active in ("bicep_curl", "shoulder_press"):
                        val = min(sig["elbow_L"], sig["elbow_R"])
                    else:
                        val = min(sig["knee_L"], sig["knee_R"])  # squats
                    counters[active].update(val)
                    st.session_state.session_stats[active]["reps"] = counters[active].count

                # Update counters display
                show_reps({k: c.count for k, c in counters.items()})

                # Live Groq coaching tip (rate-limited)
                if use_groq and groq_client is not None and fb_label in st.session_state.session_stats:
                    now = time.time()
                    if now - st.session_state.last_groq_call >= GROQ_LIVE_INTERVAL_SEC:
                        st.session_state.last_groq_call = now
                        try:
                            tip = groq_live_tip(
                                groq_client,
                                fb_label,
                                st.session_state.session_stats[fb_label]["reps"],
                                dict(st.session_state.session_stats[fb_label]["issues"]),
                            )
                            ai_slot.markdown(
                                f"""<div style="background:#4527A0; color:white; padding:8px 14px;
                                border-radius:6px; font-size:15px; font-style:italic;">🤖 {tip}</div>""",
                                unsafe_allow_html=True,
                            )
                        except Exception as e:
                            ai_slot.markdown(
                                f"""<div style="background:#616161; color:white; padding:6px 14px;
                                border-radius:6px; font-size:13px;">AI coach unavailable: {e}</div>""",
                                unsafe_allow_html=True,
                            )

                if proba is not None:
                    conf_df = pd.DataFrame({"class": classes_, "prob": proba}).sort_values("prob", ascending=False)
                    conf_slot.bar_chart(conf_df.set_index("class"))

            # Show frame
            frame_slot.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

# Main
if st.session_state.running:
    if mode == "Webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam")
        else:
            try:
                run_capture(cap)
            finally:
                cap.release()
    else:
        up = st.file_uploader("Upload a video", type=["mp4","mov","mkv"])
        if up is not None and st.session_state.running:
            import tempfile, os
            tmpdir = tempfile.mkdtemp()
            tmp_path = str(Path(tmpdir) / "upload.mp4")
            with open(tmp_path, "wb") as f:
                f.write(up.read())
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                st.error("Cannot open uploaded video")
            else:
                try:
                    run_capture(cap)
                finally:
                    cap.release()
else:
    st.info("Press Start in the sidebar to begin.")
    if st.session_state.needs_summary:
        st.session_state.needs_summary = False
        if use_groq and groq_client is not None:
            with st.spinner("Generating session summary..."):
                try:
                    summary = groq_session_summary(groq_client, st.session_state.session_stats)
                    summary_slot.markdown(
                        f"""<div style="background:#1A237E; color:white; padding:14px 18px;
                        border-radius:8px; font-size:16px; margin-top:12px;">
                        <b>Session summary</b><br/>{summary}</div>""",
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    summary_slot.warning(f"Could not generate AI session summary: {e}")