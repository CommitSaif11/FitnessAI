import argparse
import csv
import math
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

POSE_LM = mp_pose.PoseLandmark

# Angle at point b formed by vectors ba and bc (in degrees)
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
    # Using normalized (x,y) => angle is scale-invariant
    L = lambda n: get_xy(landmarks, n)
    # Key joints (left/right)
    shL, elL, wrL = L(POSE_LM.LEFT_SHOULDER), L(POSE_LM.LEFT_ELBOW), L(POSE_LM.LEFT_WRIST)
    shR, elR, wrR = L(POSE_LM.RIGHT_SHOULDER), L(POSE_LM.RIGHT_ELBOW), L(POSE_LM.RIGHT_WRIST)
    hiL, knL, anL = L(POSE_LM.LEFT_HIP), L(POSE_LM.LEFT_KNEE), L(POSE_LM.LEFT_ANKLE)
    hiR, knR, anR = L(POSE_LM.RIGHT_HIP), L(POSE_LM.RIGHT_KNEE), L(POSE_LM.RIGHT_ANKLE)

    # Elbow flexion
    ang_elbow_L = angle_deg(shL, elL, wrL)
    ang_elbow_R = angle_deg(shR, elR, wrR)
    # Shoulder (upper arm vs torso)
    # angle at shoulder between vectors (hip->shoulder) and (shoulder->elbow)
    ang_shoulder_L = angle_deg(hiL, shL, elL)
    ang_shoulder_R = angle_deg(hiR, shR, elR)
    # Hip flexion (torso vs thigh)
    ang_hip_L = angle_deg(shL, hiL, knL)
    ang_hip_R = angle_deg(shR, hiR, knR)
    # Knee flexion
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

def infer_label_from_filename(p: Path) -> str:
    name = p.stem.lower()
    if "bicep" in name:
        return "bicep_curl"
    if "shoulder" in name:
        return "shoulder_press"
    if "squat" in name:
        return "squats"
    return "unknown"

def process_video(video_path: Path, out_csv: Path, sample_every: int = 2):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open {video_path}")
        return

    label = infer_label_from_filename(video_path)

    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose, \
         open(out_csv, "w", newline="") as f:

        writer = csv.writer(f)
        header = ["video", "label", "frame_idx",
                  "angle_elbow_L","angle_elbow_R",
                  "angle_shoulder_L","angle_shoulder_R",
                  "angle_hip_L","angle_hip_R",
                  "angle_knee_L","angle_knee_R"]
        writer.writerow(header)

        frame_idx = 0
        kept = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if sample_every > 1 and (frame_idx % sample_every != 0):
                frame_idx += 1
                continue

            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = pose.process(image_rgb)

            if res.pose_landmarks:
                try:
                    angles = compute_core_angles(res.pose_landmarks.landmark)
                    row = [video_path.name, label, frame_idx] + [angles[k] for k in sorted(angles.keys())]
                    # The order above must match header after the first 3 columns
                    writer.writerow(row)
                    kept += 1
                except Exception as e:
                    # If any angle computation fails, skip the frame
                    pass

            frame_idx += 1

    cap.release()
    print(f"[OK] {video_path.name}: wrote {kept} frames -> {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="data/raw", help="Directory with input .mp4 files")
    ap.add_argument("--out", type=str, default="data/landmarks", help="Output directory for per-video CSV")
    ap.add_argument("--fps_reduce", type=int, default=2, help="Keep 1 every N frames (2 = every other frame)")
    ap.add_argument("--sample_every", type=int, default=None, help="Alias of fps_reduce (override if set)")
    args = ap.parse_args()

    in_dir = Path(args.dir)
    out_dir = Path(args.out)
    sample_every = args.sample_every if args.sample_every else max(1, args.fps_reduce)

    vids = sorted(list(in_dir.glob("*.mp4")))
    if not vids:
        print(f"[WARN] No .mp4 in {in_dir}")
        return

    for v in vids:
        out_csv = out_dir / f"{v.stem}.csv"
        process_video(v, out_csv, sample_every=sample_every)

if __name__ == "__main__":
    main()