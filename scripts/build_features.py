import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ANGLE_COLS = [
    "angle_elbow_L","angle_elbow_R",
    "angle_shoulder_L","angle_shoulder_R",
    "angle_hip_L","angle_hip_R",
    "angle_knee_L","angle_knee_R",
]

def window_stats(values: np.ndarray):
    # Return dict of stats for a 1-D array
    v = values.astype(np.float32)
    return {
        "mean": float(np.nanmean(v)),
        "std": float(np.nanstd(v)),
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
        "p10": float(np.nanpercentile(v, 10)),
        "p90": float(np.nanpercentile(v, 90)),
        "range": float(np.nanmax(v) - np.nanmin(v)),
    }

def build_windows(df: pd.DataFrame, win: int, stride: int):
    rows = []
    video = df["video"].iloc[0]
    label = df["label"].iloc[0]
    arr = df[ANGLE_COLS].to_numpy()
    n = len(df)
    for start in range(0, max(0, n - win + 1), stride):
        end = start + win
        if end > n:
            break
        feats = {}
        for i, col in enumerate(ANGLE_COLS):
            stats = window_stats(arr[start:end, i])
            for k, v in stats.items():
                feats[f"{col}_{k}"] = v
        feats["video"] = video
        feats["label"] = label
        feats["start"] = start
        feats["end"] = end
        rows.append(feats)
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks_dir", type=str, default="data/landmarks")
    ap.add_argument("--out", type=str, default="data/features/features.csv")
    ap.add_argument("--win", type=int, default=30)
    ap.add_argument("--stride", type=int, default=15)
    args = ap.parse_args()

    lm_dir = Path(args.landmarks_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csvs = sorted(lm_dir.glob("*.csv"))
    if not csvs:
        print(f"[WARN] no CSVs found in {lm_dir}")
        return

    all_df = []
    for p in csvs:
        df = pd.read_csv(p)
        # keep rows that have all angle columns present
        df = df.dropna(subset=ANGLE_COLS, how="any")
        if len(df) < args.win:
            continue
        wins = build_windows(df, args.win, args.stride)
        all_df.append(wins)

    if not all_df:
        print("[WARN] no windows generated")
        return

    out = pd.concat(all_df, ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote features -> {out_path}, shape={out.shape}")

if __name__ == "__main__":
    main()