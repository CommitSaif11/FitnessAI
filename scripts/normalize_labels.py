#!/usr/bin/env python3
"""
Normalize label names in feature CSVs so the model sees a consistent set:
  - biceps_curl
  - shoulder_press
  - squat

Usage:
  python scripts/normalize_labels.py --dir data/features_v2
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

CANON = {"biceps_curl", "shoulder_press", "squat"}

# Map common variants to canonical labels
NORMALIZE = {
    "bicep_curl": "biceps_curl",  # singular -> plural
    "biceps_curl": "biceps_curl",
    "shoulder_press": "shoulder_press",
    "squat": "squat",
    "squats": "squat",            # plural -> singular
}

def fix_labels(path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "label" not in df.columns:
        print(f"Skip (no label column): {path}")
        return
    # Normalize
    df["label"] = df["label"].map(lambda x: NORMALIZE.get(str(x).strip().lower(), str(x).strip().lower()))
    # Sanity check
    bad = sorted(set(df["label"]) - CANON)
    if bad:
        print(f"Warning: {path} contains unexpected labels: {bad}")
    df.to_csv(path, index=False)
    print(f"Normalized labels in {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/features_v2", type=str, help="Directory containing train/val/test CSVs")
    args = ap.parse_args()
    d = Path(args.dir)
    for split in ["train", "val", "test"]:
        fix_labels(d / f"{split}.csv")

if __name__ == "__main__":
    main()