import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="data/features/features.csv")
    ap.add_argument("--out", type=str, default="models/model.joblib")
    ap.add_argument("--meta", type=str, default="models/model_meta.json")
    ap.add_argument("--trees", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    # Feature columns = everything except metadata
    meta_cols = {"video","label","start","end"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feature_cols].values
    y = df["label"].values
    groups = df["video"].values  # for group-wise CV by video

    clf = RandomForestClassifier(
        n_estimators=args.trees,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Grouped CV (by video) — still train on all later per your request
    if len(set(groups)) >= 2:
        gkf = GroupKFold(n_splits=min(5, len(set(groups))))
        scores = cross_val_score(clf, X, y, cv=gkf, groups=groups, scoring="accuracy", n_jobs=-1)
        print(f"[CV] GroupKFold accuracy mean={scores.mean():.3f} ± {scores.std():.3f} over {len(scores)} folds")
    else:
        print("[CV] Not enough distinct videos for grouped CV; skipping.")

    # Fit on all data
    clf.fit(X, y)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.out)

    meta = {
        "feature_names": feature_cols,
        "classes_": list(clf.classes_),
        "window": int(df["end"].iloc[0] - df["start"].iloc[0]) if "start" in df and "end" in df else None,
        "stride": None,  # informative only
    }
    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
    with open(args.meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] saved model -> {args.out}")
    print(f"[OK] saved meta  -> {args.meta}")

if __name__ == "__main__":
    main()