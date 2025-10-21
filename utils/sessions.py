from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime

@dataclass
class FrameLog:
    ts: float
    mode: str               # "Webcam" | "Upload"
    raw_label: Optional[str]
    final_label: Optional[str]
    confidence: float
    angles: Dict[str, float]
    counts: Dict[str, int]
    feedback: List[str]

@dataclass
class SessionLogger:
    rows: List[FrameLog] = field(default_factory=list)

    def log(self, row: FrameLog):
        self.rows.append(row)

    def to_df(self) -> pd.DataFrame:
        # Flatten nested fields for a readable CSV
        records = []
        for r in self.rows:
            base = dict(
                ts=r.ts,
                mode=r.mode,
                raw_label=r.raw_label,
                final_label=r.final_label,
                confidence=r.confidence
            )
            # angles
            for k, v in r.angles.items():
                base[f"ang_{k}"] = v
            # counts
            for k, v in r.counts.items():
                base[f"count_{k}"] = v
            # feedback as single string
            base["feedback"] = " | ".join(r.feedback) if r.feedback else ""
            records.append(base)
        return pd.DataFrame.from_records(records)

    def save_csv(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"session_{ts}.csv"
        df = self.to_df()
        df.to_csv(path, index=False)
        return path