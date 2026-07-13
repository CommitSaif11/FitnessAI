from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import pandas as pd


@dataclass
class FrameLog:
    ts: float
    mode: str
    raw_label: str | None
    final_label: str | None
    confidence: float
    angles: Dict[str, float]  # signals/angles snapshot
    counts: Dict[str, int]
    feedback: List[str]


class SessionLogger:
    def __init__(self):
        self._rows: List[FrameLog] = []

    def log(self, row: FrameLog):
        self._rows.append(row)

    def to_df(self) -> pd.DataFrame:
        flat: List[Dict[str, Any]] = []
        for r in self._rows:
            base = {
                "ts": r.ts,
                "mode": r.mode,
                "raw_label": r.raw_label,
                "final_label": r.final_label,
                "confidence": r.confidence,
            }
            # flatten angles/signals
            for k, v in (r.angles or {}).items():
                base[f"sig_{k}"] = v
            # flatten counts
            for k, v in (r.counts or {}).items():
                base[f"count_{k}"] = v
            # join feedback
            base["feedback"] = " | ".join(r.feedback or [])
            flat.append(base)
        return pd.DataFrame(flat)