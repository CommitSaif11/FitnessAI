from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

# Severity levels for color-coding in UI overlays
# - "crit": red banner
# - "warn": orange banner
# - "info": blue banner
# - "ok":   green banner (rarely used for tips)
Severity = str


def _arr(win: List[Dict[str, float]], key: str) -> np.ndarray:
    return np.array([float(d.get(key, np.nan)) for d in win], dtype=float)


def _range(a: np.ndarray) -> float:
    if a.size == 0 or np.all(np.isnan(a)):
        return np.nan
    return float(np.nanmax(a) - np.nanmin(a))


def _nanpct(a: np.ndarray, p: float) -> float:
    if a.size == 0 or np.all(np.isnan(a)):
        return np.nan
    return float(np.nanpercentile(a, p))


def feedback_for(exercise: str, window: List[Dict[str, float]]) -> List[Tuple[str, Severity]]:
    """Return up to a few prioritized feedback messages for the last ~2 seconds of motion."""
    ex = (exercise or "").lower()
    tips: List[Tuple[str, Severity]] = []

    # Common signals
    eL = _arr(window, "elbow_L")
    eR = _arr(window, "elbow_R")
    sL = _arr(window, "shoulder_L")
    sR = _arr(window, "shoulder_R")
    kL = _arr(window, "knee_L")
    kR = _arr(window, "knee_R")
    hip_y = _arr(window, "hip_y")
    wyL = _arr(window, "wrist_y_rel_L")
    wyR = _arr(window, "wrist_y_rel_R")
    # Optional (may not exist; live app can provide torso_pitch)
    torso_pitch = _arr(window, "torso_pitch") if "torso_pitch" in (window[-1] if window else {}) else np.array([])

    if ex == "bicep_curl":
        eL = _arr(window, "elbow_L")
        eR = _arr(window, "elbow_R")
        sL = _arr(window, "shoulder_L")
        sR = _arr(window, "shoulder_R")

        rL = float(np.nanmax(eL) - np.nanmin(eL)) if eL.size else 0.0
        rR = float(np.nanmax(eR) - np.nanmin(eR)) if eR.size else 0.0
        rMax = max(rL, rR)

        if rMax < 20:
            tips.append(("Start curling — bend and extend the elbows.", "info"))
            return tips[:2]

        elbows = np.nanmean(np.vstack([eL, eR]), axis=0) if (eL.size and eR.size) else (eL if eL.size else eR)

        if np.nanmax(elbows) < 140:
            tips.append(("Fully extend at the bottom — open the elbows more.", "warn"))
        if np.nanmin(elbows) > 70:
            tips.append(("Curl higher — squeeze more at the top.", "info"))

        sh_range = max(_range(sL), _range(sR))
        if sh_range > 35:
            tips.append(("Keep elbows by your side — reduce shoulder swing.", "warn"))

        if abs(rL - rR) > 15:
            tips.append(("Match left/right — keep reps symmetrical.", "info"))

        # Tempo using dominant arm
        dom = eL if rL >= rR else eR
        if dom.size > 3:
            d = np.abs(np.diff(dom))
            speed = float(np.nanmean(d))
            if speed > 7.0:
                tips.append(("Control the tempo — slow down.", "info"))
            elif speed < 2.0 and rMax > 25:
                tips.append(("Drive up a bit stronger.", "info"))

        if not tips and (np.nanmax(elbows) > 150 and np.nanmin(elbows) < 55):
            tips.append(("Great control — keep the tempo steady.", "ok"))

        return tips[:2]

    elif ex == "shoulder_press":
        elbows = np.nanmean(np.vstack([eL, eR]), axis=0) if eL.size and eR.size else (eL if eL.size else eR)
        wrists_above = np.nanmean((wyL < -0.02).astype(float) + (wyR < -0.02).astype(float)) / 2.0  # fraction 0..1
        if elbows.size:
            if np.nanmin(elbows) > 40:
                tips.append(("Lock out overhead — straighten the elbows at the top.", "warn"))
            if np.nanmax(elbows) < 90:
                tips.append(("Lower to at least chin level before pressing.", "info"))
        if wrists_above < 0.5:
            tips.append(("Drive hands above the shoulders — finish tall.", "warn"))
        if not tips and np.nanmin(elbows) < 25 and np.nanmax(elbows) > 95:
            tips.append(("Nice presses — smooth rhythm.", "ok"))

    elif ex in ("squat", "squats"):
        knees = np.nanmean(np.vstack([kL, kR]), axis=0) if kL.size and kR.size else (kL if kL.size else kR)
        depth_ok = np.nanmin(knees) <= 90  # below ~parallel proxy
        hip_drop = _range(hip_y)  # more drop => larger range
        if not depth_ok:
            severity = "crit" if np.nanmin(knees) > 100 else "warn"
            tips.append(("Go deeper — aim hip crease below knee.", severity))
        if hip_drop < 0.08:  # in normalized units, ~8% of shoulder/hip scale
            tips.append(("Sit back more — increase hip travel.", "info"))
        # Chest collapse proxy: shoulder angle large at bottom (rough)
        if np.nanmean(np.vstack([sL, sR])) > 70:
            tips.append(("Keep chest up — brace your torso.", "info"))
        # Asymmetry
        if abs(np.nanmean(kL) - np.nanmean(kR)) > 12:
            tips.append(("Balance left/right — keep knees tracking evenly.", "info"))
        if not tips and depth_ok and hip_drop >= 0.1:
            tips.append(("Good depth — keep heels planted.", "ok"))

    else:
        tips.append(("Move naturally — collecting data...", "info"))

    # Prioritize: crit > warn > info > ok, return top 2
    order = {"crit": 3, "warn": 2, "info": 1, "ok": 0}
    tips.sort(key=lambda t: order.get(t[1], 0), reverse=True)
    return tips[:2]