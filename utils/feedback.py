from typing import Dict, List, Tuple
import numpy as np

def _arr(vals, default=0.0):
    a = np.asarray(vals, dtype=float)
    return a if a.size else np.array([default], dtype=float)

def bicep_feedback(window: List[Dict[str, float]]) -> List[Tuple[str, str]]:
    """
    Returns list of (message, severity) for bicep curls.
    Uses last ~2s window of angles.
    """
    if not window:
        return []
    elL = _arr([f.get("angle_elbow_L", np.nan) for f in window])
    elR = _arr([f.get("angle_elbow_R", np.nan) for f in window])
    shL = _arr([f.get("angle_shoulder_L", np.nan) for f in window])
    shR = _arr([f.get("angle_shoulder_R", np.nan) for f in window])

    el = np.nanmin(np.vstack([elL, elR]), axis=0)  # more flexed elbow per frame
    sh = np.nanmax(np.vstack([shL, shR]), axis=0)  # larger shoulder swing proxy

    msgs = []
    # Full extension sometimes? (max elbow angle)
    if np.nanmax(el) < 145:
        msgs.append(("Fully extend at the bottom", "info"))
    # Full squeeze at top?
    if np.nanmin(el) > 65:
        msgs.append(("Squeeze more at the top", "info"))
    # Shoulder swing proxy: shoulder variability high while elbow moves
    if np.nanstd(sh) > 12 and (np.nanmax(el) - np.nanmin(el)) > 25:
        msgs.append(("Reduce shoulder swing; keep elbows tucked", "warn"))
    return msgs[:2]

def shoulder_press_feedback(window: List[Dict[str, float]]) -> List[Tuple[str, str]]:
    if not window:
        return []
    elL = _arr([f.get("angle_elbow_L", np.nan) for f in window])
    elR = _arr([f.get("angle_elbow_R", np.nan) for f in window])
    hiL = _arr([f.get("angle_hip_L", np.nan) for f in window])
    hiR = _arr([f.get("angle_hip_R", np.nan) for f in window])

    el = np.nanmin(np.vstack([elL, elR]), axis=0)
    hip = np.nanmin(np.vstack([hiL, hiR]), axis=0)

    msgs = []
    if np.nanmin(el) > 28:
        msgs.append(("Lock out elbows at the top", "warn"))
    if np.nanmax(el) < 90:
        msgs.append(("Lower to at least shoulder level", "info"))
    # Back arch heuristic: large hip motion compared to elbow
    if (np.nanmax(hip) - np.nanmin(hip)) > 25 and (np.nanmax(el) - np.nanmin(el)) < 35:
        msgs.append(("Avoid back arch; brace core", "info"))
    return msgs[:2]

def squat_feedback(window: List[Dict[str, float]]) -> List[Tuple[str, str]]:
    if not window:
        return []
    knL = _arr([f.get("angle_knee_L", np.nan) for f in window])
    knR = _arr([f.get("angle_knee_R", np.nan) for f in window])
    hiL = _arr([f.get("angle_hip_L", np.nan) for f in window])
    hiR = _arr([f.get("angle_hip_R", np.nan) for f in window])

    knee = np.nanmin(np.vstack([knL, knR]), axis=0)
    hip  = np.nanmin(np.vstack([hiL, hiR]), axis=0)

    msgs = []
    if (np.nanmin(knee) > 95) and (np.nanmin(hip) > 80):
        msgs.append(("Go deeper for full range", "warn"))
    if np.nanmax(knee) < 160:
        msgs.append(("Stand fully at the top", "info"))
    # Knees caving: large L/R difference at bottom
    if np.nanmax(np.abs(knL - knR)) > 20:
        msgs.append(("Keep knees tracking over toes", "info"))
    return msgs[:2]

def feedback_for(exercise: str, window: List[Dict[str, float]]) -> List[Tuple[str, str]]:
    if exercise == "bicep_curl":
        return bicep_feedback(window)
    if exercise == "shoulder_press":
        return shoulder_press_feedback(window)
    if exercise == "squats":
        return squat_feedback(window)
    return []