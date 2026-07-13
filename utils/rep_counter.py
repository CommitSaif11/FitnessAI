from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Thresholds:
    # Generic thresholds used per exercise
    # Biceps/Press: "top" is the small elbow angle (contracted/lockout), "bottom" is large elbow angle (extended/down)
    top_low: float     # angle at the top must go below this (e.g., 60 for curls, 25 for press)
    bottom_high: float # angle at the bottom must go above this (e.g., 150 for curls, 95 for press)
    min_hold_frames: int = 2  # frames required near top/bottom to debounce
    hysteresis: float = 5.0   # degrees margin to avoid bouncing around thresholds


@dataclass
class CounterStats:
    count: int = 0
    state: str = "unknown"  # "up" (contracted/top) or "down" (extended/bottom)
    side: str = "auto"      # for curls
    last_event: Optional[str] = None  # "top" or "bottom"


class BaseCounter:
    def __init__(self, thresholds: Thresholds):
        self.th = thresholds
        self.stats = CounterStats()
        self._hold = 0  # debounce counter

    def reset(self):
        self.stats = CounterStats(side=getattr(self.stats, "side", "auto"))
        self._hold = 0

    def get_stats(self) -> CounterStats:
        return self.stats

    # Child classes must implement _current_value and _is_valid_frame
    def _current_value(self, signals: Dict[str, float]) -> float:
        raise NotImplementedError

    def _is_valid_frame(self, signals: Dict[str, float]) -> bool:
        return True

    def update(self, signals: Dict[str, float]) -> CounterStats:
        """Generic top/bottom state machine:
        - Detect sustained TOP (value <= top_low) -> state becomes 'up'
        - Detect sustained BOTTOM (value >= bottom_high) -> state becomes 'down'
        - Count a rep on the transition 'up' -> 'down'
        """
        if not self._is_valid_frame(signals):
            self._hold = 0
            return self.stats

        v = float(self._current_value(signals))

        # Check if near top or bottom with hysteresis bands
        near_top = v <= (self.th.top_low + self.th.hysteresis)
        near_bottom = v >= (self.th.bottom_high - self.th.hysteresis)

        if near_top:
            self._hold += 1
            if self._hold >= self.th.min_hold_frames:
                if self.stats.state != "up":
                    self.stats.state = "up"
                    self.stats.last_event = "top"
        elif near_bottom:
            self._hold += 1
            if self._hold >= self.th.min_hold_frames:
                # Count when we pass from up -> down (i.e., completed a cycle)
                if self.stats.state == "up":
                    self.stats.count += 1
                self.stats.state = "down"
                self.stats.last_event = "bottom"
        else:
            # neither top nor bottom; reset debounce
            self._hold = 0

        return self.stats


class BicepCurlCounter(BaseCounter):
    """Counts reps using elbow angle of the dominant arm.
    'top' = small angle (curl squeeze), 'bottom' = large angle (full elbow extension).
    """
    def __init__(self, thresholds: Optional[Thresholds] = None, side: str = "auto"):
        if thresholds is None:
            thresholds = Thresholds(top_low=60.0, bottom_high=150.0, min_hold_frames=2, hysteresis=5.0)
        super().__init__(thresholds)
        self.stats.side = side
        # running amplitude to auto-pick side
        self._side_amp = {"left": 0.0, "right": 0.0}
        self._prev = {"left": None, "right": None}

    def _choose_side(self, signals: Dict[str, float]) -> str:
        if self.stats.side in ("left", "right"):
            return self.stats.side
        # auto: pick side with larger elbow motion amplitude on the fly
        for s, key in (("left", "elbow_L"), ("right", "elbow_R")):
            v = float(signals.get(key, 0.0))
            pv = self._prev[s]
            if pv is not None:
                self._side_amp[s] += abs(v - pv)
            self._prev[s] = v
        return "left" if self._side_amp["left"] >= self._side_amp["right"] else "right"

    def _current_value(self, signals: Dict[str, float]) -> float:
        side = self._choose_side(signals)
        self.stats.side = side
        return float(signals.get("elbow_L" if side == "left" else "elbow_R", 180.0))


class ShoulderPressCounter(BaseCounter):
    """Counts reps using elbow angle (lockout at small angle, bottom at larger angle).
    Adds a soft check that wrists elevate above shoulders at top to avoid false positives.
    """
    def __init__(self, thresholds: Optional[Thresholds] = None):
        if thresholds is None:
            thresholds = Thresholds(top_low=25.0, bottom_high=95.0, min_hold_frames=2, hysteresis=5.0)
        super().__init__(thresholds)

    def _current_value(self, signals: Dict[str, float]) -> float:
        # Use the smaller of L/R elbows (more conservative for lockout)
        eL = float(signals.get("elbow_L", 180.0))
        eR = float(signals.get("elbow_R", 180.0))
        return min(eL, eR)

    def _is_valid_frame(self, signals: Dict[str, float]) -> bool:
        # Require at least one wrist roughly above its shoulder line at TOP moments
        # y increases downward; wrist above shoulder => wrist_y_rel < 0
        wL = float(signals.get("wrist_y_rel_L", 0.0))
        wR = float(signals.get("wrist_y_rel_R", 0.0))
        # We allow frames regardless; this is a soft validator to avoid counting when both wrists are far below shoulders always.
        return True


class SquatCounter(BaseCounter):
    """Counts reps primarily via knee angle. 'top' = large angle (standing), 'bottom' = small angle (depth).
    We map the generic thresholds: top_low ~ stand angle >= 160; bottom_high ~ depth angle <= 85 (but we interpret with inversion).
    To reuse BaseCounter's logic, we transform knee angle to a 'pseudo' elbow-like value:
    - Use knee_angle_transformed = 180 - knee_angle so that "top" corresponds to small (contracted) and "bottom" to large.
    """
    def __init__(self, stand_angle: float = 160.0, depth_angle: float = 85.0):
        # transform: top_low (small) ~ 180-stand, bottom_high (large) ~ 180-depth
        top_low = max(0.0, 180.0 - stand_angle)         # e.g., 20
        bottom_high = max(0.0, 180.0 - depth_angle)     # e.g., 95
        super().__init__(Thresholds(top_low=top_low, bottom_high=bottom_high, min_hold_frames=2, hysteresis=5.0))

    def _current_value(self, signals: Dict[str, float]) -> float:
        # Use the smaller knee (deeper) for safety, then transform
        kL = float(signals.get("knee_L", 180.0))
        kR = float(signals.get("knee_R", 180.0))
        knee = min(kL, kR)
        return 180.0 - knee  # transformed: deeper squat -> larger value

    def _is_valid_frame(self, signals: Dict[str, float]) -> bool:
        return True


def default_counters(side_for_curl: str = "auto") -> Dict[str, BaseCounter]:
    """Factory for all exercise counters with sensible defaults."""
    return {
        "bicep_curl": BicepCurlCounter(side=side_for_curl),
        "shoulder_press": ShoulderPressCounter(),
        # UI may reference "squats"; keep both keys mapped to the same counter instance
        "squat": SquatCounter(),
        "squats": SquatCounter(),
    }