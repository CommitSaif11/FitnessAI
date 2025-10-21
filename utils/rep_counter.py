from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

@dataclass
class RepStats:
    count: int = 0
    # For current rep tracking
    in_rep: bool = False
    phase: Optional[str] = None  # "top" | "bottom" | None
    cur_min: float = 1e9
    cur_max: float = -1e9

@dataclass
class Thresholds:
    # Generic thresholds with hysteresis (degrees)
    top_low: float         # angle <= top_low means "top" (lockout/peak)
    bottom_high: float     # angle >= bottom_high means "bottom" (depth/extension)
    hysteresis: float = 10 # degrees to avoid double flips
    min_interval_frames: int = 8  # debounce between reps

@dataclass
class CounterState:
    stats: RepStats = field(default_factory=RepStats)
    frames_since_switch: int = 0

class BaseRepCounter:
    def __init__(self, thresholds: Thresholds):
        self.th = thresholds
        self.state = CounterState()

    def _is_top_enter(self, angle: float) -> bool:
        return angle <= self.th.top_low

    def _is_top_exit(self, angle: float) -> bool:
        return angle >= (self.th.top_low + self.th.hysteresis)

    def _is_bottom_enter(self, angle: float) -> bool:
        return angle >= self.th.bottom_high

    def _is_bottom_exit(self, angle: float) -> bool:
        return angle <= (self.th.bottom_high - self.th.hysteresis)

    def _update_phase(self, angle: float):
        s = self.state.stats
        # Update min/max for current rep
        s.cur_min = min(s.cur_min, angle)
        s.cur_max = max(s.cur_max, angle)

        # Debounce
        self.state.frames_since_switch += 1

        if s.phase is None:
            if self._is_top_enter(angle):
                s.phase = "top"
                self.state.frames_since_switch = 0
            elif self._is_bottom_enter(angle):
                s.phase = "bottom"
                self.state.frames_since_switch = 0
            return

        # Phase transitions with hysteresis
        if s.phase == "top":
            if self._is_bottom_enter(angle) and self.state.frames_since_switch >= self.th.min_interval_frames:
                s.phase = "bottom"
                s.in_rep = True  # we started moving into a rep
                # reset for this rep range
                s.cur_min, s.cur_max = angle, angle
                self.state.frames_since_switch = 0
        elif s.phase == "bottom":
            if self._is_top_enter(angle) and self.state.frames_since_switch >= self.th.min_interval_frames:
                s.phase = "top"
                # Completed a full bottom->top cycle
                if s.in_rep:
                    s.count += 1
                s.in_rep = False
                # reset current rep range
                s.cur_min, s.cur_max = angle, angle
                self.state.frames_since_switch = 0

    def get_stats(self) -> RepStats:
        return self.state.stats

    def reset(self):
        self.state = CounterState()

    def choose_metric(self, angles: Dict[str, float]) -> float:
        raise NotImplementedError

    def update(self, angles: Dict[str, float]) -> RepStats:
        metric_angle = self.choose_metric(angles)
        self._update_phase(metric_angle)
        return self.get_stats()

class BicepCurlCounter(BaseRepCounter):
    """
    Use elbow flexion angle. Smaller angle means flexed (top), larger means extended (bottom).
    By default, use the more flexed elbow (min of L/R) so whichever arm is active counts.
    """
    def __init__(self, thresholds: Thresholds, side: str = "auto"):
        super().__init__(thresholds)
        self.side = side  # "auto" | "left" | "right"

    def choose_metric(self, angles: Dict[str, float]) -> float:
        elL = angles.get("angle_elbow_L", 180.0)
        elR = angles.get("angle_elbow_R", 180.0)
        if self.side == "left":
            return elL
        if self.side == "right":
            return elR
        return min(elL, elR)

class ShoulderPressCounter(BaseRepCounter):
    """
    Use elbow angle as proxy for press lockout (top) and bottom range (down).
    """
    def __init__(self, thresholds: Thresholds):
        super().__init__(thresholds)

    def choose_metric(self, angles: Dict[str, float]) -> float:
        elL = angles.get("angle_elbow_L", 180.0)
        elR = angles.get("angle_elbow_R", 180.0)
        return min(elL, elR)

class SquatCounter(BaseRepCounter):
    """
    Use knee flexion angle for depth (bottom) and standing (top).
    """
    def __init__(self, thresholds: Thresholds, use_hip_backup: bool = True):
        super().__init__(thresholds)
        self.use_hip_backup = use_hip_backup

    def choose_metric(self, angles: Dict[str, float]) -> float:
        knL = angles.get("angle_knee_L", 180.0)
        knR = angles.get("angle_knee_R", 180.0)
        # deeper squat -> smaller angle; we consider the deeper (min)
        knee_metric = min(knL, knR)
        if self.use_hip_backup:
            # Combine with hip if knees are noisy
            hiL = angles.get("angle_hip_L", 180.0)
            hiR = angles.get("angle_hip_R", 180.0)
            hip_metric = min(hiL, hiR)  # smaller = more flexed
            # Blend (weighted) to stabilize
            return 0.7 * knee_metric + 0.3 * hip_metric
        return knee_metric

def default_counters(side_for_curl: str = "auto"):
    """
    Returns exercise -> counter mapping with sensible defaults.
    """
    return {
        "bicep_curl": BicepCurlCounter(Thresholds(top_low=60, bottom_high=150, hysteresis=10, min_interval_frames=6), side=side_for_curl),
        "shoulder_press": ShoulderPressCounter(Thresholds(top_low=25, bottom_high=95, hysteresis=10, min_interval_frames=6)),
        "squats": SquatCounter(Thresholds(top_low=160, bottom_high=85, hysteresis=10, min_interval_frames=8))
    }