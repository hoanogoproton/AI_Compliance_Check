from collections import deque

from detection.behavior_detector import register_behavior
from detection.behaviors.base import BaseBehavior, DetectionResult
from detection.config import (
    BODY_TURN_MAX_GAP_FRAMES,
    BODY_TURN_MAX_STEP_DEG,
    BODY_TURN_MIN_ANGLE,
    BODY_TURN_RELEASE_RATIO,
    BODY_TURN_RESULTANT_MIN,
    BODY_TURN_SMOOTHING_FRAMES,
    BODY_TURN_STALE_FRAMES,
    BODY_TURN_VELOCITY_THRESHOLD,
    BODY_TURN_WINDOW_SECONDS,
)
from detection.pose_utils import (
    circular_mean_resultant_deg,
    compute_body_orientation,
    signed_angle_step_deg,
)


def circular_mean_deg(angles: list[float]) -> float:
    mean, _ = circular_mean_resultant_deg(angles)
    return mean


def least_squares_slope(points: list[tuple[float, float]]) -> float:
    n = len(points)
    if n < 2:
        return 0.0
    st = sum(p[0] for p in points)
    sv = sum(p[1] for p in points)
    stt = sum(p[0] * p[0] for p in points)
    stv = sum(p[0] * p[1] for p in points)
    denom = n * stt - st * st
    if abs(denom) < 1e-9:
        return 0.0
    return (n * stv - st * sv) / denom


@register_behavior("body_turn")
class BodyTurnBehavior(BaseBehavior):
    name = "body_turn"

    def __init__(self, params: dict, **kwargs):
        super().__init__(params, **kwargs)
        self._angle_history: dict[int, deque[tuple[float, int, float]]] = {}
        self._smoothed_angles: dict[int, deque[float]] = {}
        self._cumulative: dict[int, float] = {}
        self._last_stable_angle: dict[int, float] = {}
        self._holdoff: dict[int, bool] = {}
        self._last_seen_frame: dict[int, int] = {}

    def _reset_track(self, tid: int) -> None:
        self._angle_history.pop(tid, None)
        self._smoothed_angles.pop(tid, None)
        self._cumulative.pop(tid, None)
        self._last_stable_angle.pop(tid, None)
        self._holdoff.pop(tid, None)

    def _prune_stale(self, tid: int) -> None:
        for store in (
            self._angle_history,
            self._smoothed_angles,
            self._cumulative,
            self._last_stable_angle,
            self._holdoff,
            self._last_seen_frame,
        ):
            store.pop(tid, None)

    @staticmethod
    def _empty_result(tid: int, side: str = "none") -> DetectionResult:
        return DetectionResult(
            track_id=tid,
            detected=False,
            confidence=0.0,
            metadata={"delta_deg": 0.0, "velocity_deg_s": 0.0, "side": side},
        )

    def detect_person(self, person, frame, frame_idx, timestamp) -> DetectionResult:
        tid = person.track_id

        min_angle = max(float(self.params.get("min_angle", BODY_TURN_MIN_ANGLE)), 1e-3)
        velocity_threshold = max(
            float(self.params.get("velocity_threshold_deg_s", BODY_TURN_VELOCITY_THRESHOLD)),
            1e-3,
        )
        window_seconds = max(float(self.params.get("window_seconds", BODY_TURN_WINDOW_SECONDS)), 0.1)
        smoothing_frames = max(int(self.params.get("smoothing_frames", BODY_TURN_SMOOTHING_FRAMES)), 2)
        track_gap_frames = int(self.params.get("track_gap_frames", BODY_TURN_MAX_GAP_FRAMES))
        stale_frames = int(self.params.get("stale_frames", BODY_TURN_STALE_FRAMES))
        release_ratio = float(self.params.get("release_ratio", BODY_TURN_RELEASE_RATIO))
        resultant_min = float(self.params.get("resultant_min", BODY_TURN_RESULTANT_MIN))
        max_step_deg = float(self.params.get("max_step_deg", BODY_TURN_MAX_STEP_DEG))

        if frame_idx - self._last_seen_frame.get(tid, 0) > track_gap_frames:
            self._reset_track(tid)
        self._last_seen_frame[tid] = frame_idx

        if frame_idx % stale_frames == 0:
            for tid_ in [k for k, v in self._last_seen_frame.items() if frame_idx - v > stale_frames]:
                self._prune_stale(tid_)

        angle = compute_body_orientation(person.keypoints)
        if angle is None:
            return self._empty_result(tid)

        smooth_dq = self._smoothed_angles.setdefault(tid, deque(maxlen=smoothing_frames))
        smooth_dq.append(angle)

        smoothed, resultant = circular_mean_resultant_deg(list(smooth_dq))
        if resultant < resultant_min and len(smooth_dq) >= 2:
            return self._empty_result(tid)

        last_stable = self._last_stable_angle.get(tid)
        if last_stable is not None:
            step = signed_angle_step_deg(smoothed, last_stable)
            if abs(step) > max_step_deg:
                return self._empty_result(tid)
            self._cumulative[tid] = self._cumulative.get(tid, 0.0) + step
        else:
            self._cumulative.setdefault(tid, 0.0)
        self._last_stable_angle[tid] = smoothed

        cumulative_now = self._cumulative.get(tid, 0.0)

        hist = self._angle_history.setdefault(tid, deque())
        hist.append((timestamp, frame_idx, cumulative_now))

        history_seconds = max(window_seconds * 2.0, 2.0)
        cutoff_time = timestamp - history_seconds
        while hist and hist[0][0] < cutoff_time:
            hist.popleft()

        candidates = list(hist)[:-1]
        if not candidates:
            return self._empty_result(tid)

        ref_timestamp, _, ref_cumulative = min(
            candidates,
            key=lambda item: abs(item[0] - (timestamp - window_seconds)),
        )
        elapsed = timestamp - ref_timestamp
        if elapsed <= 0:
            return self._empty_result(tid)

        delta_signed = cumulative_now - ref_cumulative
        delta_abs = abs(delta_signed)

        regression_points = [
            (p[0], p[2]) for p in hist if p[0] >= timestamp - window_seconds
        ]
        span = regression_points[-1][0] - regression_points[0][0]
        if len(regression_points) >= 3 and span >= window_seconds * 0.3:
            slope = least_squares_slope(regression_points)
        else:
            slope = delta_signed / elapsed
        slope_abs = abs(slope)

        implied_velocity = min_angle / window_seconds
        effective_velocity_threshold = min(velocity_threshold, implied_velocity)

        ready_to_fire = elapsed >= window_seconds * 0.35

        side = "left" if delta_signed < 0 else "right"

        if self._holdoff.get(tid, False):
            detected = delta_abs >= min_angle * release_ratio
            if not detected:
                self._holdoff[tid] = False
        elif (
            ready_to_fire
            and delta_abs >= min_angle
            and slope_abs >= effective_velocity_threshold
        ):
            detected = True
            self._holdoff[tid] = True
        else:
            detected = False

        conf = (
            min(1.0, delta_abs / (min_angle * 2.0))
            * (0.5 + 0.5 * min(1.0, slope_abs / (effective_velocity_threshold * 2.0)))
        )

        return DetectionResult(
            track_id=tid,
            detected=detected,
            confidence=round(conf, 3),
            metadata={
                "delta_deg": round(delta_signed, 1),
                "velocity_deg_s": round(slope, 1),
                "side": side,
            },
        )
