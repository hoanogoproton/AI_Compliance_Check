from collections import deque
from math import atan2, cos, degrees, radians, sin

from detection.behavior_detector import register_behavior
from detection.behaviors.base import BaseBehavior, DetectionResult
from detection.config import (
    BODY_TURN_COOLDOWN_FRAMES,
    BODY_TURN_MAX_GAP_FRAMES,
    BODY_TURN_MIN_ANGLE,
    BODY_TURN_SMOOTHING_FRAMES,
    BODY_TURN_STALE_FRAMES,
    BODY_TURN_VELOCITY_THRESHOLD,
    BODY_TURN_WINDOW_SECONDS,
)
from detection.pose_utils import angular_difference_deg, compute_body_orientation


def circular_mean_deg(angles: list[float]) -> float:
    if not angles:
        return 0.0
    x = sum(cos(radians(a)) for a in angles)
    y = sum(sin(radians(a)) for a in angles)
    return degrees(atan2(y, x))


@register_behavior("body_turn")
class BodyTurnBehavior(BaseBehavior):
    name = "body_turn"

    def __init__(self, params: dict, **kwargs):
        super().__init__(params)
        self._angle_history: dict[int, deque[tuple[float, int, float]]] = {}
        self._smoothed_angles: dict[int, deque[float]] = {}
        self._last_seen_frame: dict[int, int] = {}
        self._last_detection_frame: dict[int, int] = {}

    def _reset_track(self, tid: int) -> None:
        self._angle_history.pop(tid, None)
        self._smoothed_angles.pop(tid, None)

    def detect_person(self, person, frame, frame_idx, timestamp) -> DetectionResult:
        tid = person.track_id

        min_angle = float(self.params.get("min_angle", BODY_TURN_MIN_ANGLE))
        velocity_threshold = float(
            self.params.get("velocity_threshold_deg_s", BODY_TURN_VELOCITY_THRESHOLD)
        )
        window_seconds = float(self.params.get("window_seconds", BODY_TURN_WINDOW_SECONDS))
        smoothing_frames = int(self.params.get("smoothing_frames", BODY_TURN_SMOOTHING_FRAMES))
        track_gap_frames = int(self.params.get("track_gap_frames", BODY_TURN_MAX_GAP_FRAMES))
        cooldown_frames = int(self.params.get("cooldown_frames", BODY_TURN_COOLDOWN_FRAMES))
        stale_frames = int(self.params.get("stale_frames", BODY_TURN_STALE_FRAMES))

        if frame_idx - self._last_seen_frame.get(tid, 0) > track_gap_frames:
            self._reset_track(tid)
        self._last_seen_frame[tid] = frame_idx

        angle = compute_body_orientation(person.keypoints)
        if angle is None:
            return DetectionResult(
                track_id=tid, detected=False, confidence=0.0,
                metadata={"delta_deg": 0.0, "velocity_deg_s": 0.0},
            )

        smooth_dq = self._smoothed_angles.setdefault(tid, deque(maxlen=smoothing_frames))
        smooth_dq.append(angle)
        smoothed = circular_mean_deg(list(smooth_dq))

        hist = self._angle_history.setdefault(tid, deque())
        hist.append((timestamp, frame_idx, smoothed))

        history_seconds = max(window_seconds * 2.0, 2.0)
        cutoff_time = timestamp - history_seconds
        while hist and hist[0][0] < cutoff_time:
            hist.popleft()

        target_time = timestamp - window_seconds
        candidates = list(hist)[:-1]
        if not candidates:
            return DetectionResult(
                track_id=tid, detected=False, confidence=0.0,
                metadata={"delta_deg": 0.0, "velocity_deg_s": 0.0},
            )

        ref_timestamp, ref_frame, ref_angle = min(
            candidates,
            key=lambda item: abs(item[0] - target_time),
        )

        elapsed = timestamp - ref_timestamp
        if elapsed <= 0:
            return DetectionResult(
                track_id=tid, detected=False, confidence=0.0,
                metadata={"delta_deg": 0.0, "velocity_deg_s": 0.0},
            )

        delta = angular_difference_deg(smoothed, ref_angle)
        velocity_deg_s = delta / elapsed

        last_det = self._last_detection_frame.get(tid, -10**9)
        can_emit = (frame_idx - last_det) >= cooldown_frames

        detected = can_emit and delta >= min_angle and velocity_deg_s >= velocity_threshold
        if detected:
            self._last_detection_frame[tid] = frame_idx

        conf = min(1.0, delta / 90.0) * min(1.0, velocity_deg_s / (velocity_threshold + 1e-6))

        if frame_idx % stale_frames == 0:
            for tid_ in list(self._last_seen_frame.keys()):
                if frame_idx - self._last_seen_frame[tid_] > stale_frames:
                    self._last_seen_frame.pop(tid_, None)
                    self._angle_history.pop(tid_, None)
                    self._smoothed_angles.pop(tid_, None)

        return DetectionResult(
            track_id=tid,
            detected=detected,
            confidence=round(conf, 3),
            metadata={
                "delta_deg": round(delta, 1),
                "velocity_deg_s": round(velocity_deg_s, 1),
            },
        )