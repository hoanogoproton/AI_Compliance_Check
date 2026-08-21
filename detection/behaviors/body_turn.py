from collections import deque

from detection.behavior_detector import register_behavior
from detection.behaviors.base import BaseBehavior, DetectionResult
from detection.config import BODY_TURN_MIN_ANGLE, BODY_TURN_VELOCITY_THRESHOLD, BODY_TURN_WINDOW_FRAMES
from detection.pose_utils import angular_difference_deg, compute_body_orientation


@register_behavior("body_turn")
class BodyTurnBehavior(BaseBehavior):
    name = "body_turn"

    def __init__(self, params: dict, **kwargs):
        super().__init__(params)
        self._angle_history: dict[int, deque[tuple[int, float]]] = {}
        self._smoothed_angles: dict[int, deque[float]] = {}

    def detect_person(self, person, frame, frame_idx, timestamp) -> DetectionResult:
        tid = person.track_id
        min_angle = self.params.get("min_angle", BODY_TURN_MIN_ANGLE)
        window = int(self.params.get("window_frames", BODY_TURN_WINDOW_FRAMES))
        velocity_threshold = self.params.get("velocity_threshold", BODY_TURN_VELOCITY_THRESHOLD)

        angle = compute_body_orientation(person.keypoints)
        if angle is None:
            return DetectionResult(track_id=tid, detected=False, confidence=0.0,
                                   metadata={"delta_deg": 0.0, "velocity": 0.0})

        smooth_dq = self._smoothed_angles.setdefault(tid, deque(maxlen=3))
        smooth_dq.append(angle)
        smoothed = sum(smooth_dq) / len(smooth_dq)

        hist = self._angle_history.setdefault(tid, deque())
        hist.append((frame_idx, smoothed))
        cutoff = frame_idx - window * 2
        while hist and hist[0][0] <= cutoff:
            hist.popleft()

        reference_idx = frame_idx - window
        ref_angle = None
        for f, a in hist:
            if f >= reference_idx:
                ref_angle = a
                break
        if ref_angle is None and len(hist) > 1:
            ref_angle = hist[0][1]

        if ref_angle is None:
            return DetectionResult(track_id=tid, detected=False, confidence=0.0,
                                   metadata={"delta_deg": 0.0, "velocity": 0.0})

        delta = angular_difference_deg(smoothed, ref_angle)
        velocity = delta / max(1, (frame_idx - (hist[0][0] if len(hist) > 1 else frame_idx)))

        detected = delta >= min_angle and velocity >= velocity_threshold

        conf = min(1.0, delta / 90.0) * min(1.0, velocity / velocity_threshold)

        return DetectionResult(
            track_id=tid,
            detected=detected,
            confidence=conf,
            metadata={"delta_deg": round(delta, 1), "velocity": round(velocity, 1)},
        )