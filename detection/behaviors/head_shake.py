from collections import deque

from detection.behavior_detector import register_behavior
from detection.behaviors.base import BaseBehavior, DetectionResult
from detection.config import (
    HEAD_SHAKE_EMA_DEADBAND,
    HEAD_SHAKE_MAX_REPROJECTION_ERROR,
    HEAD_SHAKE_MIN_FACE_CONFIDENCE,
    HEAD_SHAKE_MIN_REVERSALS,
    HEAD_SHAKE_SMOOTHING_ALPHA,
    HEAD_SHAKE_WINDOW_FRAMES,
    HEAD_SHAKE_YAW_AMPLITUDE_THRESHOLD,
)


class YawShakeTracker:
    def __init__(
        self,
        window_frames: int = HEAD_SHAKE_WINDOW_FRAMES,
        yaw_amplitude_threshold: float = HEAD_SHAKE_YAW_AMPLITUDE_THRESHOLD,
        smoothing_alpha: float = HEAD_SHAKE_SMOOTHING_ALPHA,
        ema_deadband: float = HEAD_SHAKE_EMA_DEADBAND,
        max_gap_frames: int = 5,
        min_reversals: int = HEAD_SHAKE_MIN_REVERSALS,
    ):
        self.window_frames = window_frames
        self.yaw_amplitude_threshold = yaw_amplitude_threshold
        self.smoothing_alpha = smoothing_alpha
        self.ema_deadband = ema_deadband
        self.max_gap_frames = max_gap_frames
        self.min_reversals = min_reversals
        self._direction: str = "none"
        self._peak_yaw: float = 0.0
        self._trough_yaw: float = 0.0
        self._peak_frame: int = -1
        self._trough_frame: int = -1
        self._reversal_count: int = 0
        self._reversal_frames: deque[int] = deque()
        self._last_yaw: float | None = None
        self._smoothed_yaw: float | None = None
        self._last_valid_frame: int | None = None
        self._valid_samples: deque[tuple[int, float, float]] = deque()

    @property
    def reversal_count(self) -> int:
        return self._reversal_count

    def reset(self):
        self._direction = "none"
        self._peak_yaw = 0.0
        self._trough_yaw = 0.0
        self._peak_frame = -1
        self._trough_frame = -1
        self._reversal_count = 0
        self._reversal_frames.clear()
        self._last_yaw = None
        self._smoothed_yaw = None
        self._last_valid_frame = None
        self._valid_samples.clear()

    def update(self, yaw: float | None, frame_idx: int) -> int:
        if yaw is None:
            return self._reversal_count
        if self._last_valid_frame is not None:
            gap = frame_idx - self._last_valid_frame
            if gap > self.max_gap_frames:
                self._direction = "none"
                self._peak_yaw = 0.0
                self._trough_yaw = 0.0
                self._peak_frame = -1
                self._trough_frame = -1
                self._last_yaw = None
                self._smoothed_yaw = None
        self._last_valid_frame = frame_idx
        if self._smoothed_yaw is None:
            self._smoothed_yaw = yaw
        else:
            self._smoothed_yaw = (
                self.smoothing_alpha * yaw
                + (1.0 - self.smoothing_alpha) * self._smoothed_yaw
            )
        raw_yaw = self._smoothed_yaw
        if self._last_yaw is not None:
            diff = raw_yaw - self._last_yaw
            prev_direction = self._direction
            if diff > self.ema_deadband:
                new_direction = "rising"
            elif diff < -self.ema_deadband:
                new_direction = "falling"
            else:
                new_direction = self._direction
            if new_direction != prev_direction:
                if prev_direction == "rising" and new_direction == "falling":
                    amplitude = self._peak_yaw - raw_yaw
                    if amplitude >= self.yaw_amplitude_threshold:
                        self._reversal_count += 1
                        self._reversal_frames.append(frame_idx)
                elif prev_direction == "falling" and new_direction == "rising":
                    amplitude = raw_yaw - self._trough_yaw
                    if amplitude >= self.yaw_amplitude_threshold:
                        self._reversal_count += 1
                        self._reversal_frames.append(frame_idx)
            if new_direction == "rising":
                self._peak_yaw = max(self._peak_yaw, raw_yaw)
                self._peak_frame = frame_idx
            elif new_direction == "falling":
                self._trough_yaw = min(self._trough_yaw, raw_yaw)
                self._trough_frame = frame_idx
            self._direction = new_direction
        else:
            self._peak_yaw = raw_yaw
            self._trough_yaw = raw_yaw
            self._peak_frame = frame_idx
            self._trough_frame = frame_idx
        self._last_yaw = raw_yaw
        self._valid_samples.append((frame_idx, yaw, raw_yaw))
        cutoff = frame_idx - self.window_frames
        while self._reversal_frames and self._reversal_frames[0] <= cutoff:
            self._reversal_frames.popleft()
            self._reversal_count -= 1
        while self._valid_samples and self._valid_samples[0][0] <= cutoff:
            self._valid_samples.popleft()
        return self._reversal_count


@register_behavior("head_shake")
class HeadShakeBehavior(BaseBehavior):
    name = "head_shake"

    def __init__(self, params: dict, **kwargs):
        super().__init__(params, **kwargs)
        window_frames = int(params.get("window_frames", HEAD_SHAKE_WINDOW_FRAMES))
        yaw_amplitude_threshold = float(params.get(
            "yaw_amplitude_threshold", HEAD_SHAKE_YAW_AMPLITUDE_THRESHOLD
        ))
        min_face_confidence = float(params.get(
            "min_face_confidence", HEAD_SHAKE_MIN_FACE_CONFIDENCE
        ))
        max_reprojection_error = float(params.get(
            "max_reprojection_error", HEAD_SHAKE_MAX_REPROJECTION_ERROR
        ))
        smoothing_alpha = float(params.get(
            "smoothing_alpha", HEAD_SHAKE_SMOOTHING_ALPHA
        ))
        ema_deadband = float(params.get(
            "ema_deadband", HEAD_SHAKE_EMA_DEADBAND
        ))
        max_gap_frames = int(params.get("max_gap_frames", 5))
        self.min_reversals = int(params.get("min_reversals", HEAD_SHAKE_MIN_REVERSALS))
        self.min_face_confidence = min_face_confidence
        self.max_reprojection_error = max_reprojection_error
        self._trackers: dict[int, YawShakeTracker] = {}

    def _get_tracker(self, tid: int) -> YawShakeTracker:
        if tid not in self._trackers:
            self._trackers[tid] = YawShakeTracker(
                window_frames=int(self.params.get("window_frames", HEAD_SHAKE_WINDOW_FRAMES)),
                yaw_amplitude_threshold=float(self.params.get(
                    "yaw_amplitude_threshold", HEAD_SHAKE_YAW_AMPLITUDE_THRESHOLD
                )),
                smoothing_alpha=float(self.params.get(
                    "smoothing_alpha", HEAD_SHAKE_SMOOTHING_ALPHA
                )),
                ema_deadband=float(self.params.get(
                    "ema_deadband", HEAD_SHAKE_EMA_DEADBAND
                )),
                max_gap_frames=int(self.params.get("max_gap_frames", 5)),
                min_reversals=self.min_reversals,
            )
        return self._trackers[tid]

    def detect_person(self, person, frame, frame_idx, timestamp) -> DetectionResult:
        tid = person.track_id
        if person.face_data is None or not person.face_data.pose_valid:
            tracker = self._get_tracker(tid)
            tracker.update(None, frame_idx)
            return DetectionResult(
                track_id=tid,
                detected=False,
                confidence=0.0,
                metadata={
                    "yaw": None, "pitch": None, "roll": None,
                    "reversals": tracker.reversal_count,
                    "has_face": False,
                    "face_confidence": 0.0,
                    "direction": "unknown",
                },
            )
        face = person.face_data
        tracker = self._get_tracker(tid)
        reversal_count = tracker.update(face.yaw, frame_idx)
        detected = reversal_count >= self.min_reversals
        confidence = min(1.0, reversal_count / float(self.min_reversals + 1))
        return DetectionResult(
            track_id=tid,
            detected=detected,
            confidence=confidence,
            metadata={
                "yaw": face.yaw,
                "pitch": face.pitch,
                "roll": face.roll,
                "reversals": reversal_count,
                "has_face": face.pose_valid,
                "face_confidence": face.detection_confidence,
                "direction": tracker._direction,
            },
        )