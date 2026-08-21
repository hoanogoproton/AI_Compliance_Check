from collections import deque

from detection.behavior_detector import register_behavior
from detection.behaviors.base import BaseBehavior, DetectionResult
from detection.config import HEAD_TURN_MAX_TURNS, HEAD_TURN_THRESHOLD_RATIO, HEAD_TURN_WINDOW_FRAMES
from detection.pose_utils import compute_head_yaw_offset


@register_behavior("head_turn")
class HeadTurnBehavior(BaseBehavior):
    name = "head_turn"

    def __init__(self, params: dict, **kwargs):
        super().__init__(params, **kwargs)
        self._turn_frames: dict[int, deque[int]] = {}
        self._prev_side: dict[int, str] = {}

    def detect_person(self, person, frame, frame_idx, timestamp) -> DetectionResult:
        tid = person.track_id
        threshold = self.params.get("turn_threshold_ratio", HEAD_TURN_THRESHOLD_RATIO)
        window = int(self.params.get("window_frames", HEAD_TURN_WINDOW_FRAMES))
        max_turns = int(self.params.get("max_turns", HEAD_TURN_MAX_TURNS))

        result = compute_head_yaw_offset(person.keypoints)
        if result is None:
            return DetectionResult(
                track_id=tid,
                detected=False,
                confidence=0.0,
                metadata={"side": "none", "turns": self._current_turns(tid, frame_idx, window)},
            )

        offset, _ = result
        if offset < -threshold:
            side = "left"
        elif offset > threshold:
            side = "right"
        else:
            side = "center"

        turns = self._update_turns(tid, side, frame_idx, window)
        detected = turns > max_turns
        confidence = min(1.0, turns / float(max_turns + 1))
        return DetectionResult(
            track_id=tid,
            detected=detected,
            confidence=confidence,
            metadata={"side": side, "turns": turns},
        )

    def _update_turns(self, tid: int, side: str, frame_idx: int, window: int) -> int:
        prev = self._prev_side.get(tid, "center")
        dq = self._turn_frames.setdefault(tid, deque())
        if side in ("left", "right") and prev != side:
            dq.append(frame_idx)
        self._prev_side[tid] = side
        cutoff = frame_idx - window
        while dq and dq[0] <= cutoff:
            dq.popleft()
        return len(dq)

    def _current_turns(self, tid: int, frame_idx: int, window: int) -> int:
        dq = self._turn_frames.get(tid)
        if not dq:
            return 0
        cutoff = frame_idx - window
        while dq and dq[0] <= cutoff:
            dq.popleft()
        return len(dq)
