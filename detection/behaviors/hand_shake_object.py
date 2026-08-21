from collections import deque

from detection.behavior_detector import register_behavior
from detection.behaviors.base import BaseBehavior, DetectionResult
from detection.config import (
    HAND_SHAKE_OBJECT_MIN_DISPLACEMENT_RATIO,
    HAND_SHAKE_OBJECT_MIN_REVERSALS,
    HAND_SHAKE_OBJECT_WINDOW_FRAMES,
)
from detection.pose_utils import compute_shoulder_width, get_keypoint
from detection.zones.zone_definition import Zone


@register_behavior("hand_shake_object")
class HandShakeObjectBehavior(BaseBehavior):
    name = "hand_shake_object"

    def __init__(self, params: dict, zones: list[Zone] | None = None):
        self.zones = zones or []
        super().__init__(params)
        self._prev_wrist_pos: dict[tuple[int, str], tuple[float, float] | None] = {}
        self._wrist_x_dir: dict[tuple[int, str], int] = {}
        self._wrist_y_dir: dict[tuple[int, str], int] = {}
        self._wrist_reversals: dict[tuple[int, str], deque[int]] = {}
        self._last_detections: dict[int, bool] = {}

    def _validate_params(self):
        if len(self.zones) == 0:
            raise ValueError("hand_shake_object behavior requires at least one zone")

    def detect_person(self, person, frame, frame_idx, timestamp) -> DetectionResult:
        tid = person.track_id
        kpts = person.keypoints
        window = int(self.params.get("window_frames", HAND_SHAKE_OBJECT_WINDOW_FRAMES))
        min_rev = int(self.params.get("min_reversals", HAND_SHAKE_OBJECT_MIN_REVERSALS))
        ratio = float(self.params.get("min_displacement_ratio", HAND_SHAKE_OBJECT_MIN_DISPLACEMENT_RATIO))
        conf_thresh = float(self.params.get("keypoint_conf_threshold", 0.5))

        shoulder_width = compute_shoulder_width(kpts)
        if shoulder_width <= 0:
            self._last_detections[tid] = False
            return DetectionResult(track_id=tid, detected=False, confidence=0.0,
                                   metadata={"hand": "none", "zone": self.zones[0].name, "triggered_zones": []})
        min_disp = shoulder_width * ratio

        hands = [
            ("left", get_keypoint(kpts, 9)),
            ("right", get_keypoint(kpts, 10)),
        ]

        triggered_zones = set()

        for hand_name, wrist in hands:
            wx, wy, wc = wrist
            if wc < conf_thresh:
                continue

            in_zone = any(z.contains_point(wx, wy) for z in self.zones)
            if not in_zone:
                continue

            for z in self.zones:
                if z.contains_point(wx, wy):
                    triggered_zones.add(z.name)

            key = (tid, hand_name)
            prev_wrist = self._prev_wrist_pos.get(key, None)
            wrist_revs = self._wrist_reversals.setdefault(key, deque())

            if prev_wrist is not None:
                dx = wx - prev_wrist[0]
                dy = wy - prev_wrist[1]

                x_dir = 1 if dx > 0 else (-1 if dx < 0 else 0)
                y_dir = 1 if dy > 0 else (-1 if dy < 0 else 0)

                prev_x = self._wrist_x_dir.get(key, 0)
                prev_y = self._wrist_y_dir.get(key, 0)

                if x_dir != 0 and prev_x != 0 and x_dir != prev_x and abs(dx) > min_disp:
                    wrist_revs.append(frame_idx)
                elif y_dir != 0 and prev_y != 0 and y_dir != prev_y and abs(dy) > min_disp:
                    wrist_revs.append(frame_idx)

                self._wrist_x_dir[key] = x_dir
                self._wrist_y_dir[key] = y_dir

            self._prev_wrist_pos[key] = (wx, wy)

            cutoff = frame_idx - window
            while wrist_revs and wrist_revs[0] <= cutoff:
                wrist_revs.popleft()

            count = len(wrist_revs)
            detected = count >= min_rev
            conf = min(1.0, count / float(min_rev))

            self._last_detections[tid] = detected

            return DetectionResult(
                track_id=tid,
                detected=detected,
                confidence=conf,
                metadata={
                    "hand": hand_name,
                    "wrist_reversals": count,
                    "zone": self.zones[0].name,
                    "triggered_zones": sorted(triggered_zones),
                },
            )

        self._last_detections[tid] = False
        return DetectionResult(track_id=tid, detected=False, confidence=0.0,
                               metadata={"hand": "none", "zone": self.zones[0].name, "triggered_zones": []})