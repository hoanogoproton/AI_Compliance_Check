import numpy as np

from detection.behavior_detector import register_behavior
from detection.behaviors.base import BaseBehavior, DetectionResult
from detection.config import DISTANCE_THRESHOLD_RATIO, KEYPOINT_CONFIDENCE_THRESHOLD, HEAD_KEYPOINT_CONFIDENCE_THRESHOLD, VERTICAL_OFFSET_RATIO
from detection.pose_utils import compute_head_center, compute_shoulder_width, get_keypoint


@register_behavior("hand_to_head")
class HandToHeadBehavior(BaseBehavior):
    name = "hand_to_head"

    def detect_person(self, person, frame, frame_idx, timestamp) -> DetectionResult:
        kpts = person.keypoints
        params = self.params
        threshold_ratio = params.get("distance_threshold_ratio", DISTANCE_THRESHOLD_RATIO)
        vertical_offset = params.get("vertical_offset_ratio", VERTICAL_OFFSET_RATIO)
        kpt_conf = params.get("keypoint_conf_threshold", KEYPOINT_CONFIDENCE_THRESHOLD)
        head_conf = params.get("head_keypoint_conf_threshold", HEAD_KEYPOINT_CONFIDENCE_THRESHOLD)

        head_center = compute_head_center(kpts)
        if head_center is None:
            return DetectionResult(track_id=person.track_id, detected=False, confidence=0.0, metadata={"side": "none"})
        shoulder_width = compute_shoulder_width(kpts)
        if shoulder_width < 20.0:
            return DetectionResult(track_id=person.track_id, detected=False, confidence=0.0, metadata={"side": "none"})
        max_allowed = shoulder_width * threshold_ratio
        hx, hy = head_center
        detected = False
        best_conf = 0.0
        sides = []
        for wrist_idx, side_name in [(9, "left"), (10, "right")]:
            wx, wy, wc = get_keypoint(kpts, wrist_idx)
            if wc < kpt_conf:
                continue
            if wy > hy + shoulder_width * vertical_offset:
                continue
            dist = np.sqrt((wx - hx) ** 2 + (wy - hy) ** 2)
            if dist < max_allowed:
                detected = True
                conf = 1.0 - (dist / max_allowed)
                if conf > best_conf:
                    best_conf = conf
                sides.append(side_name)
        if not detected:
            return DetectionResult(track_id=person.track_id, detected=False, confidence=0.0, metadata={"side": "none"})
        side = "both" if len(sides) == 2 else sides[0]
        return DetectionResult(
            track_id=person.track_id,
            detected=True,
            confidence=best_conf,
            metadata={"side": side},
        )
