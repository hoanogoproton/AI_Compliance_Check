from detection.behavior_detector import register_behavior
from detection.behaviors.base import BaseBehavior, DetectionResult
from detection.pose_utils import get_keypoint
from detection.zones.zone_definition import Zone


@register_behavior("hand_in_zone")
class HandInZoneBehavior(BaseBehavior):
    name = "hand_in_zone"

    def __init__(self, params: dict, zone: Zone | None = None):
        self.zone = zone
        super().__init__(params)

    def _validate_params(self):
        if self.zone is None:
            raise ValueError("hand_in_zone behavior requires a 'zone' parameter")

    def detect_person(self, person, frame, frame_idx, timestamp) -> DetectionResult:
        kpts = person.keypoints
        hand_filter = self.params.get("hand", "any")
        wrist_indices: list[tuple[int, str]] = []
        if hand_filter in ("any", "left"):
            wrist_indices.append((9, "left"))
        if hand_filter in ("any", "right"):
            wrist_indices.append((10, "right"))
        if hand_filter == "both":
            wrist_indices.extend([(9, "left"), (10, "right")])

        in_zone_sides = []
        for idx, side_name in wrist_indices:
            wx, wy, wc = get_keypoint(kpts, idx)
            if wc < 0.3:
                continue
            if self.zone.contains_point(wx, wy):
                in_zone_sides.append(side_name)

        if not in_zone_sides:
            return DetectionResult(
                track_id=person.track_id, detected=False, confidence=0.0,
                metadata={"side": "none", "zone": self.zone.name},
            )

        side = in_zone_sides[0] if len(in_zone_sides) == 1 else "both"
        return DetectionResult(
            track_id=person.track_id, detected=True, confidence=1.0,
            metadata={"side": side, "zone": self.zone.name},
        )
