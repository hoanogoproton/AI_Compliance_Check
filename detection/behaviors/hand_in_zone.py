from detection.behavior_detector import register_behavior
from detection.behaviors.base import BaseBehavior, DetectionResult
from detection.pose_utils import get_keypoint
from detection.zones.zone_definition import Zone


@register_behavior("hand_in_zone")
class HandInZoneBehavior(BaseBehavior):
    name = "hand_in_zone"

    def __init__(self, params: dict, zones: list[Zone] | None = None):
        super().__init__(params, zones=zones)

    def _validate_params(self):
        if len(self.zones) == 0:
            raise ValueError("hand_in_zone behavior requires at least one zone")

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
        triggered_zones = set()
        for idx, side_name in wrist_indices:
            wx, wy, wc = get_keypoint(kpts, idx)
            if wc < 0.3:
                continue
            for z in self.zones:
                if z.contains_point(wx, wy):
                    in_zone_sides.append(side_name)
                    triggered_zones.add(z.name)

        if not in_zone_sides:
            return DetectionResult(
                track_id=person.track_id, detected=False, confidence=0.0,
                metadata={"side": "none", "zone": self.zones[0].name, "triggered_zones": []},
            )

        side = in_zone_sides[0] if len(in_zone_sides) == 1 else "both"
        first_zone = self.zones[0].name
        return DetectionResult(
            track_id=person.track_id, detected=True, confidence=1.0,
            metadata={
                "side": side,
                "zone": first_zone,
                "triggered_zones": sorted(triggered_zones),
            },
        )