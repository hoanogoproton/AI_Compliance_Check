from detection.behavior_detector import register_behavior
from detection.behaviors.base import BaseBehavior, DetectionResult
from detection.event_manager import Event
from detection.zones.zone_definition import Zone


@register_behavior("leave_zone")
class LeaveZoneBehavior(BaseBehavior):
    name = "leave_zone"

    def __init__(self, params: dict, zones: list[Zone] | None = None):
        super().__init__(params, zones=zones)
        self._track_inside: dict[int, bool] = {}
        self._track_inside_counter: dict[int, int] = {}
        self._track_outside_counter: dict[int, int] = {}
        self._track_inside_zones: dict[int, set[str]] = {}
        self._last_leave_frame: dict[int, int] = {}
        self._last_leave_zones: dict[int, list[str]] = {}

    def _validate_params(self):
        if len(self.zones) == 0:
            raise ValueError("leave_zone behavior requires at least one zone")

    def is_person_in_flash(self, track_id: int, frame_idx: int) -> bool:
        leave_frame = self._last_leave_frame.get(track_id)
        if leave_frame is None:
            return False
        flash_frames = self.params.get("leave_flash_frames", 20)
        return frame_idx - leave_frame <= flash_frames

    def process_frame(self, people, frame, frame_idx, timestamp) -> list[Event]:
        new_events = []

        for person in people:
            result = self.detect_person(person, frame, frame_idx, timestamp)
            if result.detected:
                event = Event(
                    track_id=person.track_id,
                    start_frame=frame_idx,
                    end_frame=frame_idx,
                    start_time=timestamp,
                    end_time=timestamp,
                    max_confidence=result.confidence,
                    frames=[frame_idx],
                    hand_sides=[result.metadata.get("side", "none")],
                    metadata=result.metadata,
                )
                event.behavior_name = self.name
                new_events.append(event)
        return new_events

    def detect_person(self, person, frame, frame_idx, timestamp) -> DetectionResult:
        tid = person.track_id
        min_stay = self.params.get("min_stay_frames", 10)
        min_leave = self.params.get("min_leave_frames", 3)
        bbox = person.bbox
        is_inside = any(z.contains_bbox_center(bbox) for z in self.zones)
        inside_zones = {z.name for z in self.zones if z.contains_bbox_center(bbox)}
        first_zone = self.zones[0].name

        if tid not in self._track_inside:
            self._track_inside[tid] = False
            self._track_inside_counter[tid] = 0
            self._track_outside_counter[tid] = 0
            self._track_inside_zones[tid] = set()

        if is_inside:
            self._track_outside_counter[tid] = 0
            self._track_inside_counter[tid] += 1
            self._track_inside_zones.setdefault(tid, set()).update(inside_zones)
            if self._track_inside_counter[tid] >= min_stay:
                self._track_inside[tid] = True
            return DetectionResult(
                track_id=tid, detected=False, confidence=0.0,
                metadata={
                    "side": "none",
                    "zone": first_zone,
                    "triggered_zones": sorted(self._track_inside_zones[tid]),
                    "inside": True,
                },
            )
        else:
            self._track_inside_counter[tid] = 0
            if self._track_inside.get(tid, False):
                self._track_outside_counter[tid] += 1
                if self._track_outside_counter[tid] >= min_leave:
                    left_zones = sorted(self._track_inside_zones.pop(tid, set()))
                    self._track_inside[tid] = False
                    self._track_outside_counter[tid] = 0
                    self._last_leave_frame[tid] = frame_idx
                    self._last_leave_zones[tid] = left_zones
                    return DetectionResult(
                        track_id=tid, detected=True, confidence=1.0,
                        metadata={
                            "side": "outside",
                            "zone": left_zones[0] if left_zones else first_zone,
                            "triggered_zones": left_zones,
                            "inside": False,
                        },
                    )
                return DetectionResult(
                    track_id=tid, detected=False, confidence=0.0,
                    metadata={
                        "side": "none",
                        "zone": first_zone,
                        "triggered_zones": sorted(self._track_inside_zones.get(tid, set())),
                        "inside": True,
                    },
                )
            return DetectionResult(
                track_id=tid, detected=False, confidence=0.0,
                metadata={
                    "side": "none",
                    "zone": first_zone,
                    "triggered_zones": [],
                    "inside": False,
                },
            )