from detection.behavior_detector import register_behavior
from detection.behaviors.base import BaseBehavior, DetectionResult
from detection.event_manager import Event
from detection.zones.zone_definition import Zone


@register_behavior("leave_zone")
class LeaveZoneBehavior(BaseBehavior):
    name = "leave_zone"

    def __init__(self, params: dict, zone: Zone | None = None):
        self.zone = zone
        super().__init__(params)
        self._track_inside: dict[int, bool] = {}
        self._track_inside_counter: dict[int, int] = {}
        self._last_leave_frame: dict[int, int] = {}

    def _validate_params(self):
        if self.zone is None:
            raise ValueError("leave_zone behavior requires a 'zone' parameter")

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
        bbox = person.bbox
        is_inside = self.zone.intersects_bbox(bbox)

        if tid not in self._track_inside:
            self._track_inside[tid] = False
            self._track_inside_counter[tid] = 0

        if is_inside:
            self._track_inside_counter[tid] += 1
            if self._track_inside_counter[tid] >= min_stay:
                self._track_inside[tid] = True
            return DetectionResult(
                track_id=tid, detected=False, confidence=0.0,
                metadata={"side": "none", "zone": self.zone.name, "inside": True},
            )
        else:
            if self._track_inside.get(tid, False):
                self._track_inside[tid] = False
                self._track_inside_counter[tid] = 0
                self._last_leave_frame[tid] = frame_idx
                return DetectionResult(
                    track_id=tid, detected=True, confidence=1.0,
                    metadata={"side": "outside", "zone": self.zone.name, "inside": False},
                )
            return DetectionResult(
                track_id=tid, detected=False, confidence=0.0,
                metadata={"side": "none", "zone": self.zone.name, "inside": False},
            )
