from detection.behavior_detector import register_behavior
from detection.behaviors.base import BaseBehavior, DetectionResult
from detection.event_manager import Event
from detection.zones.zone_definition import Zone


@register_behavior("leave_zone")
class LeaveZoneBehavior(BaseBehavior):
    name = "leave_zone"

    def __init__(self, params: dict, zones: list[Zone] | None = None):
        self.zones = zones or []
        super().__init__(params)
        self._track_inside: dict[int, bool] = {}
        self._track_inside_counter: dict[int, int] = {}
        self._track_inside_zones: dict[int, set[str]] = {}
        self._last_leave_frame: dict[int, int] = {}
        self._last_leave_zones: dict[int, list[str]] = {}
        self._missing_frames: dict[int, int] = {}

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

        current_track_ids = {p.track_id for p in people}
        max_missing = self.params.get("max_missing_frames", 15)

        for tid in list(self._track_inside.keys()):
            if self._track_inside.get(tid, False) and tid not in current_track_ids:
                self._missing_frames[tid] = self._missing_frames.get(tid, 0) + 1
                if self._missing_frames[tid] >= max_missing:
                    inside_zones = sorted(self._track_inside_zones.get(tid, set()))
                    event = Event(
                        track_id=tid,
                        start_frame=frame_idx,
                        end_frame=frame_idx,
                        start_time=timestamp,
                        end_time=timestamp,
                        max_confidence=1.0,
                        frames=[frame_idx],
                        hand_sides=["none"],
                        metadata={
                            "side": "outside",
                            "zone": inside_zones[0] if inside_zones else self.zones[0].name,
                            "triggered_zones": inside_zones,
                            "inside": False,
                            "cause": "missing_track",
                        },
                    )
                    event.behavior_name = self.name
                    new_events.append(event)
                    self._last_leave_frame[tid] = frame_idx
                    self._last_leave_zones[tid] = inside_zones
                    self._track_inside[tid] = False
                    self._track_inside_counter[tid] = 0
                    self._track_inside_zones.pop(tid, None)
                    del self._missing_frames[tid]

        for tid in current_track_ids:
            if tid in self._missing_frames:
                del self._missing_frames[tid]

        for person in people:
            result = self.detect_person(person, frame, frame_idx, timestamp)
            if result.detected:
                event = Event(
                    track_id=person.track_id,
                    start_frame=frame_idx,
                    end_frame=frame_idx,
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
        is_inside = any(z.intersects_bbox(bbox) for z in self.zones)
        inside_zones = {z.name for z in self.zones if z.intersects_bbox(bbox)}
        first_zone = self.zones[0].name

        if tid not in self._track_inside:
            self._track_inside[tid] = False
            self._track_inside_counter[tid] = 0
            self._track_inside_zones[tid] = set()

        if is_inside:
            self._track_inside_counter[tid] += 1
            self._track_inside_zones[tid].update(inside_zones)
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
            if self._track_inside.get(tid, False):
                left_zones = sorted(self._track_inside_zones.pop(tid, set()))
                self._track_inside[tid] = False
                self._track_inside_counter[tid] = 0
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
                    "triggered_zones": [],
                    "inside": False,
                },
            )