from detection.behavior_detector import register_behavior
from detection.behaviors.base import BaseBehavior, DetectionResult
from detection.event_manager import Event
from detection.zones.zone_definition import Zone


@register_behavior("danger_zone")
class DangerZoneBehavior(BaseBehavior):
    name = "danger_zone"

    def __init__(self, params: dict, zones: list[Zone] | None = None):
        super().__init__(params, zones=zones)
        self._in_zone_counter: dict[int, int] = {}
        self._outside_counter: dict[int, int] = {}
        self._track_in_zone: dict[int, bool] = {}
        self._alerted: dict[int, bool] = {}
        self._inside_zones: dict[int, set[str]] = {}
        self._last_alert_frame: dict[int, int] = {}
        self._alert_zones: dict[int, list[str]] = {}
        self._frame_alert_zones: set[str] = set()

    def _validate_params(self):
        if len(self.zones) == 0:
            raise ValueError("danger_zone behavior requires at least one zone")

    def is_person_in_alert(self, track_id: int, frame_idx: int) -> bool:
        if self._track_in_zone.get(track_id, False):
            return True
        alert_frame = self._last_alert_frame.get(track_id)
        if alert_frame is None:
            return False
        flash_frames = self.params.get("alert_flash_frames", 20)
        return frame_idx - alert_frame <= flash_frames

    @property
    def current_triggered_zones(self) -> set[str]:
        return self._frame_alert_zones

    def process_frame(self, people, frame, frame_idx, timestamp) -> list[Event]:
        self._frame_alert_zones.clear()
        new_events = []

        for person in people:
            result = self.detect_person(person, frame, frame_idx, timestamp)
            if result.detected:
                tid = person.track_id
                alert_zones = result.metadata.get("triggered_zones", [])
                self._alerted[tid] = True
                self._last_alert_frame[tid] = frame_idx
                self._alert_zones[tid] = alert_zones
                event = Event(
                    track_id=tid,
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

        for tid in set(self._track_in_zone) | set(self._last_alert_frame):
            if self.is_person_in_alert(tid, frame_idx):
                self._frame_alert_zones.update(self._inside_zones.get(tid, set()))
                self._frame_alert_zones.update(self._alert_zones.get(tid, []))
        return new_events

    def detect_person(self, person, frame, frame_idx, timestamp) -> DetectionResult:
        tid = person.track_id
        min_enter = self.params.get("min_enter_frames", 3)
        min_exit = self.params.get("min_exit_frames", 5)
        bbox = person.bbox
        is_inside = any(z.contains_bbox_center(bbox) for z in self.zones)
        inside_zones = {z.name for z in self.zones if z.contains_bbox_center(bbox)}
        first_zone = self.zones[0].name

        if tid not in self._track_in_zone:
            self._track_in_zone[tid] = False
            self._in_zone_counter[tid] = 0
            self._outside_counter[tid] = 0
            self._alerted[tid] = False
            self._inside_zones[tid] = set()

        if is_inside:
            self._outside_counter[tid] = 0
            self._in_zone_counter[tid] += 1
            self._track_in_zone[tid] = True
            self._inside_zones.setdefault(tid, set()).update(inside_zones)
            if not self._alerted.get(tid, False) and self._in_zone_counter[tid] >= min_enter:
                triggered = sorted(self._inside_zones[tid])
                return DetectionResult(
                    track_id=tid, detected=True, confidence=1.0,
                    metadata={
                        "side": "inside",
                        "zone": triggered[0] if triggered else first_zone,
                        "triggered_zones": triggered,
                        "inside": True,
                    },
                )
            return DetectionResult(
                track_id=tid, detected=False, confidence=0.0,
                metadata={
                    "side": "none",
                    "zone": first_zone,
                    "triggered_zones": sorted(self._inside_zones[tid]),
                    "inside": True,
                },
            )
        else:
            self._in_zone_counter[tid] = 0
            self._track_in_zone[tid] = False
            self._outside_counter[tid] = self._outside_counter.get(tid, 0) + 1
            if self._outside_counter[tid] >= min_exit:
                self._alerted[tid] = False
                self._inside_zones.pop(tid, None)
            return DetectionResult(
                track_id=tid, detected=False, confidence=0.0,
                metadata={
                    "side": "none",
                    "zone": first_zone,
                    "triggered_zones": [],
                    "inside": False,
                },
            )