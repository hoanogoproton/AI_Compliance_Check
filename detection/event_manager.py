from dataclasses import dataclass, field

from detection.config import CONFIRMATION_FRAMES, MAX_GAP_FRAMES, MIN_EVENT_FRAMES


@dataclass
class Event:
    track_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    max_confidence: float
    frames: list[int] = field(default_factory=list)
    hand_sides: list[str] = field(default_factory=list)
    behavior_name: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class _TrackState:
    state: str = "IDLE"
    confirm_count: int = 0
    gap_count: int = 0
    event_frames: list[int] = field(default_factory=list)
    hand_sides: list[str] = field(default_factory=list)
    max_conf: float = 0.0
    start_frame: int = 0
    start_time: float = 0.0
    cooldown_count: int = 0


class StatefulEventManager:
    def __init__(self, confirmation_frames=None, max_gap_frames=None, min_event_frames=None):
        self.confirmation_frames = confirmation_frames if confirmation_frames is not None else CONFIRMATION_FRAMES
        self.max_gap_frames = max_gap_frames if max_gap_frames is not None else MAX_GAP_FRAMES
        self.min_event_frames = min_event_frames if min_event_frames is not None else MIN_EVENT_FRAMES
        self._tracks: dict[int, _TrackState] = {}
        self._event_counter = 0
        self._completed_events: list[Event] = []

    def update(
        self, frame_idx: int, timestamp: float, detections: dict[int, tuple[bool, float, str]]
    ) -> list[Event]:
        new_completed = []
        all_track_ids = set(self._tracks.keys()) | set(detections.keys())
        for tid in all_track_ids:
            detected = detections.get(tid, (False, 0.0, "none"))
            is_detected, conf, side = detected
            if tid not in self._tracks:
                self._tracks[tid] = _TrackState()
            ts = self._tracks[tid]

            if ts.state == "IDLE":
                if is_detected:
                    ts.state = "CONFIRMING"
                    ts.confirm_count = 1
                    ts.start_frame = frame_idx
                    ts.start_time = timestamp
                    ts.event_frames = [frame_idx]
                    ts.hand_sides = [side]
                    ts.max_conf = conf
                    ts.gap_count = 0

            elif ts.state == "CONFIRMING":
                if is_detected:
                    ts.confirm_count += 1
                    ts.event_frames.append(frame_idx)
                    ts.hand_sides.append(side)
                    if conf > ts.max_conf:
                        ts.max_conf = conf
                    if ts.confirm_count >= self.confirmation_frames:
                        ts.state = "ACTIVE"
                        ts.gap_count = 0
                else:
                    ts.state = "IDLE"
                    ts.confirm_count = 0
                    ts.event_frames = []
                    ts.hand_sides = []
                    ts.max_conf = 0.0

            elif ts.state == "ACTIVE":
                if is_detected:
                    ts.event_frames.append(frame_idx)
                    ts.hand_sides.append(side)
                    if conf > ts.max_conf:
                        ts.max_conf = conf
                    ts.gap_count = 0
                else:
                    ts.gap_count += 1
                    if ts.gap_count > self.max_gap_frames:
                        if len(ts.event_frames) >= self.min_event_frames:
                            event = Event(
                                track_id=tid,
                                start_frame=ts.start_frame,
                                end_frame=ts.event_frames[-1],
                                start_time=ts.start_time,
                                end_time=timestamp,
                                max_confidence=ts.max_conf,
                                frames=ts.event_frames[:],
                                hand_sides=ts.hand_sides[:],
                            )
                            self._completed_events.append(event)
                            new_completed.append(event)
                        ts.state = "COOLDOWN"
                        ts.cooldown_count = 0

            elif ts.state == "COOLDOWN":
                ts.cooldown_count += 1
                if ts.cooldown_count >= self.confirmation_frames:
                    ts.state = "IDLE"
                    ts.confirm_count = 0
                    ts.gap_count = 0
                    ts.event_frames = []
                    ts.hand_sides = []
                    ts.max_conf = 0.0

        return new_completed

    def finalize(self) -> list[Event]:
        completed = []
        for tid, ts in list(self._tracks.items()):
            if ts.state == "ACTIVE" and len(ts.event_frames) >= self.min_event_frames:
                event = Event(
                    track_id=tid,
                    start_frame=ts.start_frame,
                    end_frame=ts.event_frames[-1],
                    start_time=ts.start_time,
                    end_time=0.0,
                    max_confidence=ts.max_conf,
                    frames=ts.event_frames[:],
                    hand_sides=ts.hand_sides[:],
                )
                self._completed_events.append(event)
                completed.append(event)
            ts.state = "IDLE"
        return completed