from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from detection.detector import TrackedPerson
from detection.event_manager import StatefulEventManager, Event
from detection.config import CONFIRMATION_FRAMES, MAX_GAP_FRAMES, MIN_EVENT_FRAMES


@dataclass
class DetectionResult:
    track_id: int
    detected: bool
    confidence: float
    metadata: dict = field(default_factory=dict)


class BaseBehavior(ABC):
    name: str = ""

    def __init__(self, params: dict, zones: list | None = None, fps: float | None = None):
        self.params = params
        self.zones = zones or []
        # Video frame rate, used by fps-aware behaviors to convert time-based
        # parameters (seconds) into frame counts. Falls back to 30 when unknown.
        self.fps = float(fps) if fps is not None and fps > 0 else 30.0
        confirmation_frames = params.get("confirmation_frames", CONFIRMATION_FRAMES)
        max_gap_frames = params.get("max_gap_frames", MAX_GAP_FRAMES)
        min_event_frames = params.get("min_event_frames", MIN_EVENT_FRAMES)
        self.event_manager = StatefulEventManager(
            confirmation_frames=confirmation_frames,
            max_gap_frames=max_gap_frames,
            min_event_frames=min_event_frames,
        )
        self._validate_params()

    def _validate_params(self):
        pass

    @abstractmethod
    def detect_person(
        self,
        person: TrackedPerson,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
    ) -> DetectionResult:
        ...

    def process_frame(
        self,
        people: list[TrackedPerson],
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
    ) -> list[Event]:
        detections: dict[int, tuple[bool, float, str]] = {}
        for person in people:
            result = self.detect_person(person, frame, frame_idx, timestamp)
            side = result.metadata.get("side", "none")
            detections[person.track_id] = (result.detected, result.confidence, side)

        new_events = self.event_manager.update(frame_idx, timestamp, detections)
        for ev in new_events:
            ev.behavior_name = self.name
        return new_events
