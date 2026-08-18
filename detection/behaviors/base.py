from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from detection.detector import TrackedPerson
from detection.event_manager import StatefulEventManager, Event


@dataclass
class DetectionResult:
    track_id: int
    detected: bool
    confidence: float
    metadata: dict = field(default_factory=dict)


class BaseBehavior(ABC):
    name: str = ""

    def __init__(self, params: dict):
        self.params = params
        self.event_manager = StatefulEventManager()
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
