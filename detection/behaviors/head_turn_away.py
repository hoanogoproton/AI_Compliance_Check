from collections import deque

from detection.behavior_detector import register_behavior
from detection.behaviors.base import BaseBehavior, DetectionResult
from detection.config import (
    HEAD_TURN_AWAY_MIN_FACE_VISIBLE_FRAMES,
    HEAD_TURN_AWAY_MIN_FACE_HIDDEN_FRAMES,
    HEAD_TURN_AWAY_NOSE_CONFIDENCE_THRESHOLD,
    HEAD_TURN_AWAY_BODY_CONFIDENCE_THRESHOLD,
    HEAD_TURN_AWAY_BODY_MIN_VISIBLE_KEYPOINTS,
    HEAD_TURN_AWAY_WINDOW_FRAMES,
    HEAD_TURN_AWAY_MAX_TURNS,
)
from detection.pose_utils import is_body_trackable


@register_behavior("head_turn_away")
class HeadTurnAwayBehavior(BaseBehavior):
    name = "head_turn_away"

    def __init__(self, params: dict, **kwargs):
        super().__init__(params, **kwargs)
        self._states: dict[int, dict] = {}
        self._turn_frames: dict[int, deque[int]] = {}

    def detect_person(self, person, frame, frame_idx, timestamp) -> DetectionResult:
        tid = person.track_id
        state = self._states.setdefault(tid, {
            "face_state": "uncertain",
            "face_visible_count": 0,
            "face_hidden_count": 0,
        })
        dq = self._turn_frames.setdefault(tid, deque())

        min_face_visible = int(
            self.params.get("min_face_visible_frames", HEAD_TURN_AWAY_MIN_FACE_VISIBLE_FRAMES)
        )
        min_face_hidden = int(
            self.params.get("min_face_hidden_frames", HEAD_TURN_AWAY_MIN_FACE_HIDDEN_FRAMES)
        )
        window = int(
            self.params.get("window_frames", HEAD_TURN_AWAY_WINDOW_FRAMES)
        )
        max_turns = int(
            self.params.get("max_turns", HEAD_TURN_AWAY_MAX_TURNS)
        )
        nose_threshold = float(
            self.params.get("nose_confidence_threshold", HEAD_TURN_AWAY_NOSE_CONFIDENCE_THRESHOLD)
        )

        nose_conf = float(person.keypoints[0, 2])
        body_ok = is_body_trackable(
            person.keypoints,
            min_conf=float(self.params.get("body_confidence_threshold", HEAD_TURN_AWAY_BODY_CONFIDENCE_THRESHOLD)),
            min_visible=int(self.params.get("body_min_visible_keypoints", HEAD_TURN_AWAY_BODY_MIN_VISIBLE_KEYPOINTS)),
        )

        face_seen = nose_conf >= nose_threshold
        transition_event = None

        if state["face_state"] == "uncertain":
            if face_seen and body_ok:
                state["face_visible_count"] += 1
                state["face_hidden_count"] = 0
                if state["face_visible_count"] >= min_face_visible:
                    state["face_state"] = "visible"
            else:
                state["face_visible_count"] = 0

        elif state["face_state"] == "visible":
            if face_seen and body_ok:
                state["face_visible_count"] += 1
                state["face_hidden_count"] = 0
            elif not face_seen and body_ok:
                state["face_visible_count"] = 0
                state["face_hidden_count"] += 1
                if state["face_hidden_count"] >= min_face_hidden:
                    state["face_state"] = "hidden"
                    dq.append(frame_idx)
                    transition_event = "head_turn_away"
            else:
                state["face_visible_count"] = 0
                state["face_hidden_count"] = 0

        elif state["face_state"] == "hidden":
            if face_seen and body_ok:
                state["face_visible_count"] += 1
                state["face_hidden_count"] = 0
                if state["face_visible_count"] >= min_face_visible:
                    state["face_state"] = "visible"
            elif not face_seen and body_ok:
                state["face_hidden_count"] += 1
            else:
                state["face_visible_count"] = 0
                state["face_hidden_count"] = 0

        cutoff = frame_idx - window
        while dq and dq[0] <= cutoff:
            dq.popleft()

        turns = len(dq)
        detected = turns > max_turns
        confidence = min(1.0, turns / float(max_turns + 1)) if detected else 0.0

        return DetectionResult(
            track_id=tid,
            detected=detected,
            confidence=confidence,
            metadata={
                "event": transition_event,
                "face_state": state["face_state"],
                "nose_conf": nose_conf,
                "body_trackable": body_ok,
                "face_visible_frames": state["face_visible_count"],
                "face_hidden_frames": state["face_hidden_count"],
                "turns": turns,
            },
        )