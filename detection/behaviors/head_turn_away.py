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

    def _get_state(self, track_id: int) -> dict:
        return self._states.setdefault(
            track_id,
            {
                "face_state": "uncertain",
                "face_visible_count": 0,
                "face_hidden_count": 0,
                "hidden_start_frame": None,
            },
        )

    def detect_person(self, person, frame, frame_idx, timestamp) -> DetectionResult:
        tid = person.track_id
        state = self._get_state(tid)
        dq = self._turn_frames.setdefault(tid, deque())

        min_face_visible = max(
            1,
            int(
                self.params.get(
                    "min_face_visible_frames",
                    HEAD_TURN_AWAY_MIN_FACE_VISIBLE_FRAMES,
                )
            ),
        )
        min_face_hidden = max(
            1,
            int(
                self.params.get(
                    "min_face_hidden_frames",
                    HEAD_TURN_AWAY_MIN_FACE_HIDDEN_FRAMES,
                )
            ),
        )
        window = max(
            1,
            int(
                self.params.get(
                    "window_frames",
                    HEAD_TURN_AWAY_WINDOW_FRAMES,
                )
            ),
        )
        max_turns = max(
            0,
            int(
                self.params.get(
                    "max_turns",
                    HEAD_TURN_AWAY_MAX_TURNS,
                )
            ),
        )
        nose_threshold = float(
            self.params.get(
                "nose_confidence_threshold",
                HEAD_TURN_AWAY_NOSE_CONFIDENCE_THRESHOLD,
            )
        )

        nose_conf = float(person.keypoints[0, 2])

        body_ok = is_body_trackable(
            person.keypoints,
            min_conf=float(
                self.params.get(
                    "body_confidence_threshold",
                    HEAD_TURN_AWAY_BODY_CONFIDENCE_THRESHOLD,
                )
            ),
            min_visible=int(
                self.params.get(
                    "body_min_visible_keypoints",
                    HEAD_TURN_AWAY_BODY_MIN_VISIBLE_KEYPOINTS,
                )
            ),
        )

        face_seen = nose_conf >= nose_threshold
        transition_event = None

        # Không có đủ bằng chứng về body:
        # không được suy luận face hidden hay head turn.
        if not body_ok:
            state["face_state"] = "uncertain"
            state["face_visible_count"] = 0
            state["face_hidden_count"] = 0
            state["hidden_start_frame"] = None

        elif state["face_state"] == "uncertain":
            if face_seen:
                state["face_visible_count"] += 1
                state["face_hidden_count"] = 0

                if state["face_visible_count"] >= min_face_visible:
                    state["face_state"] = "visible"
            else:
                state["face_visible_count"] = 0
                state["face_hidden_count"] = 0

        elif state["face_state"] == "visible":
            if face_seen:
                state["face_visible_count"] += 1
                state["face_hidden_count"] = 0
                state["hidden_start_frame"] = None

            else:
                state["face_visible_count"] = 0

                if state["face_hidden_count"] == 0:
                    state["hidden_start_frame"] = frame_idx

                state["face_hidden_count"] += 1

                if state["face_hidden_count"] >= min_face_hidden:
                    state["face_state"] = "hidden"

                    # Lưu frame bắt đầu hidden thay vì frame xác nhận.
                    event_frame = state["hidden_start_frame"] or frame_idx
                    dq.append(event_frame)

                    transition_event = "head_turn_away"

        elif state["face_state"] == "hidden":
            if face_seen:
                state["face_visible_count"] += 1
                state["face_hidden_count"] = 0
                state["hidden_start_frame"] = None

                if state["face_visible_count"] >= min_face_visible:
                    state["face_state"] = "visible"
            else:
                state["face_visible_count"] = 0
                state["face_hidden_count"] += 1

        # Chỉ giữ event trong `window` frame gần nhất.
        # Ví dụ window=90, frame=100 => giữ frame 11..100.
        cutoff = frame_idx - window

        while dq and dq[0] <= cutoff:
            dq.popleft()

        turns = len(dq)

        # max_turns=2 => turns >= 3 thì detected=True.
        detected = turns > max_turns

        # Confidence phản ánh mức độ vượt ngưỡng.
        confidence = min(
            1.0,
            turns / float(max_turns + 1),
        )

        return DetectionResult(
            track_id=tid,
            detected=detected,
            confidence=confidence if detected else 0.0,
            metadata={
                "event": transition_event,
                "face_state": state["face_state"],
                "nose_conf": round(nose_conf, 3),
                "body_trackable": body_ok,
                "face_visible_frames": state["face_visible_count"],
                "face_hidden_frames": state["face_hidden_count"],
                "hidden_start_frame": state["hidden_start_frame"],
                "turns": turns,
                "window_frames": window,
                "max_turns": max_turns,
            },
        )