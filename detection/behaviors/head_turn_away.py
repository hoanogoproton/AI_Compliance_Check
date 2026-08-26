from collections import deque

from detection.behavior_detector import register_behavior
from detection.behaviors.base import BaseBehavior, DetectionResult
from detection.config import (
    HEAD_TURN_AWAY_BODY_CONFIDENCE_THRESHOLD,
    HEAD_TURN_AWAY_BODY_MIN_VISIBLE_KEYPOINTS,
    HEAD_TURN_AWAY_WINDOW_FRAMES,
    HEAD_TURN_AWAY_MAX_TURNS,
)
from detection.pose_utils import is_body_trackable


@register_behavior("head_turn_away")
class HeadTurnAwayBehavior(BaseBehavior):
    """
    Phát hiện chuyển trạng thái:

        face_toward_camera -> face_away_from_camera

    Lưu ý:
    - "away" nghĩa là mặt quay ra xa góc camera hiện tại.
    - Không đồng nghĩa tuyệt đối với "mất tập trung".
    """

    name = "head_turn_away"

    # Chỉ đúng nếu person.keypoints dùng thứ tự COCO-17 tiêu chuẩn.
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4

    FACE_KEYPOINT_INDICES = (
        NOSE,
        LEFT_EYE,
        RIGHT_EYE,
        LEFT_EAR,
        RIGHT_EAR,
    )

    def __init__(self, params: dict, **kwargs):
        super().__init__(params, **kwargs)

        self._states: dict[int, dict] = {}
        self._turn_frames: dict[int, deque[int]] = {}

    def _get_state(self, track_id: int) -> dict:
        return self._states.setdefault(
            track_id,
            {
                # Stable state: uncertain / toward / away
                "face_state": "uncertain",

                # Candidate state để debounce.
                "candidate_state": None,
                "candidate_count": 0,
                "candidate_start_frame": None,

                # Thông tin event away hiện tại.
                "away_start_frame": None,
            },
        )

    def _get_face_evidence(
        self,
        keypoints,
        confidence_threshold: float,
    ) -> dict:
        """
        Tổng hợp bằng chứng facial keypoint.

        Return:
            {
                "nose_conf": float,
                "face_confidences": {...},
                "visible_face_keypoints": int,
            }
        """

        face_confidences = {}

        for index in self.FACE_KEYPOINT_INDICES:
            try:
                confidence = float(keypoints[index, 2])
            except (IndexError, TypeError, ValueError):
                confidence = 0.0

            face_confidences[index] = confidence

        visible_face_keypoints = sum(
            confidence >= confidence_threshold
            for confidence in face_confidences.values()
        )

        return {
            "nose_conf": face_confidences[self.NOSE],
            "face_confidences": face_confidences,
            "visible_face_keypoints": visible_face_keypoints,
        }

    def _get_raw_face_state(
        self,
        body_ok: bool,
        visible_face_keypoints: int,
        min_toward_keypoints: int,
        max_away_keypoints: int,
    ) -> str:
        """
        Xác định state thô, chưa debounce.

        toward:
            Có đủ facial keypoint để tin rằng mặt đang hướng
            một phần hoặc toàn phần về camera.

        away:
            Body vẫn rõ nhưng hầu như không thấy facial keypoint.

        uncertain:
            Không đủ dữ liệu hoặc nằm trong vùng mơ hồ.
        """

        if not body_ok:
            return "uncertain"

        if visible_face_keypoints >= min_toward_keypoints:
            return "toward"

        if visible_face_keypoints <= max_away_keypoints:
            return "away"

        return "uncertain"

    def _reset_candidate(self, state: dict) -> None:
        state["candidate_state"] = None
        state["candidate_count"] = 0
        state["candidate_start_frame"] = None

    def detect_person(self, person, frame, frame_idx, timestamp) -> DetectionResult:
        tid = person.track_id
        state = self._get_state(tid)
        dq = self._turn_frames.setdefault(tid, deque())

        # -----------------------------
        # Đọc cấu hình
        # -----------------------------
        face_keypoint_threshold = float(
            self.params.get(
                "face_keypoint_confidence_threshold",
                0.45,
            )
        )

        min_toward_keypoints = max(
            1,
            int(
                self.params.get(
                    "min_face_keypoints_toward",
                    3,
                )
            ),
        )

        max_away_keypoints = max(
            0,
            int(
                self.params.get(
                    "max_face_keypoints_away",
                    1,
                )
            ),
        )

        min_toward_frames = max(
            1,
            int(
                self.params.get(
                    "min_toward_frames",
                    4,
                )
            ),
        )

        min_away_frames = max(
            1,
            int(
                self.params.get(
                    "min_away_frames",
                    8,
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

        # -----------------------------
        # Kiểm tra body trackable
        # -----------------------------
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

        # -----------------------------
        # Tổng hợp bằng chứng mặt
        # -----------------------------
        face_evidence = self._get_face_evidence(
            person.keypoints,
            confidence_threshold=face_keypoint_threshold,
        )

        raw_face_state = self._get_raw_face_state(
            body_ok=body_ok,
            visible_face_keypoints=face_evidence["visible_face_keypoints"],
            min_toward_keypoints=min_toward_keypoints,
            max_away_keypoints=max_away_keypoints,
        )

        transition_event = None

        # Không suy luận khi body không đủ tin cậy.
        if not body_ok:
            state["face_state"] = "uncertain"
            state["away_start_frame"] = None
            self._reset_candidate(state)

        # Vùng mơ hồ: không tạo transition mới.
        elif raw_face_state == "uncertain":
            self._reset_candidate(state)

        # Raw state trùng stable state: reset candidate.
        elif raw_face_state == state["face_state"]:
            self._reset_candidate(state)

        # Raw state khác stable state: debounce.
        else:
            if raw_face_state == state["candidate_state"]:
                state["candidate_count"] += 1
            else:
                state["candidate_state"] = raw_face_state
                state["candidate_count"] = 1
                state["candidate_start_frame"] = frame_idx

            needed_frames = (
                min_toward_frames
                if raw_face_state == "toward"
                else min_away_frames
            )

            # Chỉ commit state sau khi raw state ổn định đủ lâu.
            if state["candidate_count"] >= needed_frames:
                previous_state = state["face_state"]

                stable_start_frame = state["candidate_start_frame"]
                state["face_state"] = raw_face_state
                self._reset_candidate(state)

                # toward -> away: ghi nhận một lần quay ra xa camera.
                if previous_state == "toward" and raw_face_state == "away":
                    away_start_frame = (
                        stable_start_frame
                        if stable_start_frame is not None
                        else frame_idx
                    )

                    state["away_start_frame"] = away_start_frame
                    dq.append(away_start_frame)

                    transition_event = "face_turned_away_from_camera"

                # away -> toward: người quay mặt lại về camera.
                elif previous_state == "away" and raw_face_state == "toward":
                    state["away_start_frame"] = None
                    transition_event = "face_returned_toward_camera"

        # -----------------------------
        # Sliding window event count
        # -----------------------------
        cutoff = frame_idx - window

        while dq and dq[0] <= cutoff:
            dq.popleft()

        turns = len(dq)
        detected = turns > max_turns

        # Mức độ vượt ngưỡng.
        # max_turns=2:
        # - turns=3 => 0.5
        # - turns=4 => 1.0
        # - turns>=5 => 1.0
        excess_turns = max(0, turns - max_turns)

        confidence = min(
            1.0,
            excess_turns / 2.0,
        )

        return DetectionResult(
            track_id=tid,
            detected=detected,
            confidence=confidence if detected else 0.0,
            metadata={
                "event": transition_event,
                "face_state": state["face_state"],
                "raw_face_state": raw_face_state,
                "body_trackable": body_ok,
                "nose_conf": round(face_evidence["nose_conf"], 3),
                "visible_face_keypoints": face_evidence[
                    "visible_face_keypoints"
                ],
                "face_confidences": {
                    str(index): round(confidence, 3)
                    for index, confidence in face_evidence[
                        "face_confidences"
                    ].items()
                },
                "candidate_state": state["candidate_state"],
                "candidate_count": state["candidate_count"],
                "away_start_frame": state["away_start_frame"],
                "turns": turns,
                "window_frames": window,
                "max_turns": max_turns,
            },
        )