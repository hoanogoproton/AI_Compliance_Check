from collections import deque

import numpy as np

from detection.behavior_detector import register_behavior
from detection.behaviors.base import BaseBehavior, DetectionResult
from detection.config import (
    HAND_SNATCH_OBJECT_APPROACH_WINDOW,
    HAND_SNATCH_OBJECT_BASELINE_RATIO,
    HAND_SNATCH_OBJECT_KEYPOINT_CONF_THRESHOLD,
    HAND_SNATCH_OBJECT_MIN_GRASP_FRAMES,
    HAND_SNATCH_OBJECT_VELOCITY_RATIO,
)
from detection.pose_utils import compute_shoulder_width, get_keypoint
from detection.zones.zone_definition import Zone


@register_behavior("hand_snatch_object")
class HandSnatchObjectBehavior(BaseBehavior):
    name = "hand_snatch_object"

    def __init__(self, params: dict, zones: list[Zone] | None = None):
        super().__init__(params, zones=zones)
        self._wrist_history: dict[tuple[int, str], deque] = {}
        self._inside_state: dict[tuple[int, str], bool] = {}
        self._entry_frame: dict[tuple[int, str], int] = {}
        self._grasp_start: dict[tuple[int, str], int] = {}
        self._grasp_count: dict[tuple[int, str], int] = {}
        self._last_detections: dict[int, bool] = {}

    def _validate_params(self):
        if len(self.zones) == 0:
            raise ValueError("hand_snatch_object behavior requires at least one zone")

    def _compute_normalized_velocity(
        self, history: deque, shoulder_width: float
    ) -> tuple[float, list[float]]:
        if len(history) < 2:
            return 0.0, []
        recent = list(history)
        velocities = []
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i - 1][0]
            dy = recent[i][1] - recent[i - 1][1]
            df = recent[i][2] - recent[i - 1][2]
            if df <= 0:
                continue
            dist = np.sqrt(dx * dx + dy * dy)
            v = dist / df
            velocities.append(v)
        if not velocities:
            return 0.0, []
        current_v = velocities[-1]
        norm_v = current_v / shoulder_width if shoulder_width > 0 else 0.0
        return norm_v, velocities

    def _compute_baseline(
        self, velocities: list[float], window: int, shoulder_width: float
    ) -> float:
        recent = velocities[-window:] if len(velocities) > window else velocities
        non_zero = [v for v in recent if v > 0.01]
        if not non_zero:
            return 0.0
        mean_raw = sum(non_zero) / len(non_zero)
        return mean_raw / shoulder_width if shoulder_width > 0 else 0.0

    def detect_person(self, person, frame, frame_idx, timestamp) -> DetectionResult:
        tid = person.track_id
        kpts = person.keypoints

        min_grasp = int(self.params.get("min_grasp_frames", HAND_SNATCH_OBJECT_MIN_GRASP_FRAMES))
        vel_ratio = float(self.params.get("snatch_velocity_ratio", HAND_SNATCH_OBJECT_VELOCITY_RATIO))
        approach_win = int(self.params.get("approach_window", HAND_SNATCH_OBJECT_APPROACH_WINDOW))
        base_ratio = float(self.params.get("velocity_baseline_ratio", HAND_SNATCH_OBJECT_BASELINE_RATIO))
        conf_thresh = float(self.params.get("keypoint_conf_threshold", HAND_SNATCH_OBJECT_KEYPOINT_CONF_THRESHOLD))

        shoulder_width = compute_shoulder_width(kpts)
        if shoulder_width <= 0:
            self._last_detections[tid] = False
            return DetectionResult(
                track_id=tid, detected=False, confidence=0.0,
                metadata={"hand": "none", "side": "none", "zone": self.zones[0].name, "triggered_zones": []},
            )

        hands = [
            ("left", get_keypoint(kpts, 9)),
            ("right", get_keypoint(kpts, 10)),
        ]

        triggered_zones = set()

        for hand_name, wrist in hands:
            wx, wy, wc = wrist
            if wc < conf_thresh:
                continue

            in_zone = any(z.contains_point(wx, wy) for z in self.zones)
            for z in self.zones:
                if z.contains_point(wx, wy):
                    triggered_zones.add(z.name)

            key = (tid, hand_name)
            history = self._wrist_history.setdefault(key, deque())
            history.append((wx, wy, frame_idx))

            prev_inside = self._inside_state.get(key, False)

            entry_detected = in_zone and not prev_inside
            exit_detected = not in_zone and prev_inside

            if entry_detected:
                self._entry_frame[key] = frame_idx
                self._grasp_start[key] = frame_idx
                self._grasp_count[key] = 0

            if in_zone and prev_inside:
                self._grasp_count[key] = self._grasp_count.get(key, 0) + 1

            if exit_detected:
                grasp_frames = frame_idx - self._entry_frame.get(key, frame_idx)
                if grasp_frames >= min_grasp:
                    norm_v, velocities = self._compute_normalized_velocity(history, shoulder_width)
                    baseline_vels = velocities[:-1] if len(velocities) > 1 else []
                    baseline = self._compute_baseline(baseline_vels, approach_win, shoulder_width)
                    threshold = max(vel_ratio, baseline * base_ratio)
                    if norm_v > threshold:
                        peak_v = max(velocities) / shoulder_width if velocities else 0.0
                        self._last_detections[tid] = True
                        return DetectionResult(
                            track_id=tid,
                            detected=True,
                            confidence=min(1.0, norm_v / (vel_ratio * 2)),
                            metadata={
                                "hand": hand_name,
                                "side": hand_name,
                                "snatch_type": "snatch_out",
                                "exit_velocity": round(norm_v, 4),
                                "grasp_duration": grasp_frames,
                                "peak_velocity": round(peak_v, 4),
                                "zone": self.zones[0].name,
                                "triggered_zones": sorted(triggered_zones),
                            },
                        )

            if in_zone and prev_inside:
                norm_v, velocities = self._compute_normalized_velocity(history, shoulder_width)
                baseline_vels = velocities[:-1] if len(velocities) > 1 else []
                baseline = self._compute_baseline(baseline_vels, approach_win, shoulder_width)
                threshold = max(vel_ratio, baseline * base_ratio)
                if norm_v > threshold:
                    grasp_frames = frame_idx - self._entry_frame.get(key, frame_idx)
                    if grasp_frames >= min_grasp:
                        ls_x, ls_y, _ = get_keypoint(kpts, 5)
                        rs_x, rs_y, _ = get_keypoint(kpts, 6)
                        shoulder_center = np.array([(ls_x + rs_x) / 2, (ls_y + rs_y) / 2])
                        wrist_vec = np.array([wx, wy]) - shoulder_center
                        if len(history) >= 2:
                            prev = history[-2]
                            vel_vec = np.array([wx - prev[0], wy - prev[1]])
                            dot = np.dot(vel_vec, wrist_vec)
                            snatch_type = "snatch_in" if dot > 0 else "snatch_in_pull"
                        else:
                            snatch_type = "snatch_in"
                        peak_v = max(velocities) / shoulder_width if velocities else 0.0
                        self._last_detections[tid] = True
                        return DetectionResult(
                            track_id=tid,
                            detected=True,
                            confidence=min(1.0, norm_v / (vel_ratio * 2)),
                            metadata={
                                "hand": hand_name,
                                "side": hand_name,
                                "snatch_type": snatch_type,
                                "exit_velocity": round(norm_v, 4),
                                "grasp_duration": grasp_frames,
                                "peak_velocity": round(peak_v, 4),
                                "zone": self.zones[0].name,
                                "triggered_zones": sorted(triggered_zones),
                            },
                        )

            self._inside_state[key] = in_zone

        self._last_detections[tid] = False
        return DetectionResult(
            track_id=tid, detected=False, confidence=0.0,
            metadata={"hand": "none", "side": "none", "zone": self.zones[0].name, "triggered_zones": []},
        )