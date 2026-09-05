import math
from collections import deque
from dataclasses import dataclass, field

from detection.behavior_detector import register_behavior
from detection.behaviors.base import BaseBehavior, DetectionResult
from detection.config import (
    HAND_SHAKE_OBJECT_FREQ_MAX_HZ,
    HAND_SHAKE_OBJECT_FREQ_MIN_HZ,
    HAND_SHAKE_OBJECT_MIN_AMPLITUDE_RATIO,
    HAND_SHAKE_OBJECT_MIN_DISPLACEMENT_RATIO,
    HAND_SHAKE_OBJECT_MIN_REVERSALS,
    HAND_SHAKE_OBJECT_RESET_GAP_SECONDS,
    HAND_SHAKE_OBJECT_REVERSAL_DEADBAND_RATIO,
    HAND_SHAKE_OBJECT_SMOOTHING_TAU,
    HAND_SHAKE_OBJECT_STALE_TRACK_SECONDS,
    HAND_SHAKE_OBJECT_WINDOW_FRAMES,
    HAND_SHAKE_OBJECT_WINDOW_SECONDS,
)
from detection.pose_utils import compute_shoulder_width, get_keypoint
from detection.zones.zone_definition import Zone


@dataclass
class _HandMotionState:
    """Trạng thái chuyển động của một cổ tay (theo track + trái/phải).

    Dùng cho chế độ robust: vị trí EMA, pivot/cực trị từng trục để phát hiện
    đảo chiều kiểu Schmitt-trigger, mẫu vị trí trong cửa sổ và các mốc thời
    gian đảo chiều.
    """

    smoothed: tuple[float, float] | None = None
    last_t: float | None = None
    pivot: list[float] | None = None  # điểm mốc tham chiếu của từng trục
    direction: list[int] = field(default_factory=lambda: [0, 0])
    extreme: list = field(default_factory=lambda: [None, None])
    samples: deque = field(default_factory=deque)  # (t, x, y) trong cửa sổ
    reversals: deque = field(default_factory=deque)  # mốc thời gian đảo chiều

    def reset_motion(self) -> None:
        """Xoá trạng thái hướng/pivot; giữ lại lịch sử đảo chiều và mẫu."""
        self.smoothed = None
        self.last_t = None
        self.pivot = None
        self.direction = [0, 0]
        self.extreme = [None, None]


@register_behavior("hand_shake_object")
class HandShakeObjectBehavior(BaseBehavior):
    """Phát hiện hành vi lắc/xòe tay lặp lại (dao động) bên trong zone.

    Hai chế độ hoạt động:
    - ``legacy``: được chọn khi YAML cấu hình ``window_frames`` (tương thích
      ngược 100% với thuật toán gốc so dấu từng cặp frame).
    - ``robust`` (mặc định, hoặc khi có ``window_seconds``): tham số theo
      thời gian (giây), EMA smoothing, đảo chiều kiểu Schmitt-trigger có
      dead-band, gate biên độ peak-to-peak và lọc tần số dao động. Phù hợp
      với video fps thấp (ví dụ 7 fps).
    """

    name = "hand_shake_object"

    def __init__(self, params: dict, zones: list[Zone] | None = None,
                 fps: float | None = None):
        super().__init__(params, zones=zones, fps=fps)
        self._last_detections: dict[int, bool] = {}
        if "window_seconds" in self.params or "window_frames" not in self.params:
            self._mode = "robust"
            self._window_seconds = max(
                0.1, float(self.params.get("window_seconds", HAND_SHAKE_OBJECT_WINDOW_SECONDS))
            )
            self._hand_state: dict[tuple[int, str], _HandMotionState] = {}
            self._last_track_frame: dict[int, int] = {}
        else:
            self._mode = "legacy"
            self._window = int(self.params.get("window_frames", HAND_SHAKE_OBJECT_WINDOW_FRAMES))
            self._prev_wrist_pos: dict[tuple[int, str], tuple[float, float] | None] = {}
            self._wrist_x_dir: dict[tuple[int, str], int] = {}
            self._wrist_y_dir: dict[tuple[int, str], int] = {}
            self._wrist_reversals: dict[tuple[int, str], deque[int]] = {}

    def _validate_params(self):
        if len(self.zones) == 0:
            raise ValueError("hand_shake_object behavior requires at least one zone")

    def detect_person(self, person, frame, frame_idx, timestamp) -> DetectionResult:
        if self._mode == "robust":
            return self._detect_person_robust(person, frame, frame_idx, timestamp)
        return self._detect_person_legacy(person, frame, frame_idx, timestamp)

    def process_frame(self, people, frame, frame_idx, timestamp):
        events = super().process_frame(people, frame, frame_idx, timestamp)
        if self._mode == "robust":
            self._prune_stale(frame_idx, current_tids={p.track_id for p in people})
        return events

    def _prune_stale(self, frame_idx: int, current_tids: set) -> None:
        """Dọn state của track đã biến mất quá lâu (tránh rò rỉ bộ nhớ)."""
        stale_after = max(1.0, HAND_SHAKE_OBJECT_STALE_TRACK_SECONDS * self.fps)
        drop = [tid for tid, f in self._last_track_frame.items()
                if tid not in current_tids and (frame_idx - f) > stale_after]
        for tid in drop:
            self._last_track_frame.pop(tid, None)
            self._last_detections.pop(tid, None)
            for key in [k for k in self._hand_state if k[0] == tid]:
                self._hand_state.pop(key, None)

    def _empty_result(self, tid: int) -> DetectionResult:
        return DetectionResult(
            track_id=tid,
            detected=False,
            confidence=0.0,
            metadata={"hand": "none", "side": "none",
                      "zone": self.zones[0].name, "triggered_zones": []},
        )

    def _detect_person_legacy(self, person, frame, frame_idx, timestamp) -> DetectionResult:
        tid = person.track_id
        kpts = person.keypoints
        window = int(self.params.get("window_frames", HAND_SHAKE_OBJECT_WINDOW_FRAMES))
        min_rev = int(self.params.get("min_reversals", HAND_SHAKE_OBJECT_MIN_REVERSALS))
        ratio = float(self.params.get("min_displacement_ratio", HAND_SHAKE_OBJECT_MIN_DISPLACEMENT_RATIO))
        conf_thresh = float(self.params.get("keypoint_conf_threshold", 0.5))

        shoulder_width = compute_shoulder_width(kpts)
        if shoulder_width <= 0:
            self._last_detections[tid] = False
            return DetectionResult(track_id=tid, detected=False, confidence=0.0,
                                   metadata={"hand": "none", "side": "none", "zone": self.zones[0].name, "triggered_zones": []})
        min_disp = shoulder_width * ratio

        hands = [
            ("left", get_keypoint(kpts, 9)),
            ("right", get_keypoint(kpts, 10)),
        ]

        triggered_zones_all = set()
        best_result = None

        for hand_name, wrist in hands:
            wx, wy, wc = wrist
            if wc < conf_thresh:
                continue

            in_zone = any(z.contains_point(wx, wy) for z in self.zones)
            if not in_zone:
                continue

            for z in self.zones:
                if z.contains_point(wx, wy):
                    triggered_zones_all.add(z.name)

            key = (tid, hand_name)
            prev_wrist = self._prev_wrist_pos.get(key, None)
            wrist_revs = self._wrist_reversals.setdefault(key, deque())

            if prev_wrist is not None:
                dx = wx - prev_wrist[0]
                dy = wy - prev_wrist[1]

                x_dir = 1 if dx > 0 else (-1 if dx < 0 else 0)
                y_dir = 1 if dy > 0 else (-1 if dy < 0 else 0)

                prev_x = self._wrist_x_dir.get(key, 0)
                prev_y = self._wrist_y_dir.get(key, 0)

                if x_dir != 0 and prev_x != 0 and x_dir != prev_x and abs(dx) > min_disp:
                    wrist_revs.append(frame_idx)
                elif y_dir != 0 and prev_y != 0 and y_dir != prev_y and abs(dy) > min_disp:
                    wrist_revs.append(frame_idx)

                self._wrist_x_dir[key] = x_dir
                self._wrist_y_dir[key] = y_dir

            self._prev_wrist_pos[key] = (wx, wy)

            cutoff = frame_idx - window
            while wrist_revs and wrist_revs[0] <= cutoff:
                wrist_revs.popleft()

            count = len(wrist_revs)
            detected = count >= min_rev
            conf = min(1.0, count / float(min_rev))

            if detected and (best_result is None or conf > best_result.confidence):
                best_result = DetectionResult(
                    track_id=tid,
                    detected=True,
                    confidence=conf,
                    metadata={
                        "hand": hand_name,
                        "side": hand_name,
                        "wrist_reversals": count,
                        "zone": self.zones[0].name,
                        "triggered_zones": sorted(triggered_zones_all),
                    },
                )

        if best_result is not None:
            self._last_detections[tid] = True
            return best_result

        self._last_detections[tid] = False
        return DetectionResult(track_id=tid, detected=False, confidence=0.0,
                               metadata={"hand": "none", "side": "none", "zone": self.zones[0].name, "triggered_zones": []})

    def _detect_person_robust(self, person, frame, frame_idx, timestamp) -> DetectionResult:
        tid = person.track_id
        kpts = person.keypoints
        conf_thresh = float(self.params.get("keypoint_conf_threshold", 0.5))
        min_rev = int(self.params.get("min_reversals", HAND_SHAKE_OBJECT_MIN_REVERSALS))
        disp_ratio = float(self.params.get("min_displacement_ratio", HAND_SHAKE_OBJECT_MIN_DISPLACEMENT_RATIO))
        amp_ratio = float(self.params.get("min_amplitude_ratio", HAND_SHAKE_OBJECT_MIN_AMPLITUDE_RATIO))
        deadband_ratio = float(self.params.get("reversal_deadband_ratio", HAND_SHAKE_OBJECT_REVERSAL_DEADBAND_RATIO))
        tau = float(self.params.get("smoothing_tau", HAND_SHAKE_OBJECT_SMOOTHING_TAU))
        freq_min = float(self.params.get("min_frequency_hz", HAND_SHAKE_OBJECT_FREQ_MIN_HZ))
        freq_max = float(self.params.get("max_frequency_hz", HAND_SHAKE_OBJECT_FREQ_MAX_HZ))
        reset_gap = float(self.params.get("reset_gap_seconds", HAND_SHAKE_OBJECT_RESET_GAP_SECONDS))

        self._last_track_frame[tid] = frame_idx
        shoulder_width = compute_shoulder_width(kpts)
        if shoulder_width <= 0:
            self._last_detections[tid] = False
            return self._empty_result(tid)
        min_disp = shoulder_width * disp_ratio
        min_amp = shoulder_width * amp_ratio
        now = float(timestamp)

        triggered_zones_all: set[str] = set()
        best_result: DetectionResult | None = None

        for hand_name, (wx, wy, wc) in [
            ("left", get_keypoint(kpts, 9)),
            ("right", get_keypoint(kpts, 10)),
        ]:
            key = (tid, hand_name)
            if wc < conf_thresh:
                # Mất dấu tay: xoá cực trị để lần tái xuất không tạo đảo chiều ảo
                if key in self._hand_state:
                    self._hand_state[key].reset_motion()
                continue

            in_zone = False
            for z in self.zones:
                if z.contains_point(wx, wy):
                    in_zone = True
                    triggered_zones_all.add(z.name)
            if not in_zone:
                if key in self._hand_state:
                    self._hand_state[key].reset_motion()
                continue

            st = self._hand_state.setdefault(key, _HandMotionState())

            # Mất dấu quá lâu -> reset hướng/pivot (lịch sử đảo chiều tự hết hạn theo cửa sổ)
            if st.last_t is not None and (now - st.last_t) > reset_gap:
                st.reset_motion()

            # EMA smoothing theo thời gian -> bất biến theo fps
            if st.smoothed is None or st.last_t is None:
                sx, sy = wx, wy
            else:
                dt = max(now - st.last_t, 1e-3)
                alpha = 1.0 - math.exp(-dt / max(tau, 1e-3))
                sx = st.smoothed[0] + alpha * (wx - st.smoothed[0])
                sy = st.smoothed[1] + alpha * (wy - st.smoothed[1])
            st.smoothed = (sx, sy)
            st.last_t = now

            if st.pivot is None:
                st.pivot = [sx, sy]

            # Schmitt-trigger từng trục: chỉ đảo chiều khi rời cực trị quá dead-band
            for axis in (0, 1):
                p = (sx, sy)[axis]
                d = st.direction[axis]
                if d == 0:
                    delta = p - st.pivot[axis]
                    if abs(delta) > min_disp:
                        st.direction[axis] = 1 if delta > 0 else -1
                        st.extreme[axis] = p
                else:
                    travel = abs(st.extreme[axis] - st.pivot[axis])
                    dead_zone = max(min_disp, deadband_ratio * travel)
                    retrace = (st.extreme[axis] - p) * d
                    if retrace > dead_zone:
                        # Mỗi frame chỉ đếm tối đa 1 reversal (ưu tiên trục xét trước)
                        st.reversals.append(now)
                        st.pivot[axis] = p
                        st.extreme[axis] = p
                        st.direction[axis] = -d
                        break
                    if (p - st.extreme[axis]) * d > 0:
                        st.extreme[axis] = p

            # Cửa sổ trượt theo thời gian (giây), không phụ thuộc fps
            st.samples.append((now, sx, sy))
            cutoff = now - self._window_seconds
            while st.samples and st.samples[0][0] < cutoff:
                st.samples.popleft()
            while st.reversals and st.reversals[0] < cutoff:
                st.reversals.popleft()

            count = len(st.reversals)
            amp = 0.0
            if st.samples:
                xs = [s[1] for s in st.samples]
                ys = [s[2] for s in st.samples]
                amp = max(max(xs) - min(xs), max(ys) - min(ys))

            # Ước lượng tần số dao động: chu kỳ ≈ 2 reversal
            freq = None
            if count >= 2:
                span = st.reversals[-1] - st.reversals[0]
                if span > 1e-6:
                    freq = (count - 1) / 2.0 / span

            amp_ok = amp >= min_amp
            freq_ok = True
            if count >= 2:
                freq_ok = freq is not None and freq_min <= freq <= freq_max

            detected = count >= min_rev and amp_ok and freq_ok
            conf = 0.0
            if detected:
                conf = 0.6 * min(1.0, count / float(min_rev))
                conf += 0.4 * (min(1.0, amp / min_amp) if min_amp > 0 else 1.0)
                conf = min(1.0, conf)

            if detected and (best_result is None or conf > best_result.confidence):
                best_result = DetectionResult(
                    track_id=tid,
                    detected=True,
                    confidence=conf,
                    metadata={
                        "hand": hand_name,
                        "side": hand_name,
                        "wrist_reversals": count,
                        "amplitude_px": round(amp, 2),
                        "amplitude_ratio": round(amp / shoulder_width, 4),
                        "frequency_hz": round(freq, 3) if freq is not None else None,
                        "window_seconds": round(self._window_seconds, 3),
                        "fps": self.fps,
                        "zone": self.zones[0].name,
                        "triggered_zones": sorted(triggered_zones_all),
                    },
                )

        if best_result is not None:
            self._last_detections[tid] = True
            return best_result

        self._last_detections[tid] = False
        return self._empty_result(tid)