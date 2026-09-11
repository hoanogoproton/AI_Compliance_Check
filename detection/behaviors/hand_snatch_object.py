import math
from collections import deque
from dataclasses import dataclass, field

from detection.behavior_detector import register_behavior
from detection.behaviors.base import BaseBehavior, DetectionResult
from detection.config import (
    HAND_SNATCH_OBJECT_APPROACH_WINDOW,
    HAND_SNATCH_OBJECT_ARMED_WINDOW_SECONDS,
    HAND_SNATCH_OBJECT_BASELINE_RATIO,
    HAND_SNATCH_OBJECT_COOLDOWN_SECONDS,
    HAND_SNATCH_OBJECT_EMA_TAU,
    HAND_SNATCH_OBJECT_HOLD_STILL_SPEED,
    HAND_SNATCH_OBJECT_KEYPOINT_CONF_THRESHOLD,
    HAND_SNATCH_OBJECT_MIN_HOLD_SECONDS,
    HAND_SNATCH_OBJECT_MIN_JERK_DISPLACEMENT_RATIO,
    HAND_SNATCH_OBJECT_RESET_GAP_SECONDS,
    HAND_SNATCH_OBJECT_SPIKE_RATIO,
    HAND_SNATCH_OBJECT_STALE_TRACK_SECONDS,
    HAND_SNATCH_OBJECT_SUSTAIN_MAX_SECONDS,
    HAND_SNATCH_OBJECT_VELOCITY_RATIO,
)
from detection.pose_utils import compute_shoulder_width, get_keypoint
from detection.zones.zone_definition import Zone

# Số mẫu tốc độ tối đa lưu trong baseline lúc "đang giữ" (chống rò rỉ bộ nhớ)
_MAX_BASELINE_SAMPLES = 60


@dataclass
class _HandSnatchState:
    """Trạng thái máy trạng thái "giữ rồi giật" của một cổ tay (track + tay).

    Vị trí cổ tay được đo tương đối với tâm vai (triệt tiêu chuyển động
    của thân người) rồi làm mượt EMA. Máy trạng thái: IDLE → HOLDING (đang
    giữ yên trong zone) → JERK (cửa sổ chờ cú giật) → bắn sự kiện.
    """

    smoothed_rel: tuple[float, float] | None = None  # vị trí tương đối đã EMA (px)
    last_t: float | None = None
    state: str = "IDLE"  # IDLE / HOLDING / JERK
    hold_seconds: float = 0.0
    baseline: deque = field(default_factory=deque)  # tốc độ lúc giữ yên (sw/s)
    recent: deque = field(default_factory=deque)  # (t, tốc độ) gần đây (sw/s)
    jerk_start_rel: tuple[float, float] | None = None  # vị trí tương đối lúc stillness vỡ
    jerk_start_t: float | None = None
    jerk_still_baseline: float = 0.0  # baseline giữ yên tại thời điểm vào JERK
    jerk_recent_baseline: float = 0.0  # baseline tốc độ gần đây trước cú giật
    peak_velocity: float = 0.0  # tốc độ đỉnh trong JERK (sw/s)
    cooldown_until: float = 0.0  # mốc thời gian hết cooldown của tay này
    sustain_remaining: int = 0  # số frame còn phải sustain detected=True
    sustain_result: "DetectionResult | None" = None  # kết quả giữ nguyên metadata khi sustain
    fire_seq: int = 0  # thứ tự lần bắn (chọn sustain mới nhất khi cả hai tay cùng sustain)

    def reset_motion(self) -> None:
        """Xoá toàn bộ trạng thái (mất dấu tay / gap quá lâu)."""
        self.smoothed_rel = None
        self.last_t = None
        self.recent.clear()
        self.reset_pattern()

    def reset_pattern(self) -> None:
        """Xoá máy trạng thái hold/jerk; giữ lại EMA và lịch sử gần đây."""
        self.state = "IDLE"
        self.hold_seconds = 0.0
        self.baseline.clear()
        self.jerk_start_rel = None
        self.jerk_start_t = None
        self.jerk_still_baseline = 0.0
        self.jerk_recent_baseline = 0.0
        self.peak_velocity = 0.0


@register_behavior("hand_snatch_object")
class HandSnatchObjectBehavior(BaseBehavior):
    """Phát hiện hành vi giật/xé đứng đồ bên trong zone theo mẫu "giữ rồi giật".

    Pipeline không có bộ nhận diện vật thể nên "đang giữ đồ" được suy ra từ
    độ tĩnh của cổ tay **tương đối với tâm vai** (wrist - shoulder_center):
    thành phần chuyển động của cả thân người bị triệt tiêu nên người đi ngang
    qua zone với tay trong zone không thể gây báo động giả.

    Máy trạng thái theo từng (track_id, tay):
    - IDLE: vào zone và tốc độ tương đối ≤ ``hold_still_speed`` → HOLDING.
    - HOLDING: tích luỹ ``hold_seconds``; rời zone với tốc độ ≥ floor →
      ``snatch_out``; giữ đủ lâu rồi tốc độ vượt ngưỡng → JERK; giữ chưa đủ
      lâu mà di chuyển → về IDLE (chỉ là đi lại, tiêu diệt báo động giả).
    - JERK: trong ``armed_window_seconds`` bắn khi tốc độ ≥ max(floor,
      spike_ratio × baseline giữ, baseline_ratio × baseline gần đây) và dịch
      chuyển ròng ≥ ``min_jerk_displacement_ratio`` × vai. Hướng ra xa vai →
      ``snatch_in``, hướng về thân → ``snatch_in_pull``. Rời zone trong cửa
      sổ → ``snatch_out``.

    Sau khi bắn, kết quả ``detected=True`` được sustain thêm một số frame
    (= min(min_event_frames, sustain_max_seconds × fps)) để một cú giật chỉ
    kéo dài 1–3 frame vẫn tạo được Event khi event manager yêu cầu
    confirmation_frames/min_event_frames lớn.
    """

    name = "hand_snatch_object"

    def __init__(self, params: dict, zones: list[Zone] | None = None,
                 fps: float | None = None):
        super().__init__(params, zones=zones, fps=fps)
        self._hand_state: dict[tuple[int, str], _HandSnatchState] = {}
        self._last_track_frame: dict[int, int] = {}
        self._last_detections: dict[int, bool] = {}
        self._frame_triggered_zones: set[str] = set()

        self._conf_thresh = float(self.params.get(
            "keypoint_conf_threshold", HAND_SNATCH_OBJECT_KEYPOINT_CONF_THRESHOLD))
        self._hold_still_speed = float(self.params.get(
            "hold_still_speed", HAND_SNATCH_OBJECT_HOLD_STILL_SPEED))
        self._min_hold_seconds = float(self.params.get(
            "min_hold_seconds", HAND_SNATCH_OBJECT_MIN_HOLD_SECONDS))
        self._armed_window = float(self.params.get(
            "armed_window_seconds", HAND_SNATCH_OBJECT_ARMED_WINDOW_SECONDS))
        self._spike_ratio = float(self.params.get(
            "spike_ratio", HAND_SNATCH_OBJECT_SPIKE_RATIO))
        self._min_jerk_disp = float(self.params.get(
            "min_jerk_displacement_ratio", HAND_SNATCH_OBJECT_MIN_JERK_DISPLACEMENT_RATIO))
        self._tau = float(self.params.get("smoothing_tau", HAND_SNATCH_OBJECT_EMA_TAU))
        self._reset_gap = float(self.params.get(
            "reset_gap_seconds", HAND_SNATCH_OBJECT_RESET_GAP_SECONDS))
        self._cooldown = float(self.params.get(
            "cooldown_seconds", HAND_SNATCH_OBJECT_COOLDOWN_SECONDS))
        self._sustain_max_seconds = float(self.params.get(
            "sustain_max_seconds", HAND_SNATCH_OBJECT_SUSTAIN_MAX_SECONDS))
        self._base_ratio = float(self.params.get(
            "velocity_baseline_ratio", HAND_SNATCH_OBJECT_BASELINE_RATIO))
        self._approach_window = max(2, int(self.params.get(
            "approach_window", HAND_SNATCH_OBJECT_APPROACH_WINDOW)))
        # speed_floor (sw/s): snatch_velocity_ratio cũ tính theo px/frame/×vai,
        # nhân với fps để giữ nguyên ý nghĩa giá trị YAML cũ.
        vel_ratio = float(self.params.get(
            "snatch_velocity_ratio", HAND_SNATCH_OBJECT_VELOCITY_RATIO))
        self._speed_floor = max(0.0, vel_ratio * self.fps)
        # Sustain đủ dài để event manager kịp xác nhận một cú giật 1–3 frame.
        self._sustain_frames = max(1, min(
            int(self.event_manager.min_event_frames),
            math.ceil(self._sustain_max_seconds * self.fps)))

    def _validate_params(self):
        if len(self.zones) == 0:
            raise ValueError("hand_snatch_object behavior requires at least one zone")

    @property
    def current_triggered_zones(self) -> set[str]:
        return self._frame_triggered_zones

    def process_frame(self, people, frame, frame_idx, timestamp):
        self._frame_triggered_zones.clear()
        events = super().process_frame(people, frame, frame_idx, timestamp)
        self._prune_stale(frame_idx, current_tids={p.track_id for p in people})
        return events

    def _prune_stale(self, frame_idx: int, current_tids: set) -> None:
        """Dọn state của track đã biến mất quá lâu (tránh rò rỉ bộ nhớ)."""
        stale_after = max(1.0, HAND_SNATCH_OBJECT_STALE_TRACK_SECONDS * self.fps)
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

    def detect_person(self, person, frame, frame_idx, timestamp) -> DetectionResult:
        tid = person.track_id
        kpts = person.keypoints
        self._last_track_frame[tid] = frame_idx

        shoulder_width = compute_shoulder_width(kpts)
        if shoulder_width <= 0:
            for key in [k for k in self._hand_state if k[0] == tid]:
                self._hand_state[key].reset_motion()
            return self._finish_frame(tid, set())

        ls_x, ls_y, _ = get_keypoint(kpts, 5)
        rs_x, rs_y, _ = get_keypoint(kpts, 6)
        sc = ((ls_x + rs_x) / 2.0, (ls_y + rs_y) / 2.0)
        now = float(timestamp)

        fired_keys: set[tuple[int, str]] = set()
        fired_results: list[DetectionResult] = []

        for hand_name, wrist_idx in (("left", 9), ("right", 10)):
            key = (tid, hand_name)
            wx, wy, wc = get_keypoint(kpts, wrist_idx)
            if wc < self._conf_thresh:
                # Mất dấu tay: xoá toàn bộ trạng thái chuyển động của tay này
                # (sustain vẫn được giữ để event manager không bị đứt).
                if key in self._hand_state:
                    self._hand_state[key].reset_motion()
                continue

            in_zone = False
            for z in self.zones:
                if z.contains_point(wx, wy):
                    in_zone = True
                    self._frame_triggered_zones.add(z.name)

            st = self._hand_state.setdefault(key, _HandSnatchState())
            result = self._update_hand_state(st, key, in_zone, wx, wy, now, sc, shoulder_width)
            if result is not None:
                fired_keys.add(key)
                fired_results.append(result)

        return self._finish_frame(tid, fired_keys, fired_results)

    def _update_hand_state(
        self,
        st: _HandSnatchState,
        key: tuple[int, str],
        in_zone: bool,
        wx: float,
        wy: float,
        now: float,
        sc: tuple[float, float],
        shoulder_width: float,
    ) -> DetectionResult | None:
        """Cập nhật máy trạng thái của một tay; trả về DetectionResult nếu bắn."""
        tid, hand_name = key

        # Mất dấu quá lâu → reset toàn bộ (track re-ID / giật frame)
        if st.last_t is not None and (now - st.last_t) > self._reset_gap:
            st.reset_motion()

        # EMA theo thời gian của vị trí tương đối (wrist - tâm vai) → bất biến fps
        rel_raw = (wx - sc[0], wy - sc[1])
        prev_sr = st.smoothed_rel
        prev_t = st.last_t
        if prev_sr is None or prev_t is None:
            sr = rel_raw
            dt = 0.0
        else:
            dt = max(now - prev_t, 1e-3)
            alpha = 1.0 - math.exp(-dt / max(self._tau, 1e-3))
            sr = (
                prev_sr[0] + alpha * (rel_raw[0] - prev_sr[0]),
                prev_sr[1] + alpha * (rel_raw[1] - prev_sr[1]),
            )
        st.smoothed_rel = sr
        st.last_t = now

        if prev_sr is None:
            rel_speed = 0.0
            vel_vec = (0.0, 0.0)
        else:
            dx = sr[0] - prev_sr[0]
            dy = sr[1] - prev_sr[1]
            rel_speed = math.hypot(dx, dy) / dt / shoulder_width
            vel_vec = (dx, dy)

        # Lịch sử tốc độ gần đây (đóng góp vào baseline trước cú giật)
        st.recent.append((now, rel_speed))
        while len(st.recent) > self._approach_window:
            st.recent.popleft()

        result: DetectionResult | None = None

        if st.state == "IDLE":
            if in_zone and rel_speed <= self._hold_still_speed:
                st.state = "HOLDING"
                st.hold_seconds = 0.0
                st.baseline.clear()
                st.baseline.append(rel_speed)
        elif st.state == "HOLDING":
            if in_zone:
                if rel_speed <= self._hold_still_speed:
                    st.hold_seconds += dt
                    st.baseline.append(rel_speed)
                    while len(st.baseline) > _MAX_BASELINE_SAMPLES:
                        st.baseline.popleft()
                elif st.hold_seconds >= self._min_hold_seconds:
                    # Stillness vỡ sau khi giữ đủ lâu → arm JERK
                    st.state = "JERK"
                    st.jerk_start_rel = prev_sr
                    st.jerk_start_t = now
                    st.jerk_still_baseline = (
                        sum(st.baseline) / len(st.baseline) if st.baseline else 0.0
                    )
                    pre = [s for (t, s) in st.recent if t < now]
                    st.jerk_recent_baseline = sum(pre) / len(pre) if pre else 0.0
                    st.peak_velocity = rel_speed
                    result = self._evaluate_jerk(
                        st, hand_name, tid, rel_speed, vel_vec, sr, now, shoulder_width
                    )
                else:
                    # Giữ chưa đủ lâu mà đã di chuyển → chỉ là đi lại, về IDLE
                    st.reset_pattern()
            else:
                # Rời zone khi đang giữ: thoát nhanh → snatch_out, thoát chậm
                # (đi mang đồ đi) → không phải giật
                if (st.hold_seconds >= self._min_hold_seconds
                        and rel_speed >= self._speed_floor
                        and now >= st.cooldown_until):
                    result = self._make_fire(
                        hand_name, tid, "snatch_out", rel_speed, st,
                        displacement_ratio=rel_speed * dt,
                        now=now,
                    )
                st.reset_pattern()
        elif st.state == "JERK":
            if not in_zone:
                # Rời zone trong lúc giật → coi như giật ra ngoài
                if rel_speed >= self._speed_floor and now >= st.cooldown_until:
                    result = self._make_fire(
                        hand_name, tid, "snatch_out", rel_speed, st,
                        displacement_ratio=self._displacement_ratio(st, sr, shoulder_width),
                        now=now,
                    )
                st.reset_pattern()
            elif (now - (st.jerk_start_t or now)) > self._armed_window:
                # Hết cửa sổ mà chưa đạt gate → về IDLE
                st.reset_pattern()
            else:
                st.peak_velocity = max(st.peak_velocity, rel_speed)
                result = self._evaluate_jerk(
                    st, hand_name, tid, rel_speed, vel_vec, sr, now, shoulder_width
                )

        return result

    def _displacement_ratio(
        self, st: _HandSnatchState, sr: tuple[float, float], shoulder_width: float
    ) -> float:
        """Dịch chuyển ròng kể từ lúc stillness vỡ, chuẩn hoá theo chiều rộng vai."""
        if st.jerk_start_rel is None or shoulder_width <= 0:
            return 0.0
        return math.hypot(sr[0] - st.jerk_start_rel[0],
                          sr[1] - st.jerk_start_rel[1]) / shoulder_width

    def _evaluate_jerk(
        self,
        st: _HandSnatchState,
        hand_name: str,
        tid: int,
        rel_speed: float,
        vel_vec: tuple[float, float],
        sr: tuple[float, float],
        now: float,
        shoulder_width: float,
    ) -> DetectionResult | None:
        """Kiểm tra các gate của cú giật; trả về kết quả nếu đủ điều kiện bắn."""
        threshold = max(
            self._speed_floor,
            self._spike_ratio * st.jerk_still_baseline,
            self._base_ratio * st.jerk_recent_baseline,
        )
        disp_ratio = self._displacement_ratio(st, sr, shoulder_width)
        if rel_speed < threshold or disp_ratio < self._min_jerk_disp:
            return None
        if now < st.cooldown_until:
            return None
        r_vec = (sr[0], sr[1])
        dot = vel_vec[0] * r_vec[0] + vel_vec[1] * r_vec[1]
        snatch_type = "snatch_in" if dot > 0 else "snatch_in_pull"
        return self._make_fire(hand_name, tid, snatch_type, rel_speed, st,
                               displacement_ratio=disp_ratio, now=now)

    def _make_fire(
        self,
        hand_name: str,
        tid: int,
        snatch_type: str,
        rel_speed: float,
        st: _HandSnatchState,
        displacement_ratio: float,
        now: float,
    ) -> DetectionResult:
        """Tạo kết quả bắn, reset pattern của tay và bật sustain cho event manager."""
        st.fire_seq += 1
        st.peak_velocity = max(st.peak_velocity, rel_speed)
        st.sustain_remaining = max(0, self._sustain_frames - 1)
        st.sustain_result = DetectionResult(
            track_id=tid,
            detected=True,
            confidence=self._confidence(rel_speed),
            metadata={
                "hand": hand_name,
                "side": hand_name,
                "snatch_type": snatch_type,
                "exit_velocity": round(rel_speed, 4),
                "hold_duration": round(st.hold_seconds, 4),
                "spike_ratio": (round(rel_speed / st.jerk_still_baseline, 4)
                                if st.jerk_still_baseline > 1e-9 else None),
                "jerk_displacement_ratio": round(displacement_ratio, 4),
                "peak_velocity": round(st.peak_velocity, 4),
                "zone": self.zones[0].name,
                "triggered_zones": sorted(self._frame_triggered_zones),
                "fps": self.fps,
            },
        )
        st.reset_pattern()
        st.cooldown_until = now + self._cooldown
        return st.sustain_result

    def _confidence(self, rel_speed: float) -> float:
        if self._speed_floor <= 0:
            return 1.0
        return min(1.0, rel_speed / (2.0 * self._speed_floor))

    def _finish_frame(
        self,
        tid: int,
        fired_keys: set[tuple[int, str]],
        fired_results: list[DetectionResult] | None = None,
    ) -> DetectionResult:
        """Chọn kết quả của frame: ưu tiên lần bắn mới, sau đó đến sustain."""
        fired_results = fired_results or []
        if fired_results:
            best = max(fired_results, key=lambda r: r.confidence)
            self._last_detections[tid] = True
            return best

        best_seq = -1
        best_sustain: DetectionResult | None = None
        for key, st in self._hand_state.items():
            if key[0] != tid or key in fired_keys:
                continue
            if st.sustain_remaining > 0:
                st.sustain_remaining -= 1
                if st.sustain_result is not None and st.fire_seq > best_seq:
                    best_seq = st.fire_seq
                    best_sustain = st.sustain_result

        if best_sustain is not None:
            self._last_detections[tid] = True
            return best_sustain

        self._last_detections[tid] = False
        return self._empty_result(tid)