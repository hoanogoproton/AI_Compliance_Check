import numpy as np

from detection.behaviors.hand_snatch_object import HandSnatchObjectBehavior
from detection.detector import TrackedPerson
from detection.zones.zone_definition import Zone

# Fixture: shoulders (200,100) / (260,100) -> chiều rộng vai 60px, tâm vai (230,100).
# Zone mặc định 100×80 quanh tâm vai, đủ rộng để thực hiện cú giật bên trong zone.


def _make_zone(points=None):
    if points is None:
        points = [[200, 100], [300, 100], [300, 180], [200, 180]]
    return Zone(name="TestZone", label="TestZone", points=points)


def _make_kpts(wrist_l_x=250.0, wrist_l_y=140.0, wrist_l_conf=0.9,
               wrist_r_x=None, wrist_r_y=140.0, wrist_r_conf=0.9,
               shoulder_l=(200, 100, 0.9), shoulder_r=(260, 100, 0.9)):
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[5] = shoulder_l
    kpts[6] = shoulder_r
    kpts[9] = [wrist_l_x, wrist_l_y, wrist_l_conf]
    if wrist_r_x is not None:
        kpts[10] = [wrist_r_x, wrist_r_y, wrist_r_conf]
    return kpts


def _make_person(track_id, **kpt_kwargs):
    return TrackedPerson(
        track_id=track_id,
        bbox=(0, 0, 400, 300),
        keypoints=_make_kpts(**kpt_kwargs),
        conf=0.9,
    )


def _run_sequence(behavior, track_id, keypoint_seq, fps=30.0):
    results = []
    for i, kpt_kwargs in enumerate(keypoint_seq):
        person = _make_person(track_id, **kpt_kwargs)
        results.append(behavior.detect_person(person, None, i, i / fps))
    return results


def _hold(x=250.0, y=140.0, n=15, **extra):
    return [{"wrist_l_x": x, "wrist_l_y": y, **extra} for _ in range(n)]


def test_no_zone_raises():
    try:
        HandSnatchObjectBehavior({})
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_hold_then_fast_exit_snatch_out():
    behavior = HandSnatchObjectBehavior({"snatch_velocity_ratio": 0.15}, zones=[_make_zone()], fps=30)
    seq = _hold(x=250.0, n=15) + [{"wrist_l_x": 350, "wrist_l_y": 140}]
    results = _run_sequence(behavior, 1, seq)
    assert all(not r.detected for r in results[:15]), "Holding alone must not fire"
    assert results[-1].detected, "Fast exit after hold should trigger snatch-out"
    assert results[-1].metadata["snatch_type"] == "snatch_out"
    assert results[-1].metadata["hand"] == "left"


def test_snatch_out_rich_metadata():
    behavior = HandSnatchObjectBehavior({"snatch_velocity_ratio": 0.15}, zones=[_make_zone()], fps=30)
    seq = _hold(x=250.0, n=15) + [{"wrist_l_x": 350, "wrist_l_y": 140}]
    results = _run_sequence(behavior, 1, seq)
    md = results[-1].metadata
    for key in ("hand", "side", "snatch_type", "exit_velocity", "hold_duration",
                "spike_ratio", "jerk_displacement_ratio", "peak_velocity",
                "zone", "triggered_zones", "fps"):
        assert key in md, f"metadata missing '{key}'"
    assert md["side"] == "left"
    assert md["zone"] == "TestZone"
    assert md["fps"] == 30.0
    assert md["exit_velocity"] > 4.0, "exit velocity in shoulder-widths/second"
    assert md["hold_duration"] > 0.4
    assert md["peak_velocity"] > 0
    assert md["jerk_displacement_ratio"] > 0


def test_hold_then_jerk_toward_body_snatch_in_pull():
    behavior = HandSnatchObjectBehavior({"snatch_velocity_ratio": 0.15}, zones=[_make_zone()], fps=30)
    seq = _hold(x=280.0, n=15) + [
        {"wrist_l_x": 230, "wrist_l_y": 140},
        {"wrist_l_x": 230, "wrist_l_y": 140},
    ]
    results = _run_sequence(behavior, 1, seq)
    assert not results[14].detected, "Still holding must not fire"
    assert not results[15].detected, "Displacement gate not met yet"
    assert results[16].detected, "Fast jerk toward body after hold should fire"
    assert results[16].metadata["snatch_type"] == "snatch_in_pull"


def test_hold_then_jerk_away_snatch_in():
    behavior = HandSnatchObjectBehavior({"snatch_velocity_ratio": 0.15}, zones=[_make_zone()], fps=30)
    seq = _hold(x=230.0, n=15) + [{"wrist_l_x": 290, "wrist_l_y": 140}]
    results = _run_sequence(behavior, 1, seq)
    assert not results[14].detected, "Still holding must not fire"
    assert results[15].detected, "Fast jerk away from body after hold should fire"
    assert results[15].metadata["snatch_type"] == "snatch_in"


def test_wandering_in_zone_never_fires():
    behavior = HandSnatchObjectBehavior({"snatch_velocity_ratio": 0.15}, zones=[_make_zone()], fps=30)
    # Di chuyển liên tục trong zone với tốc độ hỗn hợp, không có pha tĩnh ≥ 0.4 s
    seq = [
        {"wrist_l_x": 240, "wrist_l_y": 140},
        {"wrist_l_x": 241, "wrist_l_y": 140},
        {"wrist_l_x": 242, "wrist_l_y": 140},
        {"wrist_l_x": 260, "wrist_l_y": 140},
        {"wrist_l_x": 261, "wrist_l_y": 140},
        {"wrist_l_x": 262, "wrist_l_y": 140},
        {"wrist_l_x": 280, "wrist_l_y": 140},
        {"wrist_l_x": 281, "wrist_l_y": 140},
        {"wrist_l_x": 282, "wrist_l_y": 140},
        {"wrist_l_x": 230, "wrist_l_y": 140},
        {"wrist_l_x": 231, "wrist_l_y": 140},
        {"wrist_l_x": 232, "wrist_l_y": 140},
        {"wrist_l_x": 250, "wrist_l_y": 140},
        {"wrist_l_x": 251, "wrist_l_y": 140},
        {"wrist_l_x": 252, "wrist_l_y": 140},
        {"wrist_l_x": 380, "wrist_l_y": 140},
    ]
    results = _run_sequence(behavior, 1, seq)
    assert all(not r.detected for r in results), "Wandering without a still phase must never fire"


def test_body_translation_never_fires():
    behavior = HandSnatchObjectBehavior({"snatch_velocity_ratio": 0.15}, zones=[_make_zone()], fps=30)
    # Cả vai và cổ tay cùng tịnh tiến (người đi bộ): wrist đo tương đối với tâm
    # vai phải triệt tiêu toàn bộ chuyển động -> không bao giờ bắn
    seq = []
    for i in range(18):
        seq.append({
            "wrist_l_x": 250.0 + 3 * i,
            "shoulder_l": (200.0 + 3 * i, 100.0, 0.9),
            "shoulder_r": (260.0 + 3 * i, 100.0, 0.9),
        })
    results = _run_sequence(behavior, 1, seq)
    assert all(not r.detected for r in results), "Whole-body translation must never fire"


def test_slow_exit_after_hold_never_fires():
    behavior = HandSnatchObjectBehavior({"snatch_velocity_ratio": 0.15}, zones=[_make_zone()], fps=30)
    seq = _hold(x=290.0, n=15)
    for x in range(291, 303):
        seq.append({"wrist_l_x": x, "wrist_l_y": 140})
    results = _run_sequence(behavior, 1, seq)
    assert all(not r.detected for r in results), "Slow exit while holding is a 'take', not a snatch"


def test_pass_through_without_hold_never_fires():
    behavior = HandSnatchObjectBehavior({"snatch_velocity_ratio": 0.15}, zones=[_make_zone()], fps=30)
    seq = [
        {"wrist_l_x": 180, "wrist_l_y": 140},
        {"wrist_l_x": 240, "wrist_l_y": 140},
        {"wrist_l_x": 310, "wrist_l_y": 140},
        {"wrist_l_x": 380, "wrist_l_y": 140},
    ]
    results = _run_sequence(behavior, 1, seq)
    assert all(not r.detected for r in results), "Pass-through without hold must not fire"


def test_small_jerk_after_hold_never_fires():
    behavior = HandSnatchObjectBehavior({"snatch_velocity_ratio": 0.15}, zones=[_make_zone()], fps=30)
    seq = _hold(x=250.0, n=15) + [{"wrist_l_x": 244, "wrist_l_y": 140} for _ in range(6)]
    results = _run_sequence(behavior, 1, seq)
    assert all(not r.detected for r in results), "Tiny displacement jerk must not fire"


def test_low_confidence_never_fires_and_resets_state():
    behavior = HandSnatchObjectBehavior(
        {"snatch_velocity_ratio": 0.15, "keypoint_conf_threshold": 0.5},
        zones=[_make_zone()], fps=30,
    )
    seq = _hold(x=250.0, n=15)
    results = _run_sequence(behavior, 1, seq)
    assert all(not r.detected for r in results)

    # Mất dấu tay: reset toàn bộ trạng thái chuyển động của tay
    for i in range(15, 18):
        person = _make_person(1, wrist_l_x=250, wrist_l_conf=0.3)
        behavior.detect_person(person, None, i, i / 30.0)
    st = behavior._hand_state[(1, "left")]
    assert st.state == "IDLE", "Conf gap must reset the hand state machine"
    assert st.smoothed_rel is None, "Conf gap must drop the smoothed position"

    # Tay xuất hiện lại ở ngoài zone: không được bắn từ state cũ
    person = _make_person(1, wrist_l_x=330, wrist_l_y=140)
    result = behavior.detect_person(person, None, 18, 18 / 30.0)
    assert not result.detected, "Re-appearance after conf gap must not fire from stale state"


def test_outside_zone_not_detected():
    behavior = HandSnatchObjectBehavior({"snatch_velocity_ratio": 0.15}, zones=[_make_zone()], fps=30)
    seq = [
        {"wrist_l_x": 320, "wrist_l_y": 140},
        {"wrist_l_x": 350, "wrist_l_y": 140},
    ]
    results = _run_sequence(behavior, 1, seq)
    assert all(not r.detected for r in results), "Outside zone should not trigger"


def test_both_hands_independent():
    behavior = HandSnatchObjectBehavior({"snatch_velocity_ratio": 0.15}, zones=[_make_zone()], fps=30)
    seq = [{"wrist_l_x": 280, "wrist_l_y": 140, "wrist_r_x": 280, "wrist_r_y": 140}
           for _ in range(15)]
    seq.append({"wrist_l_x": 280, "wrist_l_y": 140, "wrist_r_x": 230, "wrist_r_y": 140})
    seq.append({"wrist_l_x": 280, "wrist_l_y": 140, "wrist_r_x": 230, "wrist_r_y": 140})
    results = _run_sequence(behavior, 1, seq)
    assert results[16].detected, "Right hand jerk should be detected"
    assert results[16].metadata["hand"] == "right"
    assert results[16].metadata["snatch_type"] == "snatch_in_pull"


def test_cooldown_blocks_immediate_refire():
    behavior = HandSnatchObjectBehavior(
        {"snatch_velocity_ratio": 0.15, "cooldown_seconds": 0.7,
         "min_event_frames": 1, "smoothing_tau": 0.01},
        zones=[_make_zone()], fps=30,
    )
    seq = _hold(x=230.0, n=15)
    seq.append({"wrist_l_x": 290, "wrist_l_y": 140})     # frame 15: bắn lần 1 (t=0.5)
    seq += _hold(x=290.0, n=15)                          # frames 16-30: giữ lại trong zone
    seq.append({"wrist_l_x": 230, "wrist_l_y": 140})     # frame 31: t=1.03 < cooldown 1.2
    seq += _hold(x=230.0, n=19)                          # frames 32-50: cửa sổ JERK hết hạn
    seq += _hold(x=230.0, n=14)                          # frames 51-64: giữ đủ lâu lần nữa
    seq.append({"wrist_l_x": 290, "wrist_l_y": 140})     # frame 65: t=2.17 > 1.2
    results = _run_sequence(behavior, 1, seq)
    assert results[15].detected, "First snatch-in should fire"
    assert not results[31].detected, "Refire inside cooldown must be blocked"
    assert results[65].detected, "Snatch after cooldown should fire"


def test_event_manager_sustain_completes_event():
    behavior = HandSnatchObjectBehavior(
        {"confirmation_frames": 10, "max_gap_frames": 3, "min_event_frames": 15},
        zones=[_make_zone()], fps=30,
    )
    events = []
    for i in range(34):
        # Frames 0-14: giữ yên trong zone; frame 15: giật ra xa -> bắn 1 frame
        wrist_x = 230.0 if i < 15 else 290.0
        person = _make_person(1, wrist_l_x=wrist_x, wrist_l_y=140.0)
        events.extend(behavior.process_frame([person], None, i, i / 30.0))
    # Frame 15: giật ra xa -> bắn; sustain giữ detected=True thêm 14 frame
    assert len(events) == 1, "A 1-frame snatch must still complete one Event"
    ev = events[0]
    assert ev.behavior_name == "hand_snatch_object"
    assert ev.track_id == 1
    assert len(ev.frames) == 15, "Event should span the fire frame plus sustained frames"
    assert ev.hand_sides == ["left"] * 15
    assert ev.max_confidence > 0


def test_low_fps_still_fires():
    behavior = HandSnatchObjectBehavior({"snatch_velocity_ratio": 0.15}, zones=[_make_zone()], fps=7)
    seq = _hold(x=250.0, n=4) + [{"wrist_l_x": 350, "wrist_l_y": 140}]
    results = _run_sequence(behavior, 1, seq, fps=7)
    assert results[-1].detected, "Hold 3 frames @7fps (~0.43s) then fast exit should fire"
    assert results[-1].metadata["snatch_type"] == "snatch_out"