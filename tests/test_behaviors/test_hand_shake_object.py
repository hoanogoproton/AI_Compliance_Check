import math

import numpy as np

from detection.behaviors.hand_shake_object import HandShakeObjectBehavior
from detection.detector import TrackedPerson
from detection.zones.zone_definition import Zone


def _make_zone(points=None):
    if points is None:
        points = [[230, 120], [270, 120], [270, 160], [230, 160]]
    return Zone(name="TestZone", label="TestZone", points=points)


def _make_kpts(wrist_l_x=250.0, wrist_l_y=140.0, wrist_l_conf=0.9,
               shoulder_l=(200, 100, 0.9), shoulder_r=(260, 100, 0.9)):
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[5] = shoulder_l
    kpts[6] = shoulder_r
    kpts[9] = [wrist_l_x, wrist_l_y, wrist_l_conf]
    return kpts


def _make_person(track_id, **kpt_kwargs):
    return TrackedPerson(
        track_id=track_id,
        bbox=(0, 0, 400, 300),
        keypoints=_make_kpts(**kpt_kwargs),
        conf=0.9,
    )


def _run_sequence(behavior, track_id, keypoint_seq):
    results = []
    for i, kpt_kwargs in enumerate(keypoint_seq):
        person = _make_person(track_id, **kpt_kwargs)
        results.append(behavior.detect_person(person, None, i, i / 30.0))
    return results


def test_no_zone_raises():
    try:
        HandShakeObjectBehavior({})
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_no_shake_outside_zone():
    behavior = HandShakeObjectBehavior({"window_frames": 40, "min_reversals": 3}, zones=[_make_zone()])
    seq = [
        {"wrist_l_x": 300, "wrist_l_y": 140},
        {"wrist_l_x": 305, "wrist_l_y": 140},
        {"wrist_l_x": 300, "wrist_l_y": 140},
        {"wrist_l_x": 305, "wrist_l_y": 140},
        {"wrist_l_x": 300, "wrist_l_y": 140},
        {"wrist_l_x": 305, "wrist_l_y": 140},
    ]
    results = _run_sequence(behavior, 1, seq)
    assert all(not r.detected for r in results), "Outside zone should not trigger"


def test_shake_inside_zone_detected():
    behavior = HandShakeObjectBehavior({"window_frames": 40, "min_reversals": 3}, zones=[_make_zone()])
    seq = [
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 250, "wrist_l_y": 140},
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 250, "wrist_l_y": 140},
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 250, "wrist_l_y": 140},
    ]
    results = _run_sequence(behavior, 1, seq)
    assert results[-1].detected, "Shaking inside zone should trigger"
    assert results[-1].metadata["wrist_reversals"] >= 3


def test_insufficient_reversals():
    behavior = HandShakeObjectBehavior({"window_frames": 40, "min_reversals": 3}, zones=[_make_zone()])
    seq = [
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 250, "wrist_l_y": 140},
        {"wrist_l_x": 245, "wrist_l_y": 140},
    ]
    results = _run_sequence(behavior, 1, seq)
    assert not results[-1].detected, "2 reversals should not trigger"


def test_low_confidence():
    behavior = HandShakeObjectBehavior({"window_frames": 40, "min_reversals": 3, "keypoint_conf_threshold": 0.5}, zones=[_make_zone()])
    seq = [
        {"wrist_l_x": 245, "wrist_l_y": 140, "wrist_l_conf": 0.3},
        {"wrist_l_x": 250, "wrist_l_y": 140, "wrist_l_conf": 0.3},
        {"wrist_l_x": 245, "wrist_l_y": 140, "wrist_l_conf": 0.3},
        {"wrist_l_x": 250, "wrist_l_y": 140, "wrist_l_conf": 0.3},
        {"wrist_l_x": 245, "wrist_l_y": 140, "wrist_l_conf": 0.3},
        {"wrist_l_x": 250, "wrist_l_y": 140, "wrist_l_conf": 0.3},
    ]
    results = _run_sequence(behavior, 1, seq)
    assert all(not r.detected for r in results), "Low confidence should not trigger"


def test_window_expiry():
    behavior = HandShakeObjectBehavior({"window_frames": 10, "min_reversals": 3}, zones=[_make_zone()])
    # Build 3+ reversals by oscillating x position
    seq = [
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 250, "wrist_l_y": 140},
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 250, "wrist_l_y": 140},
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 250, "wrist_l_y": 140},
    ]
    results = _run_sequence(behavior, 1, seq)
    assert results[5].detected, "Should detect at frame 5"

    seq2 = [{"wrist_l_x": 245, "wrist_l_y": 140} for _ in range(15)]
    results2 = _run_sequence(behavior, 1, seq2)
    assert not results2[-1].detected, "Old reversals should expire"


def test_hand_enters_zone_then_shakes():
    behavior = HandShakeObjectBehavior({"window_frames": 40, "min_reversals": 3}, zones=[_make_zone()])
    seq = [
        {"wrist_l_x": 300, "wrist_l_y": 140},
        {"wrist_l_x": 250, "wrist_l_y": 140},
        {"wrist_l_x": 255, "wrist_l_y": 140},
        {"wrist_l_x": 250, "wrist_l_y": 140},
        {"wrist_l_x": 255, "wrist_l_y": 140},
        {"wrist_l_x": 250, "wrist_l_y": 140},
        {"wrist_l_x": 255, "wrist_l_y": 140},
    ]
    results = _run_sequence(behavior, 1, seq)
    assert not results[0].detected
    assert results[-1].detected, "Should detect after entering zone and shaking"


# ---------------------------------------------------------------------------
# Robust (fps-aware) mode tests — video 7 fps
# ---------------------------------------------------------------------------

def _make_behavior_robust(fps=7.0, **overrides):
    params = {
        "window_seconds": 2.0,
        "min_reversals": 3,
        "keypoint_conf_threshold": 0.5,
        "confirmation_frames": 2,
        "max_gap_frames": 3,
        "min_event_frames": 7,
    }
    params.update(overrides)
    return HandShakeObjectBehavior(params, zones=[_make_zone()], fps=fps)


def _oscillation_seq(fps=7.0, freq=1.5, amplitude=8.0, duration=2.5,
                     center_x=250.0, y=140.0, phase=0.0, conf=0.9):
    n = int(round(duration * fps))
    return [
        {"wrist_l_x": center_x + amplitude * math.sin(2 * math.pi * freq * (i / fps) + phase),
         "wrist_l_y": y, "wrist_l_conf": conf}
        for i in range(n)
    ]


def _run_sequence_fps(behavior, track_id, keypoint_seq, fps=7.0):
    results = []
    for i, kpt_kwargs in enumerate(keypoint_seq):
        person = _make_person(track_id, **kpt_kwargs)
        results.append(behavior.detect_person(person, None, i, i / fps))
    return results


def test_mode_selection_and_fps():
    legacy = HandShakeObjectBehavior({"window_frames": 40}, zones=[_make_zone()])
    assert legacy._mode == "legacy"
    robust = HandShakeObjectBehavior({"window_seconds": 2.0}, zones=[_make_zone()], fps=7.0)
    assert robust._mode == "robust"
    assert robust.fps == 7.0
    default = HandShakeObjectBehavior({}, zones=[_make_zone()])
    assert default._mode == "robust"


def test_robust_7fps_shake_detected():
    behavior = _make_behavior_robust()
    results = _run_sequence_fps(behavior, 1, _oscillation_seq(duration=2.5))
    assert results[-1].detected, "1.5Hz shake at 7fps should trigger"
    freq = results[-1].metadata.get("frequency_hz")
    assert freq is not None and abs(freq - 1.5) < 0.5
    assert results[-1].metadata["amplitude_ratio"] > 0.05


def test_robust_7fps_slow_drift_not_detected():
    behavior = _make_behavior_robust()
    results = _run_sequence_fps(behavior, 1, _oscillation_seq(freq=0.2, duration=8.0))
    assert not any(r.detected for r in results), "slow drift below freq band should not trigger"


def test_robust_7fps_micro_tremor_not_detected():
    behavior = _make_behavior_robust()
    results = _run_sequence_fps(behavior, 1, _oscillation_seq(freq=2.5, amplitude=1.2, duration=4.0))
    assert not any(r.detected for r in results), "micro tremor should not trigger"


def test_robust_7fps_gap_then_recover():
    behavior = _make_behavior_robust()
    seq = _oscillation_seq(duration=1.5)
    seq += [{"wrist_l_x": 250.0, "wrist_l_y": 140.0, "wrist_l_conf": 0.1} for _ in range(5)]
    seq += _oscillation_seq(duration=1.5, phase=2 * math.pi * 1.5 * 1.5)
    results = _run_sequence_fps(behavior, 1, seq)
    # Jump lúc tái xuất (sau reset) không được tạo đảo chiều ảo: số reversal
    # ngay frame quay lại không được lớn hơn số reversal trước khi mất dấu
    before_gap = results[9].metadata.get("wrist_reversals", 0)
    at_reentry = results[15].metadata.get("wrist_reversals", 0)
    assert at_reentry <= before_gap, "re-entry jump must not create phantom reversals"
    assert results[-1].detected, "should detect again after shake resumes"


def test_robust_7fps_window_expiry():
    behavior = _make_behavior_robust()
    seq = _oscillation_seq(duration=2.5)
    seq += [{"wrist_l_x": 250.0, "wrist_l_y": 140.0} for _ in range(int(3.0 * 7))]
    results = _run_sequence_fps(behavior, 1, seq)
    assert any(r.detected for r in results)
    assert not results[-1].detected, "reversals should expire from time window"


def test_robust_max_frequency_gate():
    behavior = _make_behavior_robust(max_frequency_hz=1.0)
    results = _run_sequence_fps(behavior, 1, _oscillation_seq(freq=1.5, duration=3.0))
    assert not any(r.detected for r in results), "freq above max_frequency_hz should not trigger"


def test_robust_event_emitted_at_7fps():
    behavior = _make_behavior_robust()
    seq = _oscillation_seq(duration=2.2)
    seq += [{"wrist_l_x": 300.0, "wrist_l_y": 140.0} for _ in range(6)]
    events = []
    for i, kpt_kwargs in enumerate(seq):
        person = _make_person(1, **kpt_kwargs)
        events.extend(behavior.process_frame([person], None, i, i / 7.0))
    assert events, "event should be emitted at 7fps with min_event_frames=7"
    assert len(events[0].frames) >= 7
