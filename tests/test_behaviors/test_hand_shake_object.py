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
