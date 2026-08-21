import numpy as np

from detection.behaviors.hand_snatch_object import HandSnatchObjectBehavior
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
        HandSnatchObjectBehavior({})
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_snatch_out_detected():
    behavior = HandSnatchObjectBehavior(
        {"min_grasp_frames": 3, "snatch_velocity_ratio": 0.15, "approach_window": 10},
        zones=[_make_zone()],
    )
    seq = [
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 250, "wrist_l_y": 140},
        {"wrist_l_x": 350, "wrist_l_y": 140},
    ]
    results = _run_sequence(behavior, 1, seq)
    assert results[-1].detected, "Fast exit should trigger snatch-out"
    assert results[-1].metadata["snatch_type"] == "snatch_out"


def test_slow_exit_not_detected():
    behavior = HandSnatchObjectBehavior(
        {"min_grasp_frames": 3, "snatch_velocity_ratio": 0.15, "approach_window": 10},
        zones=[_make_zone()],
    )
    seq = [
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 246, "wrist_l_y": 140},
        {"wrist_l_x": 247, "wrist_l_y": 140},
        {"wrist_l_x": 248, "wrist_l_y": 140},
        {"wrist_l_x": 249, "wrist_l_y": 140},
        {"wrist_l_x": 250, "wrist_l_y": 140},
    ]
    results = _run_sequence(behavior, 1, seq)
    assert not results[-1].detected, "Slow exit should not trigger snatch"


def test_pass_through_not_detected():
    behavior = HandSnatchObjectBehavior(
        {"min_grasp_frames": 3, "snatch_velocity_ratio": 0.15, "approach_window": 10},
        zones=[_make_zone()],
    )
    seq = [
        {"wrist_l_x": 300, "wrist_l_y": 140},
        {"wrist_l_x": 250, "wrist_l_y": 140},
        {"wrist_l_x": 350, "wrist_l_y": 140},
    ]
    results = _run_sequence(behavior, 1, seq)
    assert not results[-1].detected, "Pass-through (< min_grasp_frames) should not trigger"


def test_snatch_in_detected():
    behavior = HandSnatchObjectBehavior(
        {"min_grasp_frames": 3, "snatch_velocity_ratio": 0.15, "approach_window": 10},
        zones=[_make_zone()],
    )
    seq = [
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 245, "wrist_l_y": 140},
        {"wrist_l_x": 235, "wrist_l_y": 140},
    ]
    results = _run_sequence(behavior, 1, seq)
    assert results[-1].detected, "Fast movement inside zone should trigger snatch-in"


def test_low_confidence():
    behavior = HandSnatchObjectBehavior(
        {"min_grasp_frames": 3, "snatch_velocity_ratio": 0.15,
         "approach_window": 10, "keypoint_conf_threshold": 0.5},
        zones=[_make_zone()],
    )
    seq = [
        {"wrist_l_x": 245, "wrist_l_y": 140, "wrist_l_conf": 0.3},
        {"wrist_l_x": 245, "wrist_l_y": 140, "wrist_l_conf": 0.3},
        {"wrist_l_x": 245, "wrist_l_y": 140, "wrist_l_conf": 0.3},
        {"wrist_l_x": 245, "wrist_l_y": 140, "wrist_l_conf": 0.3},
        {"wrist_l_x": 350, "wrist_l_y": 140, "wrist_l_conf": 0.3},
    ]
    results = _run_sequence(behavior, 1, seq)
    assert all(not r.detected for r in results), "Low confidence should not trigger"


def test_outside_zone_not_detected():
    behavior = HandSnatchObjectBehavior(
        {"min_grasp_frames": 3, "snatch_velocity_ratio": 0.15, "approach_window": 10},
        zones=[_make_zone()],
    )
    seq = [
        {"wrist_l_x": 300, "wrist_l_y": 140},
        {"wrist_l_x": 350, "wrist_l_y": 140},
    ]
    results = _run_sequence(behavior, 1, seq)
    assert all(not r.detected for r in results), "Outside zone should not trigger"


def test_both_hands_independent():
    behavior = HandSnatchObjectBehavior(
        {"min_grasp_frames": 3, "snatch_velocity_ratio": 0.15, "approach_window": 10},
        zones=[_make_zone()],
    )

    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[5] = (200, 100, 0.9)
    kpts[6] = (260, 100, 0.9)
    kpts[9] = (245, 140, 0.9)
    kpts[10] = (245, 140, 0.9)

    person = TrackedPerson(
        track_id=1, bbox=(0, 0, 400, 300), keypoints=kpts, conf=0.9,
    )
    behavior.detect_person(person, None, 0, 0.0)
    behavior.detect_person(person, None, 1, 1 / 30.0)
    behavior.detect_person(person, None, 2, 2 / 30.0)
    behavior.detect_person(person, None, 3, 3 / 30.0)

    kpts2 = np.zeros((17, 3), dtype=np.float32)
    kpts2[5] = (200, 100, 0.9)
    kpts2[6] = (260, 100, 0.9)
    kpts2[9] = (245, 140, 0.9)
    kpts2[10] = (350, 140, 0.9)
    person2 = TrackedPerson(
        track_id=1, bbox=(0, 0, 400, 300), keypoints=kpts2, conf=0.9,
    )
    result = behavior.detect_person(person2, None, 4, 4 / 30.0)
    assert result.detected, "Right hand snatch should be detected"
    assert result.metadata["hand"] == "right", "Should be right hand"