import numpy as np

from detection.behaviors.head_turn import HeadTurnBehavior
from detection.detector import TrackedPerson
from detection.pose_utils import compute_head_yaw_offset


def _make_kpts(nose_x=100.0, nose_conf=0.9):
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = [nose_x, 45, nose_conf]
    kpts[3] = [80, 50, 0.9]
    kpts[4] = [120, 50, 0.9]
    return kpts


def _make_person(track_id, nose_x=100.0):
    return TrackedPerson(track_id=track_id, bbox=(0, 0, 100, 200), keypoints=_make_kpts(nose_x), conf=0.9)


def _run(behavior, track_id, nose_sequence):
    results = []
    for i, nose_x in enumerate(nose_sequence):
        person = _make_person(track_id, nose_x)
        results.append(behavior.detect_person(person, None, i, i / 30.0))
    return results


def test_compute_head_yaw_offset():
    offset, width = compute_head_yaw_offset(_make_kpts(100.0))
    assert width == 40.0
    assert offset == 0.0

    offset, _ = compute_head_yaw_offset(_make_kpts(88.0))
    assert round(offset, 6) == -0.3


def test_compute_head_yaw_offset_none():
    kpts = _make_kpts(nose_conf=0.1)
    assert compute_head_yaw_offset(kpts) is None


def test_head_turn_not_detected_below_threshold():
    behavior = HeadTurnBehavior({"turn_threshold_ratio": 0.25, "window_frames": 90, "max_turns": 3})
    results = _run(behavior, 1, [100, 80, 100, 120, 100, 80])
    assert all(not r.detected for r in results)


def test_head_turn_detected_after_four_turns():
    behavior = HeadTurnBehavior({"turn_threshold_ratio": 0.25, "window_frames": 90, "max_turns": 3})
    # nose x: center->left->center->right->center->left->center->right (4 turns)
    results = _run(behavior, 1, [100, 80, 100, 120, 100, 80, 100, 120])
    assert not results[0].detected
    assert results[5].metadata["turns"] == 3
    assert not results[5].detected
    assert results[-1].metadata["turns"] == 4
    assert results[-1].detected


def test_head_turn_window_expiry():
    behavior = HeadTurnBehavior({"turn_threshold_ratio": 0.25, "window_frames": 10, "max_turns": 3})
    # 4 turns then a long pause; old turns should slide out of the window
    results = _run(behavior, 1, [100, 80, 100, 120, 100, 80, 100, 120] + [100] * 17)
    assert results[7].metadata["turns"] == 4
    assert results[7].detected
    assert results[-1].metadata["turns"] == 0
    assert not results[-1].detected


def test_head_turn_metadata_side():
    behavior = HeadTurnBehavior({"turn_threshold_ratio": 0.25, "window_frames": 90, "max_turns": 3})
    results = _run(behavior, 2, [100, 80])
    assert results[1].metadata["side"] == "left"
    assert not results[1].detected
