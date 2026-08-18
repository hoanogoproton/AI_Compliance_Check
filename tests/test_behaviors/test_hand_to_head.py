import numpy as np

from detection.behaviors.hand_to_head import HandToHeadBehavior
from detection.detector import TrackedPerson


def _make_kpts(arr_17x3):
    return np.array(arr_17x3, dtype=np.float32)


def _make_person(track_id, kpts):
    return TrackedPerson(track_id=track_id, bbox=(0, 0, 100, 200), keypoints=kpts, conf=0.9)


def test_hand_to_head_detected_left():
    kpts = _make_kpts([[0, 0, 0]] * 17)
    kpts[3] = [80, 50, 0.9]
    kpts[4] = [120, 50, 0.9]
    kpts[5] = [60, 80, 0.9]
    kpts[6] = [140, 80, 0.9]
    kpts[9] = [95, 55, 0.9]
    behavior = HandToHeadBehavior({"distance_threshold_ratio": 0.25, "vertical_offset_ratio": 0.2, "keypoint_conf_threshold": 0.5, "head_keypoint_conf_threshold": 0.5})
    person = _make_person(1, kpts)
    result = behavior.detect_person(person, None, 0, 0.0)
    assert result.detected
    assert result.metadata["side"] == "left"


def test_hand_to_head_not_detected():
    kpts = _make_kpts([[0, 0, 0]] * 17)
    kpts[3] = [80, 50, 0.9]
    kpts[4] = [120, 50, 0.9]
    kpts[5] = [60, 80, 0.9]
    kpts[6] = [140, 80, 0.9]
    kpts[9] = [400, 400, 0.9]
    behavior = HandToHeadBehavior({"distance_threshold_ratio": 0.25, "vertical_offset_ratio": 0.2, "keypoint_conf_threshold": 0.5, "head_keypoint_conf_threshold": 0.5})
    person = _make_person(1, kpts)
    result = behavior.detect_person(person, None, 0, 0.0)
    assert not result.detected
    assert result.metadata["side"] == "none"


def test_hand_to_head_too_small():
    kpts = _make_kpts([[0, 0, 0]] * 17)
    kpts[5] = [99, 0, 0.9]
    kpts[6] = [100, 0, 0.9]
    behavior = HandToHeadBehavior({"distance_threshold_ratio": 0.9, "vertical_offset_ratio": 0.2, "keypoint_conf_threshold": 0.5, "head_keypoint_conf_threshold": 0.5})
    person = _make_person(1, kpts)
    result = behavior.detect_person(person, None, 0, 0.0)
    assert not result.detected


def test_hand_to_head_both():
    kpts = _make_kpts([[0, 0, 0]] * 17)
    kpts[3] = [80, 50, 0.9]
    kpts[4] = [120, 50, 0.9]
    kpts[5] = [60, 80, 0.9]
    kpts[6] = [140, 80, 0.9]
    kpts[9] = [95, 55, 0.9]
    kpts[10] = [105, 55, 0.9]
    behavior = HandToHeadBehavior({"distance_threshold_ratio": 0.25, "vertical_offset_ratio": 0.2, "keypoint_conf_threshold": 0.5, "head_keypoint_conf_threshold": 0.5})
    person = _make_person(1, kpts)
    result = behavior.detect_person(person, None, 0, 0.0)
    assert result.detected
    assert result.metadata["side"] == "both"
